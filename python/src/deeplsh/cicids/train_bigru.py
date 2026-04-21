import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Dense, Embedding, GRU, GlobalMaxPooling1D, Input, Lambda, LayerNormalization, Multiply
from tensorflow.keras.models import Model

from deeplsh._paths import cicids_artifacts_dir
from deeplsh.cicids.pipeline import (
    build_sequence_matrix,
    default_processed_data_dir,
    default_raw_data_dir,
    load_metadata,
    load_prepared_pairs,
    load_prepared_token_flows,
    load_vocab,
    prepared_paths,
    prepare_cicids_dataset,
)
from deeplsh.core.deep_hashing_models import (
    intermediate_model_trained,
    predict_with_tqdm,
    siamese_contrastive_model,
    siamese_model,
    train_siamese_contrastive_model_with_warmup,
    train_siamese_model,
)
from deeplsh.core.lsh_search import convert_to_hamming, create_hash_tables, lsh_hyperparams


def _normalize_max_samples(max_samples):
    if max_samples is None:
        return None
    if max_samples <= 0:
        return None
    return max_samples


def _build_bigru_encoder(
    vocab_size: int,
    max_length: int,
    embed_dim: int,
    gru_units: int,
    dense_dim: int,
    size_hash_vector: int,
    attention_pooling: bool = True,
    layer_norm: bool = True,
    batch_norm: bool = False,
) -> Model:
    input_tensor = Input(shape=(max_length,), name="log_tokens")
    x = Embedding(vocab_size, embed_dim, mask_zero=True, name="token_embedding")(input_tensor)
    x = Bidirectional(GRU(gru_units, return_sequences=True), name="bigru")(x)
    if attention_pooling:
        attention_scores = Dense(1, name="attention_score")(x)
        attention_weights = Lambda(lambda scores: tf.nn.softmax(scores, axis=1), name="attention_weights")(attention_scores)
        x = Multiply(name="attention_weighted_tokens")([x, attention_weights])
        x = Lambda(lambda tensor: tf.reduce_sum(tensor, axis=1), name="attention_pool")(x)
    else:
        x = GlobalMaxPooling1D(name="token_pool")(x)
    if layer_norm:
        x = LayerNormalization(name="post_pool_layer_norm")(x)
    x = Dense(dense_dim, activation="relu", name="semantic_dense")(x)
    if batch_norm:
        x = BatchNormalization(name="semantic_batch_norm")(x)
    elif layer_norm:
        x = LayerNormalization(name="semantic_layer_norm")(x)
    output = Dense(size_hash_vector, activation="tanh", name="hash_projection")(x)
    return Model(inputs=input_tensor, outputs=output, name="cicids_bigru_encoder")


def _prepare_if_needed(args):
    paths = prepared_paths(args.output_dir)
    required = [paths["flows"], paths["flows_tokens"], paths["pairs"], paths["metadata"], paths["preprocessor"], paths["vocab"]]
    should_prepare = args.force_prepare or not all(os.path.exists(path) for path in required)
    if not should_prepare and os.path.exists(paths["metadata"]):
        with open(paths["metadata"], "r", encoding="utf-8") as f:
            metadata = json.load(f)
        should_prepare = (
            metadata.get("negative_strategy", "random") != args.negative_strategy
            or float(metadata.get("hard_negative_min_jaccard", 0.3)) != float(args.hard_negative_min_jaccard)
            or float(metadata.get("hard_negative_max_jaccard", 0.5)) != float(args.hard_negative_max_jaccard)
        )
    if should_prepare:
        prepare_cicids_dataset(
            data_dir=args.data_repo,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            max_pairs=args.max_pairs,
            seed=args.seed,
            negative_strategy=args.negative_strategy,
            hard_negative_min_jaccard=args.hard_negative_min_jaccard,
            hard_negative_max_jaccard=args.hard_negative_max_jaccard,
        )


def _write_hash_diagnostics(embeddings_hamming: np.ndarray, pairs_df, results_dir: str) -> dict:
    os.makedirs(results_dir, exist_ok=True)
    positive_rate = np.mean(embeddings_hamming > 0, axis=0)
    eps = 1e-12
    entropy = -(positive_rate * np.log2(positive_rate + eps) + (1.0 - positive_rate) * np.log2(1.0 - positive_rate + eps))
    bit_df = {
        "bit_index": np.arange(embeddings_hamming.shape[1], dtype=int),
        "mean_value": np.mean(embeddings_hamming, axis=0),
        "positive_rate": positive_rate,
        "entropy": entropy,
    }
    bit_path = os.path.join(results_dir, "hash_diagnostics_bigru.csv")
    import pandas as pd

    pd.DataFrame(bit_df).to_csv(bit_path, index=False)

    idx1 = pairs_df["flow_index_1"].to_numpy(dtype=int)
    idx2 = pairs_df["flow_index_2"].to_numpy(dtype=int)
    pair_sims = np.mean(embeddings_hamming[idx1] == embeddings_hamming[idx2], axis=1)
    duplicate_mask = pairs_df["is_duplicate"].to_numpy(dtype=int) == 1
    non_duplicate_sims = pair_sims[~duplicate_mask]
    duplicate_sims = pair_sims[duplicate_mask]
    summary = {
        "duplicate_count": int(duplicate_sims.size),
        "duplicate_pred_sim_mean": float(np.mean(duplicate_sims)) if duplicate_sims.size else 0.0,
        "duplicate_pred_sim_median": float(np.median(duplicate_sims)) if duplicate_sims.size else 0.0,
        "non_duplicate_count": int(non_duplicate_sims.size),
        "non_duplicate_pred_sim_mean": float(np.mean(non_duplicate_sims)) if non_duplicate_sims.size else 0.0,
        "non_duplicate_pred_sim_median": float(np.median(non_duplicate_sims)) if non_duplicate_sims.size else 0.0,
        "non_duplicate_pred_sim_ge_0_9_count": int(np.sum(non_duplicate_sims >= 0.9)),
        "mean_bit_entropy": float(np.mean(entropy)),
        "low_entropy_bit_count": int(np.sum(entropy < 0.1)),
        "low_entropy_bit_ratio": float(np.mean(entropy < 0.1)),
    }
    summary["collapse_warning"] = bool(summary["mean_bit_entropy"] < 0.3 or summary["low_entropy_bit_ratio"] > 0.25)
    summary_path = os.path.join(results_dir, "hash_pair_similarity_diagnostics_bigru.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    summary["hash_diagnostics_path"] = bit_path
    summary["pair_diagnostics_path"] = summary_path
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train Bi-GRU + DeepLSH on CIC-IDS-2017 tokenized flow logs.")
    parser.add_argument("--data-repo", default=None, help="Path to CIC-IDS raw CSV directory (default: datasets/cicids/raw)")
    parser.add_argument("--output-dir", default=None, help="Prepared data directory")
    parser.add_argument("--max-samples", type=int, default=12000, help="Maximum raw flows to read when preparing; use 0 for all rows")
    parser.add_argument("--max-pairs", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--b", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lsh-param-index", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--gru-units", type=int, default=64)
    parser.add_argument("--dense-dim", type=int, default=128)
    parser.add_argument("--loss-type", choices=["mse", "contrastive"], default="contrastive")
    parser.add_argument("--margin", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--quantization-warmup-epochs", type=int, default=3)
    parser.add_argument("--quantization-weight", type=float, default=None, help="Deprecated alias for --gamma")
    parser.add_argument("--balance-weight", type=float, default=None, help="Deprecated alias for --beta")
    parser.add_argument("--negative-strategy", choices=["random", "hard"], default="hard")
    parser.add_argument("--hard-negative-min-jaccard", type=float, default=0.3)
    parser.add_argument("--hard-negative-max-jaccard", type=float, default=0.5)
    parser.add_argument("--attention-pooling", dest="attention_pooling", action="store_true", default=True)
    parser.add_argument("--no-attention-pooling", dest="attention_pooling", action="store_false")
    parser.add_argument("--layer-norm", dest="layer_norm", action="store_true", default=True)
    parser.add_argument("--no-layer-norm", dest="layer_norm", action="store_false")
    parser.add_argument("--batch-norm", dest="batch_norm", action="store_true", default=False)
    parser.add_argument("--no-batch-norm", dest="batch_norm", action="store_false")
    parser.add_argument("--force-prepare", action="store_true")
    args = parser.parse_args()

    effective_beta = args.beta if args.beta is not None else (args.balance_weight if args.balance_weight is not None else 0.1)
    effective_gamma = args.gamma if args.gamma is not None else (args.quantization_weight if args.quantization_weight is not None else 0.01)

    if args.data_repo is None:
        args.data_repo = default_raw_data_dir()
    if args.output_dir is None:
        args.output_dir = default_processed_data_dir()
    args.max_samples = _normalize_max_samples(args.max_samples)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    _prepare_if_needed(args)

    token_flows_df = load_prepared_token_flows(args.output_dir)
    pairs_df = load_prepared_pairs(args.output_dir)
    metadata = load_metadata(args.output_dir)
    vocab = load_vocab(args.output_dir)

    sequence_matrix = build_sequence_matrix(token_flows_df, vocab, max_length=vocab["max_sequence_length"])
    pair_indices = pairs_df.index.to_numpy()
    stratify = pairs_df["is_duplicate"] if pairs_df["is_duplicate"].nunique() > 1 else None
    train_idx, validation_idx = train_test_split(
        pair_indices,
        test_size=0.2,
        random_state=args.seed,
        stratify=stratify,
    )

    def build_pair_inputs(indices):
        subset = pairs_df.iloc[indices]
        y_values = subset["is_duplicate"].to_numpy(dtype=np.float32) if args.loss_type == "contrastive" else subset["target_similarity"].to_numpy(dtype=np.float32)
        return {
            "stack_1": sequence_matrix[subset["flow_index_1"].to_numpy(dtype=int)],
            "stack_2": sequence_matrix[subset["flow_index_2"].to_numpy(dtype=int)],
        }, y_values

    X_train, Y_train = build_pair_inputs(train_idx)
    X_validation, Y_validation = build_pair_inputs(validation_idx)

    size_hash_vector = args.m * args.b
    shared_model = _build_bigru_encoder(
        vocab_size=max(vocab["token_to_id"].values(), default=1) + 1,
        max_length=sequence_matrix.shape[1],
        embed_dim=args.embed_dim,
        gru_units=args.gru_units,
        dense_dim=args.dense_dim,
        size_hash_vector=size_hash_vector,
        attention_pooling=args.attention_pooling,
        layer_norm=args.layer_norm,
        batch_norm=args.batch_norm,
    )
    if args.loss_type == "contrastive":
        def build_contrastive_wrapper(gamma_value):
            return siamese_contrastive_model(
                shared_model,
                input_shape=(sequence_matrix.shape[1],),
                b=args.b,
                m=args.m,
                margin=args.margin,
                alpha=args.alpha,
                beta=effective_beta,
                gamma=gamma_value,
                is_sparse=False,
                print_summary=False,
            )

        warmup_epochs = min(max(args.quantization_warmup_epochs, 0), args.epochs)
        if warmup_epochs > 0:
            model = build_contrastive_wrapper(0.0)
            train_siamese_contrastive_model_with_warmup(
                model,
                X_train,
                X_validation,
                Y_train,
                Y_validation,
                size_hash_vector=size_hash_vector,
                batch_size=args.batch_size,
                epochs=warmup_epochs,
                quantization_warmup_epochs=0,
                progress_desc="Bi-GRU warmup",
            )
        remaining_epochs = args.epochs - warmup_epochs
        if remaining_epochs > 0:
            model = build_contrastive_wrapper(effective_gamma)
            train_siamese_contrastive_model_with_warmup(
                model,
                X_train,
                X_validation,
                Y_train,
                Y_validation,
                size_hash_vector=size_hash_vector,
                batch_size=args.batch_size,
                epochs=remaining_epochs,
                quantization_warmup_epochs=0,
                progress_desc="Bi-GRU train",
            )
    else:
        model = siamese_model(
            shared_model,
            input_shape=(sequence_matrix.shape[1],),
            b=args.b,
            m=args.m,
            is_sparse=False,
            print_summary=False,
        )
        train_siamese_model(
            model,
            X_train,
            X_validation,
            Y_train,
            Y_validation,
            batch_size=args.batch_size,
            epochs=args.epochs,
            progress_desc="Bi-GRU train",
        )

    intermediate_model = intermediate_model_trained(shared_model, output_layer=-1, CNN=False)
    embeddings = predict_with_tqdm(intermediate_model, sequence_matrix, batch_size=args.batch_size, desc="Bi-GRU embeddings")
    print("stage=bigru_convert_hamming status=start", flush=True)
    embeddings_hamming = convert_to_hamming(embeddings)
    print("stage=bigru_convert_hamming status=done", flush=True)

    params = lsh_hyperparams(args.m)
    if args.lsh_param_index < 0 or args.lsh_param_index >= len(params):
        raise ValueError(f"--lsh-param-index out of range: {args.lsh_param_index} (len={len(params)})")
    L, K = params[args.lsh_param_index]
    hash_tables = create_hash_tables(L, K, args.b, embeddings_hamming)

    base = cicids_artifacts_dir()
    models_dir = os.path.join(str(base), "models")
    hash_tables_dir = os.path.join(str(base), "hash_tables")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(hash_tables_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "model-deep-lsh-cicids-bigru.model")
    hash_tables_path = os.path.join(hash_tables_dir, "hash_tables_deeplsh_cicids_bigru.pkl")
    embeddings_path = os.path.join(models_dir, "cicids_bigru_embeddings.npy")
    embeddings_hamming_path = os.path.join(models_dir, "cicids_bigru_embeddings_hamming.npy")
    sequence_matrix_path = os.path.join(models_dir, "cicids_bigru_sequences.npy")
    corpus_path = os.path.join(models_dir, "cicids_tokens.csv")
    train_meta_path = os.path.join(models_dir, "cicids_bigru_train_metadata.json")

    print("stage=bigru_save_artifacts status=start", flush=True)
    intermediate_model.save(model_path)
    with open(hash_tables_path, "wb") as f:
        pickle.dump(hash_tables, f)
    np.save(embeddings_path, embeddings)
    np.save(embeddings_hamming_path, embeddings_hamming)
    np.save(sequence_matrix_path, sequence_matrix)
    token_flows_df.to_csv(corpus_path, index=False)
    results_dir = os.path.join(str(base), "results", Path(args.output_dir).name)
    os.makedirs(results_dir, exist_ok=True)
    validation_predictions = model.predict(
        [X_validation["stack_1"], X_validation["stack_2"]],
        batch_size=args.batch_size,
        verbose=0,
    )
    if isinstance(validation_predictions, (list, tuple)):
        validation_predictions = validation_predictions[0]
    validation_df = pairs_df.iloc[validation_idx].copy()
    validation_df["pred_sim"] = np.asarray(validation_predictions).reshape(-1)
    validation_df.to_csv(os.path.join(results_dir, "pairs_validation_bigru.csv"), index=False)
    diagnostics = _write_hash_diagnostics(embeddings_hamming, pairs_df, results_dir=results_dir)
    print("stage=bigru_save_artifacts status=done", flush=True)

    train_meta = {
        "model_type": "bigru",
        "data_repo": args.data_repo,
        "prepared_output_dir": args.output_dir,
        "selected_features": vocab["selected_features"],
        "max_sequence_length": int(sequence_matrix.shape[1]),
        "vocab_size": int(max(vocab["token_to_id"].values(), default=1) + 1),
        "n_flows": int(token_flows_df.shape[0]),
        "n_pairs": int(pairs_df.shape[0]),
        "lsh": {"L": int(L), "K": int(K), "b": int(args.b), "m": int(args.m)},
        "embed_dim": int(args.embed_dim),
        "gru_units": int(args.gru_units),
        "dense_dim": int(args.dense_dim),
        "loss_type": args.loss_type,
        "margin": float(args.margin),
        "alpha": float(args.alpha),
        "beta": float(effective_beta),
        "gamma": float(effective_gamma),
        "quantization_warmup_epochs": int(args.quantization_warmup_epochs),
        "quantization_weight": float(effective_gamma),
        "balance_weight": float(effective_beta),
        "negative_strategy": args.negative_strategy,
        "hard_negative_min_jaccard": float(args.hard_negative_min_jaccard),
        "hard_negative_max_jaccard": float(args.hard_negative_max_jaccard),
        "attention_pooling": bool(args.attention_pooling),
        "layer_norm": bool(args.layer_norm),
        "batch_norm": bool(args.batch_norm),
        "collapse_warning": bool(diagnostics["collapse_warning"]),
        "mean_bit_entropy": float(diagnostics["mean_bit_entropy"]),
        "low_entropy_bit_count": int(diagnostics["low_entropy_bit_count"]),
        "low_entropy_bit_ratio": float(diagnostics["low_entropy_bit_ratio"]),
        "hash_diagnostics_path": diagnostics["hash_diagnostics_path"],
        "pair_diagnostics_path": diagnostics["pair_diagnostics_path"],
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "label_counts": metadata["label_counts"],
    }
    with open(train_meta_path, "w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=2)

    print(
        " ".join(
            [
                "done",
                "dataset=CIC-IDS-2017",
                "model=bigru-deeplsh",
                f"n_flows={token_flows_df.shape[0]}",
                f"n_pairs={pairs_df.shape[0]}",
                f"max_length={sequence_matrix.shape[1]}",
                f"vocab_size={train_meta['vocab_size']}",
                f"size_hash_vector={size_hash_vector}",
                f"lsh=(L={L},K={K})",
                f"loss_type={args.loss_type}",
                f"alpha={args.alpha:.6f}",
                f"beta={effective_beta:.6f}",
                f"gamma={effective_gamma:.6f}",
                f"quantization_warmup_epochs={args.quantization_warmup_epochs}",
                f"negative_strategy={args.negative_strategy}",
                f"mean_bit_entropy={diagnostics['mean_bit_entropy']:.6f}",
                f"low_entropy_bit_count={diagnostics['low_entropy_bit_count']}",
                f"collapse_warning={diagnostics['collapse_warning']}",
                f"non_duplicate_pred_sim_median={diagnostics['non_duplicate_pred_sim_median']:.6f}",
                f"non_duplicate_pred_sim_ge_0_9_count={diagnostics['non_duplicate_pred_sim_ge_0_9_count']}",
                f"model_path={model_path}",
                f"hash_tables_path={hash_tables_path}",
            ]
        )
    )


if __name__ == "__main__":
    main()
