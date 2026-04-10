import argparse
import json
import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Bidirectional, Dense, Embedding, GRU, GlobalMaxPooling1D, Input
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
from deeplsh.core.deep_hashing_models import intermediate_model_trained, siamese_model, train_siamese_model
from deeplsh.core.lsh_search import convert_to_hamming, create_hash_tables, lsh_hyperparams


def _build_bigru_encoder(vocab_size: int, max_length: int, embed_dim: int, gru_units: int, dense_dim: int, size_hash_vector: int) -> Model:
    input_tensor = Input(shape=(max_length,), name="log_tokens")
    x = Embedding(vocab_size, embed_dim, mask_zero=True, name="token_embedding")(input_tensor)
    x = Bidirectional(GRU(gru_units, return_sequences=True), name="bigru")(x)
    x = GlobalMaxPooling1D(name="token_pool")(x)
    x = Dense(dense_dim, activation="relu", name="semantic_dense")(x)
    output = Dense(size_hash_vector, activation="tanh", name="hash_projection")(x)
    return Model(inputs=input_tensor, outputs=output, name="cicids_bigru_encoder")


def _prepare_if_needed(args):
    paths = prepared_paths(args.output_dir)
    required = [paths["flows"], paths["flows_tokens"], paths["pairs"], paths["metadata"], paths["preprocessor"], paths["vocab"]]
    if args.force_prepare or not all(os.path.exists(path) for path in required):
        prepare_cicids_dataset(
            data_dir=args.data_repo,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            max_pairs=args.max_pairs,
            seed=args.seed,
        )


def main():
    parser = argparse.ArgumentParser(description="Train Bi-GRU + DeepLSH on CIC-IDS-2017 tokenized flow logs.")
    parser.add_argument("--data-repo", default=None, help="Path to CIC-IDS raw CSV directory (default: datasets/cicids/raw)")
    parser.add_argument("--output-dir", default=None, help="Prepared data directory")
    parser.add_argument("--max-samples", type=int, default=12000)
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
    parser.add_argument("--force-prepare", action="store_true")
    args = parser.parse_args()

    if args.data_repo is None:
        args.data_repo = default_raw_data_dir()
    if args.output_dir is None:
        args.output_dir = default_processed_data_dir()

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
        return {
            "stack_1": sequence_matrix[subset["flow_index_1"].to_numpy(dtype=int)],
            "stack_2": sequence_matrix[subset["flow_index_2"].to_numpy(dtype=int)],
        }, subset["target_similarity"].to_numpy(dtype=np.float32)

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
    )
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
    )

    intermediate_model = intermediate_model_trained(shared_model, output_layer=-1, CNN=False)
    embeddings = intermediate_model.predict(sequence_matrix, verbose=0)
    embeddings_hamming = convert_to_hamming(embeddings)

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

    intermediate_model.save(model_path)
    with open(hash_tables_path, "wb") as f:
        pickle.dump(hash_tables, f)
    np.save(embeddings_path, embeddings)
    np.save(embeddings_hamming_path, embeddings_hamming)
    np.save(sequence_matrix_path, sequence_matrix)
    token_flows_df.to_csv(corpus_path, index=False)

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
                f"model_path={model_path}",
                f"hash_tables_path={hash_tables_path}",
            ]
        )
    )


if __name__ == "__main__":
    main()
