import argparse
import json
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from deeplsh._paths import cicids_artifacts_dir
from deeplsh.cicids.pipeline import (
    default_processed_data_dir,
    default_raw_data_dir,
    load_metadata,
    load_prepared_flows,
    load_prepared_pairs,
    prepared_paths,
    prepare_cicids_dataset,
)
from deeplsh.core.deep_hashing_models import intermediate_model_trained, siamese_model, train_siamese_model
from deeplsh.core.lsh_search import convert_to_hamming, create_hash_tables, lsh_hyperparams


def _build_encoder(input_dim: int, hidden_dims, size_hash_vector: int) -> Model:
    input_tensor = Input(shape=(input_dim,), name="flow_features")
    x = input_tensor
    for layer_index, units in enumerate(hidden_dims):
        x = Dense(units, activation="relu", name=f"dense_{layer_index + 1}")(x)
    output = Dense(size_hash_vector, activation="tanh", name="hash_projection")(x)
    return Model(inputs=input_tensor, outputs=output, name="cicids_encoder")


def _prepare_if_needed(args):
    paths = prepared_paths(args.output_dir)
    if args.force_prepare or not all(os.path.exists(path) for path in paths.values()):
        prepare_cicids_dataset(
            data_dir=args.data_repo,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            max_pairs=args.max_pairs,
            seed=args.seed,
        )


def main():
    parser = argparse.ArgumentParser(description="Train DeepLSH on CIC-IDS-2017 numeric flow features.")
    parser.add_argument("--data-repo", default=None, help="Path to CIC-IDS raw CSV directory (default: datasets/cicids/raw)")
    parser.add_argument("--output-dir", default=None, help="Prepared data directory")
    parser.add_argument("--max-samples", type=int, default=12000)
    parser.add_argument("--max-pairs", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--b", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lsh-param-index", type=int, default=2)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--force-prepare", action="store_true")
    args = parser.parse_args()

    if args.data_repo is None:
        args.data_repo = default_raw_data_dir()
    if args.output_dir is None:
        args.output_dir = default_processed_data_dir()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    _prepare_if_needed(args)

    flows_df = load_prepared_flows(args.output_dir)
    pairs_df = load_prepared_pairs(args.output_dir)
    metadata = load_metadata(args.output_dir)
    feature_columns = metadata["feature_columns"]

    feature_matrix = flows_df[feature_columns].to_numpy(dtype=np.float32)
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
            "stack_1": feature_matrix[subset["flow_index_1"].to_numpy(dtype=int)],
            "stack_2": feature_matrix[subset["flow_index_2"].to_numpy(dtype=int)],
        }, subset["target_similarity"].to_numpy(dtype=np.float32)

    X_train, Y_train = build_pair_inputs(train_idx)
    X_validation, Y_validation = build_pair_inputs(validation_idx)

    size_hash_vector = args.m * args.b
    shared_model = _build_encoder(feature_matrix.shape[1], args.hidden_dims, size_hash_vector)
    model = siamese_model(
        shared_model,
        input_shape=(feature_matrix.shape[1],),
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
    embeddings = intermediate_model.predict(feature_matrix, verbose=0)
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

    model_path = os.path.join(models_dir, "model-deep-lsh-cicids.model")
    hash_tables_path = os.path.join(hash_tables_dir, "hash_tables_deeplsh_cicids.pkl")
    preprocessor_path = os.path.join(models_dir, "cicids_preprocessor.pkl")
    embeddings_path = os.path.join(models_dir, "cicids_embeddings.npy")
    embeddings_hamming_path = os.path.join(models_dir, "cicids_embeddings_hamming.npy")
    corpus_path = os.path.join(models_dir, "cicids_flows.csv")
    train_meta_path = os.path.join(models_dir, "cicids_train_metadata.json")

    intermediate_model.save(model_path)
    with open(hash_tables_path, "wb") as f:
        pickle.dump(hash_tables, f)
    np.save(embeddings_path, embeddings)
    np.save(embeddings_hamming_path, embeddings_hamming)
    flows_df.to_csv(corpus_path, index=False)

    prepared = prepared_paths(args.output_dir)
    shutil.copyfile(prepared["preprocessor"], preprocessor_path)

    train_meta = {
        "model_type": "mlp",
        "data_repo": args.data_repo,
        "prepared_output_dir": args.output_dir,
        "feature_columns": feature_columns,
        "n_flows": int(flows_df.shape[0]),
        "n_pairs": int(pairs_df.shape[0]),
        "lsh": {"L": int(L), "K": int(K), "b": int(args.b), "m": int(args.m)},
        "hidden_dims": [int(x) for x in args.hidden_dims],
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
    }
    with open(train_meta_path, "w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=2)

    print(
        " ".join(
            [
                "done",
                "dataset=CIC-IDS-2017",
                f"n_flows={flows_df.shape[0]}",
                f"n_pairs={pairs_df.shape[0]}",
                f"n_features={len(feature_columns)}",
                f"size_hash_vector={size_hash_vector}",
                f"lsh=(L={L},K={K})",
                f"model_path={model_path}",
                f"hash_tables_path={hash_tables_path}",
            ]
        )
    )


if __name__ == "__main__":
    main()
