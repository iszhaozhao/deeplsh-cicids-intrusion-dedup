import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_python_packages_on_path() -> None:
    root = _project_root()
    code_dir = os.path.join(root, "code")
    python_packages = os.path.join(root, "code", "python-packages")
    for path in (code_dir, python_packages):
        if path not in sys.path:
            sys.path.insert(0, path)


def cmd_list(args) -> int:
    root = _project_root()
    data_repo = args.data_repo or os.path.join(root, "data")
    df_measures = pd.read_csv(os.path.join(data_repo, "similarity-measures-pairs.csv"), index_col=0)

    print("available_measures:")
    for col in df_measures.columns:
        print(col)

    print("\navailable_commands:")
    print("lite   - query a single pair similarity value from similarity-measures-pairs.csv")
    print("deeplsh - train DeepLSH for a selected measure and build LSH hash tables")
    print("cicids-prepare - prepare CIC-IDS-2017 CSV flow data and training pairs")
    print("cicids-train - train DeepLSH on CIC-IDS-2017 numeric flow features")
    print("cicids-query - query near-duplicate CIC-IDS flows from trained artifacts")
    print("cicids-list-labels - inspect CIC-IDS label distribution")
    return 0


def cmd_lite(args) -> int:
    _ensure_python_packages_on_path()
    from similarities import get_index_sim

    root = _project_root()
    data_repo = args.data_repo or os.path.join(root, "data")

    df_stacks = pd.read_csv(os.path.join(data_repo, "frequent_stack_traces.csv"), index_col=0)
    n_stacks = int(args.n_stacks or df_stacks.shape[0])

    if n_stacks != 1000:
        raise ValueError(
            "lite mode uses similarity-measures-pairs.csv which corresponds to 1000 stacks in this repo. "
            "Please use --n-stacks 1000."
        )

    df_measures = pd.read_csv(os.path.join(data_repo, "similarity-measures-pairs.csv"), index_col=0)
    if args.measure not in df_measures.columns:
        raise ValueError(
            f"Unknown --measure '{args.measure}'. Available: {', '.join(df_measures.columns.tolist())}"
        )

    a = int(args.index_a)
    b = int(args.index_b)
    if a == b:
        raise ValueError("index_a and index_b must be different")
    if a > b:
        a, b = b, a

    row_idx = get_index_sim(n_stacks, a, b)
    score = float(df_measures[args.measure].loc[row_idx])
    print(f"mode=lite measure={args.measure} n_stacks={n_stacks} a={a} b={b} row={row_idx} score={score:.6f}")
    return 0


def cmd_deeplsh(args) -> int:
    from train_deeplsh import main as train_main

    argv = [
        "--measure",
        args.measure,
        "--n",
        str(args.n),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--m",
        str(args.m),
        "--b",
        str(args.b),
        "--seed",
        str(args.seed),
        "--lsh-param-index",
        str(args.lsh_param_index),
    ]
    if args.kw:
        argv.extend(["--kw", *[str(x) for x in args.kw]])
    if args.data_repo:
        argv.extend(["--data-repo", args.data_repo])

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        train_main()
    finally:
        sys.argv = old_argv
    return 0


def cmd_cicids_prepare(args) -> int:
    _ensure_python_packages_on_path()
    from cicids_pipeline import default_processed_data_dir, default_raw_data_dir, prepare_cicids_dataset

    data_repo = args.data_repo or default_raw_data_dir()
    output_dir = args.output_dir or default_processed_data_dir()
    paths = prepare_cicids_dataset(
        data_dir=data_repo,
        output_dir=output_dir,
        max_samples=args.max_samples,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    print(
        " ".join(
            [
                "mode=cicids-prepare",
                f"data_repo={data_repo}",
                f"output_dir={output_dir}",
                f"flows={paths['flows']}",
                f"pairs={paths['pairs']}",
                f"metadata={paths['metadata']}",
            ]
        )
    )
    return 0


def cmd_cicids_train(args) -> int:
    _ensure_python_packages_on_path()
    from cicids_pipeline import default_processed_data_dir, default_raw_data_dir
    from train_cicids_deeplsh import main as train_main

    data_repo = args.data_repo or default_raw_data_dir()
    output_dir = args.output_dir or default_processed_data_dir()

    argv = [
        "--data-repo",
        data_repo,
        "--output-dir",
        output_dir,
        "--max-samples",
        str(args.max_samples),
        "--max-pairs",
        str(args.max_pairs),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--m",
        str(args.m),
        "--b",
        str(args.b),
        "--seed",
        str(args.seed),
        "--lsh-param-index",
        str(args.lsh_param_index),
    ]
    if args.hidden_dims:
        argv.extend(["--hidden-dims", *[str(x) for x in args.hidden_dims]])
    if args.force_prepare:
        argv.append("--force-prepare")

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        train_main()
    finally:
        sys.argv = old_argv
    return 0


def cmd_cicids_list_labels(args) -> int:
    _ensure_python_packages_on_path()
    from cicids_pipeline import default_processed_data_dir, default_raw_data_dir, load_prepared_flows, load_cicids_raw_flows

    try:
        if args.from_raw:
            raise FileNotFoundError("forced raw read")
        output_dir = args.output_dir or default_processed_data_dir()
        df = load_prepared_flows(output_dir)
        source = output_dir
    except FileNotFoundError:
        data_repo = args.data_repo or default_raw_data_dir()
        df = load_cicids_raw_flows(data_dir=data_repo, max_samples=args.max_samples, seed=args.seed)
        source = data_repo

    counts = df["Label"].value_counts().sort_values(ascending=False)
    print(f"mode=cicids-list-labels source={source}")
    for label, count in counts.items():
        print(f"{label}\t{count}")
    return 0


def _cosine_01(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denom == 0.0:
        return 0.0
    cosine = float(np.dot(vector_a, vector_b) / denom)
    cosine = max(-1.0, min(1.0, cosine))
    return (cosine + 1.0) / 2.0


def _candidate_hit_counts(query_hamming, hash_tables, L, K, b):
    n_bits = K * b
    hit_counts = {}
    for bucket_index in range(L):
        key = query_hamming[bucket_index * n_bits : (bucket_index + 1) * n_bits].tobytes()
        entry = hash_tables.get(f"entry_{bucket_index}", {})
        if key not in entry:
            continue
        for candidate_index in entry[key]:
            candidate_index = int(candidate_index)
            hit_counts[candidate_index] = hit_counts.get(candidate_index, 0) + 1
    return hit_counts


def cmd_cicids_query(args) -> int:
    _ensure_python_packages_on_path()
    from tensorflow.keras.models import load_model

    root = _project_root()
    models_dir = os.path.join(root, "code", "Models")
    hash_tables_dir = os.path.join(root, "code", "Hash-Tables")

    model_path = os.path.join(models_dir, "model-deep-lsh-cicids.model")
    corpus_path = os.path.join(models_dir, "cicids_flows.csv")
    embeddings_path = os.path.join(models_dir, "cicids_embeddings.npy")
    preprocessor_path = os.path.join(models_dir, "cicids_preprocessor.pkl")
    train_meta_path = os.path.join(models_dir, "cicids_train_metadata.json")
    hash_tables_path = os.path.join(hash_tables_dir, "hash_tables_deeplsh_cicids.pkl")

    for required in [model_path, corpus_path, embeddings_path, preprocessor_path, train_meta_path, hash_tables_path]:
        if not os.path.exists(required):
            raise FileNotFoundError(f"Required CIC-IDS artifact not found: {required}")

    flows_df = pd.read_csv(corpus_path)
    embeddings = np.load(embeddings_path)
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)
    with open(train_meta_path, "r", encoding="utf-8") as f:
        train_meta = json.load(f)
    with open(hash_tables_path, "rb") as f:
        hash_tables = pickle.load(f)

    feature_columns = preprocessor["feature_columns"]
    model = load_model(model_path)
    lsh = train_meta["lsh"]
    L = int(lsh["L"])
    K = int(lsh["K"])
    b = int(lsh["b"])

    if args.sample_id is not None:
        matches = flows_df.index[flows_df["sample_id"] == args.sample_id].tolist()
        if not matches:
            raise ValueError(f"Unknown sample_id: {args.sample_id}")
        query_index = int(matches[0])
    elif args.row_index is not None:
        query_index = int(args.row_index)
        if query_index < 0 or query_index >= flows_df.shape[0]:
            raise ValueError(f"--row-index out of range: {query_index}")
    else:
        raise ValueError("One of --sample-id or --row-index is required.")

    query_vector = flows_df[feature_columns].iloc[[query_index]].to_numpy(dtype=np.float32)
    query_embedding = model.predict(query_vector, verbose=0)[0]
    query_hamming = np.where(query_embedding > 0, 1, -1).astype(np.int8)
    hit_counts = _candidate_hit_counts(query_hamming, hash_tables, L, K, b)

    label_scope = args.label_scope
    query_label = str(flows_df.iloc[query_index]["Label"])
    candidate_indices = [idx for idx in hit_counts.keys() if idx != query_index]
    if label_scope == "same":
        candidate_indices = [idx for idx in candidate_indices if str(flows_df.iloc[idx]["Label"]) == query_label]

    if not candidate_indices:
        candidate_indices = [idx for idx in range(flows_df.shape[0]) if idx != query_index]
        if label_scope == "same":
            candidate_indices = [idx for idx in candidate_indices if str(flows_df.iloc[idx]["Label"]) == query_label]

    if not candidate_indices:
        raise ValueError("No candidate flows available for the requested query scope.")

    records = []
    for candidate_index in candidate_indices:
        candidate_embedding = embeddings[candidate_index]
        records.append(
            {
                "query_sample_id": flows_df.iloc[query_index]["sample_id"],
                "candidate_sample_id": flows_df.iloc[candidate_index]["sample_id"],
                "query_label": query_label,
                "candidate_label": str(flows_df.iloc[candidate_index]["Label"]),
                "hash_bucket_hits": int(hit_counts.get(candidate_index, 0)),
                "embedding_similarity": _cosine_01(query_embedding, candidate_embedding),
                "is_same_label": int(str(flows_df.iloc[candidate_index]["Label"]) == query_label),
                "source_file": str(flows_df.iloc[candidate_index]["source_file"]),
            }
        )

    results = pd.DataFrame.from_records(records)
    results = results.sort_values(
        by=["hash_bucket_hits", "embedding_similarity", "candidate_sample_id"],
        ascending=[False, False, True],
    ).head(args.top_k)

    if args.output_csv:
        results.to_csv(args.output_csv, index=False)
        print(f"mode=cicids-query output_csv={args.output_csv} rows={results.shape[0]}")
    else:
        print(results.to_string(index=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepLSH local runner (non-notebook)")
    parser.add_argument("--data-repo", default=None, help="Path to data directory (default: <repo>/data)")
    parser.add_argument("--output-dir", default=None, help="Path to processed CIC-IDS data directory")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List available measures and commands")
    p_list.set_defaults(func=cmd_list)

    p_lite = sub.add_parser("lite", help="Query one similarity value from the precomputed pairs file")
    p_lite.add_argument("--measure", required=True)
    p_lite.add_argument("--index-a", type=int, required=True)
    p_lite.add_argument("--index-b", type=int, required=True)
    p_lite.add_argument("--n-stacks", type=int, default=1000)
    p_lite.set_defaults(func=cmd_lite)

    p_deep = sub.add_parser("deeplsh", help="Train DeepLSH for a selected measure and build LSH hash tables")
    p_deep.add_argument("--measure", required=True)
    p_deep.add_argument("--n", type=int, default=1000)
    p_deep.add_argument("--epochs", type=int, default=20)
    p_deep.add_argument("--batch-size", type=int, default=512)
    p_deep.add_argument("--m", type=int, default=64)
    p_deep.add_argument("--b", type=int, default=16)
    p_deep.add_argument("--kw", type=int, nargs="+", default=[3, 4])
    p_deep.add_argument("--seed", type=int, default=42)
    p_deep.add_argument("--lsh-param-index", type=int, default=2)
    p_deep.set_defaults(func=cmd_deeplsh)

    p_prepare = sub.add_parser("cicids-prepare", help="Prepare CIC-IDS-2017 CSV data and sampled pairs")
    p_prepare.add_argument("--data-repo", default=None)
    p_prepare.add_argument("--output-dir", default=None)
    p_prepare.add_argument("--max-samples", type=int, default=12000)
    p_prepare.add_argument("--max-pairs", type=int, default=20000)
    p_prepare.add_argument("--seed", type=int, default=42)
    p_prepare.set_defaults(func=cmd_cicids_prepare)

    p_train = sub.add_parser("cicids-train", help="Train DeepLSH on CIC-IDS-2017 flows")
    p_train.add_argument("--data-repo", default=None)
    p_train.add_argument("--output-dir", default=None)
    p_train.add_argument("--max-samples", type=int, default=12000)
    p_train.add_argument("--max-pairs", type=int, default=20000)
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument("--m", type=int, default=64)
    p_train.add_argument("--b", type=int, default=16)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--lsh-param-index", type=int, default=2)
    p_train.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    p_train.add_argument("--force-prepare", action="store_true")
    p_train.set_defaults(func=cmd_cicids_train)

    p_query = sub.add_parser("cicids-query", help="Query near-duplicate CIC-IDS flows")
    p_query.add_argument("--sample-id", default=None)
    p_query.add_argument("--row-index", type=int, default=None)
    p_query.add_argument("--label-scope", choices=["same", "all"], default="same")
    p_query.add_argument("--top-k", type=int, default=10)
    p_query.add_argument("--output-csv", default=None)
    p_query.set_defaults(func=cmd_cicids_query)

    p_labels = sub.add_parser("cicids-list-labels", help="List available CIC-IDS labels")
    p_labels.add_argument("--data-repo", default=None)
    p_labels.add_argument("--output-dir", default=None)
    p_labels.add_argument("--from-raw", action="store_true")
    p_labels.add_argument("--max-samples", type=int, default=None)
    p_labels.add_argument("--seed", type=int, default=42)
    p_labels.set_defaults(func=cmd_cicids_list_labels)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
