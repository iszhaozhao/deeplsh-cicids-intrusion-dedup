import argparse
import os
import sys

import pandas as pd

from deeplsh._paths import cicids_processed_dir, cicids_raw_dir, stacktraces_dataset_dir


def cmd_list(args) -> int:
    data_repo = args.data_repo or str(stacktraces_dataset_dir())
    df_measures = pd.read_csv(os.path.join(data_repo, "similarity-measures-pairs.csv"), index_col=0)

    print("available_measures:")
    for col in df_measures.columns:
        print(col)

    print("\navailable_commands:")
    print("lite   - query a single pair similarity value from similarity-measures-pairs.csv")
    print("deeplsh - train DeepLSH for a selected measure and build LSH hash tables (stack traces)")
    print("cicids-prepare - prepare CIC-IDS-2017 flow and sequence data")
    print("cicids-prepare-flow - alias of cicids-prepare for MLP baseline")
    print("cicids-prepare-seq - alias of cicids-prepare for Bi-GRU main model")
    print("cicids-train - train DeepLSH on CIC-IDS-2017 numeric flow features")
    print("cicids-train-mlp - alias of cicids-train for the baseline model")
    print("cicids-train-bigru - train Bi-GRU + DeepLSH on tokenized CIC-IDS logs")
    print("cicids-eval - evaluate md5/simhash/mlp/bigru metrics and write results")
    print("cicids-query - query near-duplicate CIC-IDS flows from trained artifacts")
    print("cicids-list-labels - inspect CIC-IDS label distribution")
    return 0


def cmd_lite(args) -> int:
    from deeplsh.core.similarities import get_index_sim

    data_repo = args.data_repo or str(stacktraces_dataset_dir())

    df_stacks = pd.read_csv(os.path.join(data_repo, "frequent_stack_traces.csv"), index_col=0)
    n_stacks = int(args.n_stacks or df_stacks.shape[0])

    if n_stacks != 1000:
        raise ValueError(
            "lite mode uses similarity-measures-pairs.csv which corresponds to 1000 stacks in this repo. "
            "Please use --n-stacks 1000."
        )

    df_measures = pd.read_csv(os.path.join(data_repo, "similarity-measures-pairs.csv"), index_col=0)
    if args.measure not in df_measures.columns:
        raise ValueError(f"Unknown --measure '{args.measure}'. Available: {', '.join(df_measures.columns.tolist())}")

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
    from deeplsh.stacktraces.train import main as train_main

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
    from deeplsh.cicids.pipeline import prepare_cicids_dataset

    data_repo = args.data_repo or str(cicids_raw_dir())
    output_dir = args.output_dir or str(cicids_processed_dir("full"))
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
                f"flows_tokens={paths['flows_tokens']}",
                f"pairs={paths['pairs']}",
                f"metadata={paths['metadata']}",
                f"vocab={paths['vocab']}",
            ]
        )
    )
    return 0


def cmd_cicids_train(args) -> int:
    from deeplsh.cicids.train_mlp import main as train_main

    data_repo = args.data_repo or str(cicids_raw_dir())
    output_dir = args.output_dir or str(cicids_processed_dir("full"))

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


def cmd_cicids_train_bigru(args) -> int:
    from deeplsh.cicids.train_bigru import main as train_main

    data_repo = args.data_repo or str(cicids_raw_dir())
    output_dir = args.output_dir or str(cicids_processed_dir("full"))

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
        "--embed-dim",
        str(args.embed_dim),
        "--gru-units",
        str(args.gru_units),
        "--dense-dim",
        str(args.dense_dim),
    ]
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
    from deeplsh.cicids.pipeline import load_cicids_raw_flows, load_prepared_flows

    try:
        if args.from_raw:
            raise FileNotFoundError("forced raw read")
        output_dir = args.output_dir or str(cicids_processed_dir("full"))
        df = load_prepared_flows(output_dir)
        source = output_dir
    except FileNotFoundError:
        data_repo = args.data_repo or str(cicids_raw_dir())
        df = load_cicids_raw_flows(data_dir=data_repo, max_samples=args.max_samples, seed=args.seed)
        source = data_repo

    counts = df["Label"].value_counts().sort_values(ascending=False)
    print(f"mode=cicids-list-labels source={source}")
    for label, count in counts.items():
        print(f"{label}\t{count}")
    return 0


def cmd_cicids_query(args) -> int:
    from deeplsh.cicids.runtime import load_runtime_bundle, query_top_k

    bundle = load_runtime_bundle(args.model_type)
    corpus_df = bundle["corpus_df"]

    if args.sample_id is not None:
        matches = corpus_df.index[corpus_df["sample_id"] == args.sample_id].tolist()
        if not matches:
            raise ValueError(f"Unknown sample_id: {args.sample_id}")
        query_index = int(matches[0])
    elif args.row_index is not None:
        query_index = int(args.row_index)
        if query_index < 0 or query_index >= corpus_df.shape[0]:
            raise ValueError(f"--row-index out of range: {query_index}")
    else:
        raise ValueError("One of --sample-id or --row-index is required.")

    results = query_top_k(bundle, query_index=query_index, top_k=args.top_k, label_scope=args.label_scope)

    if args.output_csv:
        results.to_csv(args.output_csv, index=False)
        print(f"mode=cicids-query model_type={args.model_type} output_csv={args.output_csv} rows={results.shape[0]}")
    else:
        print(results.to_string(index=False))
    return 0


def cmd_cicids_eval(args) -> int:
    from deeplsh.cicids.evaluate import main as eval_main

    output_dir = args.output_dir or str(cicids_processed_dir("full"))
    argv = [
        "--output-dir",
        output_dir,
        "--top-k",
        str(args.top_k),
        "--sample-limit",
        str(args.sample_limit),
    ]
    if args.results_dir:
        argv.extend(["--results-dir", args.results_dir])

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        eval_main()
    finally:
        sys.argv = old_argv
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepLSH local runner (CLI)")
    parser.add_argument("--data-repo", default=None, help="StackTraces/CIC-IDS data root override (per-command)")
    parser.add_argument("--output-dir", default=None, help="Processed CIC-IDS data directory override")

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

    p_prepare = sub.add_parser("cicids-prepare", help="Prepare CIC-IDS-2017 flow and sequence data")
    p_prepare.add_argument("--data-repo", default=None)
    p_prepare.add_argument("--output-dir", default=None)
    p_prepare.add_argument("--max-samples", type=int, default=12000)
    p_prepare.add_argument("--max-pairs", type=int, default=20000)
    p_prepare.add_argument("--seed", type=int, default=42)
    p_prepare.set_defaults(func=cmd_cicids_prepare)

    p_prepare_flow = sub.add_parser("cicids-prepare-flow", help="Alias of cicids-prepare for the flow baseline")
    p_prepare_flow.add_argument("--data-repo", default=None)
    p_prepare_flow.add_argument("--output-dir", default=None)
    p_prepare_flow.add_argument("--max-samples", type=int, default=12000)
    p_prepare_flow.add_argument("--max-pairs", type=int, default=20000)
    p_prepare_flow.add_argument("--seed", type=int, default=42)
    p_prepare_flow.set_defaults(func=cmd_cicids_prepare)

    p_prepare_seq = sub.add_parser("cicids-prepare-seq", help="Alias of cicids-prepare for Bi-GRU sequence data")
    p_prepare_seq.add_argument("--data-repo", default=None)
    p_prepare_seq.add_argument("--output-dir", default=None)
    p_prepare_seq.add_argument("--max-samples", type=int, default=12000)
    p_prepare_seq.add_argument("--max-pairs", type=int, default=20000)
    p_prepare_seq.add_argument("--seed", type=int, default=42)
    p_prepare_seq.set_defaults(func=cmd_cicids_prepare)

    p_train = sub.add_parser("cicids-train", help="Train the MLP DeepLSH baseline on CIC-IDS-2017 flows")
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

    p_train_mlp = sub.add_parser("cicids-train-mlp", help="Alias of cicids-train for the flow baseline")
    p_train_mlp.add_argument("--data-repo", default=None)
    p_train_mlp.add_argument("--output-dir", default=None)
    p_train_mlp.add_argument("--max-samples", type=int, default=12000)
    p_train_mlp.add_argument("--max-pairs", type=int, default=20000)
    p_train_mlp.add_argument("--epochs", type=int, default=10)
    p_train_mlp.add_argument("--batch-size", type=int, default=256)
    p_train_mlp.add_argument("--m", type=int, default=64)
    p_train_mlp.add_argument("--b", type=int, default=16)
    p_train_mlp.add_argument("--seed", type=int, default=42)
    p_train_mlp.add_argument("--lsh-param-index", type=int, default=2)
    p_train_mlp.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    p_train_mlp.add_argument("--force-prepare", action="store_true")
    p_train_mlp.set_defaults(func=cmd_cicids_train)

    p_train_bigru = sub.add_parser("cicids-train-bigru", help="Train the Bi-GRU + DeepLSH paper model")
    p_train_bigru.add_argument("--data-repo", default=None)
    p_train_bigru.add_argument("--output-dir", default=None)
    p_train_bigru.add_argument("--max-samples", type=int, default=12000)
    p_train_bigru.add_argument("--max-pairs", type=int, default=20000)
    p_train_bigru.add_argument("--epochs", type=int, default=10)
    p_train_bigru.add_argument("--batch-size", type=int, default=128)
    p_train_bigru.add_argument("--m", type=int, default=64)
    p_train_bigru.add_argument("--b", type=int, default=16)
    p_train_bigru.add_argument("--seed", type=int, default=42)
    p_train_bigru.add_argument("--lsh-param-index", type=int, default=2)
    p_train_bigru.add_argument("--embed-dim", type=int, default=64)
    p_train_bigru.add_argument("--gru-units", type=int, default=64)
    p_train_bigru.add_argument("--dense-dim", type=int, default=128)
    p_train_bigru.add_argument("--force-prepare", action="store_true")
    p_train_bigru.set_defaults(func=cmd_cicids_train_bigru)

    p_eval = sub.add_parser("cicids-eval", help="Evaluate md5/simhash/mlp/bigru experiment metrics")
    p_eval.add_argument("--output-dir", default=None)
    p_eval.add_argument("--results-dir", default=None)
    p_eval.add_argument("--top-k", type=int, default=10)
    p_eval.add_argument("--sample-limit", type=int, default=50)
    p_eval.set_defaults(func=cmd_cicids_eval)

    p_query = sub.add_parser("cicids-query", help="Query near-duplicate CIC-IDS flows")
    p_query.add_argument("--model-type", choices=["mlp", "bigru"], default="bigru")
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

