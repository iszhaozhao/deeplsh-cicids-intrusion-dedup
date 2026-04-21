import argparse
import sys

from deeplsh._paths import cicids_processed_dir, cicids_raw_dir


def _normalize_max_samples(max_samples):
    if max_samples is None:
        return None
    if max_samples <= 0:
        return None
    return max_samples


def cmd_cicids_prepare(args) -> int:
    from deeplsh.cicids.pipeline import prepare_cicids_dataset

    data_repo = args.data_repo or str(cicids_raw_dir())
    output_dir = args.output_dir or str(cicids_processed_dir("full"))
    max_samples = _normalize_max_samples(args.max_samples)
    paths = prepare_cicids_dataset(
        data_dir=data_repo,
        output_dir=output_dir,
        max_samples=max_samples,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    print(
        " ".join(
            [
                "mode=cicids-prepare",
                f"data_repo={data_repo}",
                f"output_dir={output_dir}",
                f"max_samples={'all' if max_samples is None else max_samples}",
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
        "--loss-type",
        args.loss_type,
        "--margin",
        str(args.margin),
        "--alpha",
        str(args.alpha),
        "--quantization-warmup-epochs",
        str(args.quantization_warmup_epochs),
        "--negative-strategy",
        args.negative_strategy,
        "--hard-negative-min-jaccard",
        str(args.hard_negative_min_jaccard),
        "--hard-negative-max-jaccard",
        str(args.hard_negative_max_jaccard),
    ]
    if args.beta is not None:
        argv.extend(["--beta", str(args.beta)])
    if args.gamma is not None:
        argv.extend(["--gamma", str(args.gamma)])
    if args.quantization_weight is not None:
        argv.extend(["--quantization-weight", str(args.quantization_weight)])
    if args.balance_weight is not None:
        argv.extend(["--balance-weight", str(args.balance_weight)])
    if not args.attention_pooling:
        argv.append("--no-attention-pooling")
    if not args.layer_norm:
        argv.append("--no-layer-norm")
    if args.batch_norm:
        argv.append("--batch-norm")
    if args.force_prepare:
        argv.append("--force-prepare")

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        train_main()
    finally:
        sys.argv = old_argv
    return 0


def cmd_cicids_train_paper_lsh(args) -> int:
    from deeplsh.cicids.train_paper_lsh import main as train_main

    data_repo = args.data_repo or str(cicids_raw_dir())
    output_dir = args.output_dir or str(cicids_processed_dir("full"))

    argv = [
        "--data-repo",
        data_repo,
        "--output-dir",
        output_dir,
        "--similarity",
        args.similarity,
        "--max-samples",
        str(args.max_samples),
        "--max-pairs",
        str(args.max_pairs),
        "--min-nonempty-bins",
        str(args.min_nonempty_bins),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
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
        "--validation-fraction",
        str(args.validation_fraction),
        "--plot-fraction",
        str(args.plot_fraction),
        "--target-correlation",
        str(args.target_correlation),
        "--hash-configs",
        *args.hash_configs,
    ]
    if not args.attention_pooling:
        argv.append("--no-attention-pooling")
    if not args.layer_norm:
        argv.append("--no-layer-norm")
    if args.force_prepare:
        argv.append("--force-prepare")
    if args.force_pairs:
        argv.append("--force-pairs")

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
        df = load_cicids_raw_flows(data_dir=data_repo, max_samples=_normalize_max_samples(args.max_samples), seed=args.seed)
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


def cmd_cicids_plot_correlation(args) -> int:
    from deeplsh.cicids.plot_correlation import main as plot_main

    output_dir = args.output_dir or str(cicids_processed_dir("full"))
    argv = [
        "--output-dir",
        output_dir,
        "--model-type",
        args.model_type,
        "--similarity",
        args.similarity,
    ]
    if args.results_dir:
        argv.extend(["--results-dir", args.results_dir])

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        plot_main()
    finally:
        sys.argv = old_argv
    return 0


def cmd_cicids_plot_paper_results(args) -> int:
    from deeplsh.cicids.plot_paper_results import main as plot_main

    output_dir = args.output_dir or str(cicids_processed_dir("full"))
    argv = [
        "--output-dir",
        output_dir,
    ]
    if args.results_dir:
        argv.extend(["--results-dir", args.results_dir])
    if args.figures_dir:
        argv.extend(["--figures-dir", args.figures_dir])

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        plot_main()
    finally:
        sys.argv = old_argv
    return 0


def cmd_cicids_plot_paper_lsh_sensitivity(args) -> int:
    from deeplsh.cicids.plot_paper_lsh_sensitivity import main as plot_main

    output_dir = args.output_dir or str(cicids_processed_dir("full"))
    argv = [
        "--output-dir",
        output_dir,
    ]
    if args.results_dir:
        argv.extend(["--results-dir", args.results_dir])

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        plot_main()
    finally:
        sys.argv = old_argv
    return 0


def cmd_cicids_export_matlab_plot_data(args) -> int:
    from deeplsh.cicids.export_matlab_plot_data import main as export_main

    output_dir = args.output_dir or str(cicids_processed_dir("full"))
    argv = [
        "--output-dir",
        output_dir,
        "--query-limit",
        str(args.query_limit),
        "--folds",
        str(args.folds),
    ]
    if args.export_dir:
        argv.extend(["--export-dir", args.export_dir])

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        export_main()
    finally:
        sys.argv = old_argv
    return 0


def cmd_cicids_export_paper_matlab_data(args) -> int:
    from deeplsh.cicids.export_paper_matlab_data import main as export_main

    output_dir = args.output_dir or str(cicids_processed_dir("full"))
    argv = [
        "--output-dir",
        output_dir,
        "--query-limit",
        str(args.query_limit),
        "--folds",
        str(args.folds),
    ]
    if args.export_dir:
        argv.extend(["--export-dir", args.export_dir])

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        export_main()
    finally:
        sys.argv = old_argv
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Network intrusion log deduplication runner (CLI)")
    parser.add_argument("--data-repo", default=None, help="CIC-IDS raw data directory override")
    parser.add_argument("--output-dir", default=None, help="Processed CIC-IDS data directory override")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("cicids-prepare", help="Prepare CIC-IDS-2017 flow and sequence data")
    p_prepare.add_argument("--data-repo", default=None)
    p_prepare.add_argument("--output-dir", default=None)
    p_prepare.add_argument("--max-samples", type=int, default=12000, help="Maximum raw flows to read; use 0 for all rows")
    p_prepare.add_argument("--max-pairs", type=int, default=20000)
    p_prepare.add_argument("--seed", type=int, default=42)
    p_prepare.set_defaults(func=cmd_cicids_prepare)

    p_prepare_flow = sub.add_parser("cicids-prepare-flow", help="Alias of cicids-prepare for the flow baseline")
    p_prepare_flow.add_argument("--data-repo", default=None)
    p_prepare_flow.add_argument("--output-dir", default=None)
    p_prepare_flow.add_argument("--max-samples", type=int, default=12000, help="Maximum raw flows to read; use 0 for all rows")
    p_prepare_flow.add_argument("--max-pairs", type=int, default=20000)
    p_prepare_flow.add_argument("--seed", type=int, default=42)
    p_prepare_flow.set_defaults(func=cmd_cicids_prepare)

    p_prepare_seq = sub.add_parser("cicids-prepare-seq", help="Alias of cicids-prepare for sequence training data")
    p_prepare_seq.add_argument("--data-repo", default=None)
    p_prepare_seq.add_argument("--output-dir", default=None)
    p_prepare_seq.add_argument("--max-samples", type=int, default=12000, help="Maximum raw flows to read; use 0 for all rows")
    p_prepare_seq.add_argument("--max-pairs", type=int, default=20000)
    p_prepare_seq.add_argument("--seed", type=int, default=42)
    p_prepare_seq.set_defaults(func=cmd_cicids_prepare)

    p_train = sub.add_parser("cicids-train", help="Train the MLP DeepLSH baseline on CIC-IDS-2017 flows")
    p_train.add_argument("--data-repo", default=None)
    p_train.add_argument("--output-dir", default=None)
    p_train.add_argument("--max-samples", type=int, default=12000, help="Maximum raw flows to read; use 0 for all rows")
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
    p_train_mlp.add_argument("--max-samples", type=int, default=12000, help="Maximum raw flows to read; use 0 for all rows")
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

    p_train_bigru = sub.add_parser("cicids-train-bigru", help="Train the Bi-GRU + DeepLSH main model")
    p_train_bigru.add_argument("--data-repo", default=None)
    p_train_bigru.add_argument("--output-dir", default=None)
    p_train_bigru.add_argument("--max-samples", type=int, default=12000, help="Maximum raw flows to read; use 0 for all rows")
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
    p_train_bigru.add_argument("--loss-type", choices=["mse", "contrastive"], default="contrastive")
    p_train_bigru.add_argument("--margin", type=float, default=0.7)
    p_train_bigru.add_argument("--alpha", type=float, default=1.0)
    p_train_bigru.add_argument("--beta", type=float, default=0.1)
    p_train_bigru.add_argument("--gamma", type=float, default=0.01)
    p_train_bigru.add_argument("--quantization-warmup-epochs", type=int, default=3)
    p_train_bigru.add_argument("--quantization-weight", type=float, default=None, help="Deprecated alias for --gamma")
    p_train_bigru.add_argument("--balance-weight", type=float, default=None, help="Deprecated alias for --beta")
    p_train_bigru.add_argument("--negative-strategy", choices=["random", "hard"], default="hard")
    p_train_bigru.add_argument("--hard-negative-min-jaccard", type=float, default=0.3)
    p_train_bigru.add_argument("--hard-negative-max-jaccard", type=float, default=0.5)
    p_train_bigru.add_argument("--attention-pooling", dest="attention_pooling", action="store_true", default=True)
    p_train_bigru.add_argument("--no-attention-pooling", dest="attention_pooling", action="store_false")
    p_train_bigru.add_argument("--layer-norm", dest="layer_norm", action="store_true", default=True)
    p_train_bigru.add_argument("--no-layer-norm", dest="layer_norm", action="store_false")
    p_train_bigru.add_argument("--batch-norm", dest="batch_norm", action="store_true", default=False)
    p_train_bigru.add_argument("--no-batch-norm", dest="batch_norm", action="store_false")
    p_train_bigru.add_argument("--force-prepare", action="store_true")
    p_train_bigru.set_defaults(func=cmd_cicids_train_bigru)

    p_train_paper = sub.add_parser("cicids-train-paper-lsh", help="Train paper-faithful BiGRU-DeepLSH Jaccard correlation model")
    p_train_paper.add_argument("--data-repo", default=None)
    p_train_paper.add_argument("--output-dir", default=None)
    p_train_paper.add_argument("--similarity", choices=["jaccard"], default="jaccard")
    p_train_paper.add_argument("--max-samples", type=int, default=12000, help="Maximum raw flows to read; use 0 for all rows")
    p_train_paper.add_argument("--max-pairs", type=int, default=120000)
    p_train_paper.add_argument("--min-nonempty-bins", type=int, default=8)
    p_train_paper.add_argument("--epochs", type=int, default=60)
    p_train_paper.add_argument("--batch-size", type=int, default=128)
    p_train_paper.add_argument("--seed", type=int, default=42)
    p_train_paper.add_argument("--lsh-param-index", type=int, default=4)
    p_train_paper.add_argument("--embed-dim", type=int, default=64)
    p_train_paper.add_argument("--gru-units", type=int, default=64)
    p_train_paper.add_argument("--dense-dim", type=int, default=128)
    p_train_paper.add_argument("--hash-configs", nargs="+", default=["128:8", "256:4", "512:2"])
    p_train_paper.add_argument("--validation-fraction", type=float, default=0.2)
    p_train_paper.add_argument("--plot-fraction", type=float, default=0.2)
    p_train_paper.add_argument("--target-correlation", type=float, default=0.75)
    p_train_paper.add_argument("--attention-pooling", dest="attention_pooling", action="store_true", default=True)
    p_train_paper.add_argument("--no-attention-pooling", dest="attention_pooling", action="store_false")
    p_train_paper.add_argument("--layer-norm", dest="layer_norm", action="store_true", default=True)
    p_train_paper.add_argument("--no-layer-norm", dest="layer_norm", action="store_false")
    p_train_paper.add_argument("--force-prepare", action="store_true")
    p_train_paper.add_argument("--force-pairs", action="store_true")
    p_train_paper.set_defaults(func=cmd_cicids_train_paper_lsh)

    p_eval = sub.add_parser("cicids-eval", help="Evaluate md5/simhash/mlp/bigru experiment metrics")
    p_eval.add_argument("--output-dir", default=None)
    p_eval.add_argument("--results-dir", default=None)
    p_eval.add_argument("--top-k", type=int, default=10)
    p_eval.add_argument("--sample-limit", type=int, default=50)
    p_eval.set_defaults(func=cmd_cicids_eval)

    p_plot_correlation = sub.add_parser("cicids-plot-correlation", help="Plot true similarity versus DeepLSH Hamming similarity")
    p_plot_correlation.add_argument("--output-dir", default=None)
    p_plot_correlation.add_argument("--results-dir", default=None)
    p_plot_correlation.add_argument("--model-type", choices=["mlp", "bigru"], default="bigru")
    p_plot_correlation.add_argument("--similarity", choices=["jaccard"], default="jaccard")
    p_plot_correlation.set_defaults(func=cmd_cicids_plot_correlation)

    p_plot_paper = sub.add_parser("cicids-plot-paper-results", help="Generate paper-ready CIC-IDS experiment figures")
    p_plot_paper.add_argument("--output-dir", default=None)
    p_plot_paper.add_argument("--results-dir", default=None)
    p_plot_paper.add_argument("--figures-dir", default=None)
    p_plot_paper.set_defaults(func=cmd_cicids_plot_paper_results)

    p_plot_paper_lsh = sub.add_parser(
        "cicids-plot-paper-lsh-sensitivity",
        help="Plot paper-faithful BiGRU DeepLSH LSH parameter sensitivity",
    )
    p_plot_paper_lsh.add_argument("--output-dir", default=None)
    p_plot_paper_lsh.add_argument("--results-dir", default=None)
    p_plot_paper_lsh.set_defaults(func=cmd_cicids_plot_paper_lsh_sensitivity)

    p_matlab = sub.add_parser("cicids-export-matlab-plot-data", help="Export MATLAB-ready CSV files for paper figures")
    p_matlab.add_argument("--output-dir", default=None)
    p_matlab.add_argument("--export-dir", default=None)
    p_matlab.add_argument("--query-limit", type=int, default=120)
    p_matlab.add_argument("--folds", type=int, default=10)
    p_matlab.set_defaults(func=cmd_cicids_export_matlab_plot_data)

    p_paper_matlab = sub.add_parser(
        "cicids-export-paper-matlab-data",
        help="Export paper-faithful MATLAB CSV files and scripts for CIC-IDS figures",
    )
    p_paper_matlab.add_argument("--output-dir", default=None)
    p_paper_matlab.add_argument("--export-dir", default=None)
    p_paper_matlab.add_argument("--query-limit", type=int, default=120)
    p_paper_matlab.add_argument("--folds", type=int, default=10)
    p_paper_matlab.set_defaults(func=cmd_cicids_export_paper_matlab_data)

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
