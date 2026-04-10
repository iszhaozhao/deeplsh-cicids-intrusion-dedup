import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from deeplsh._paths import cicids_artifacts_dir
from deeplsh.cicids.pipeline import default_processed_data_dir, load_prepared_pairs, load_prepared_token_flows
from deeplsh.cicids.runtime import (
    average_query_latency_ms,
    best_threshold_metrics,
    binary_hash_collision_rate,
    classification_metrics,
    load_runtime_bundle,
    pair_scores_from_embeddings,
    simhash_pair_scores,
    simhash_query_latency_ms,
    simhash_signatures,
)


def _series_collision_rate(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return 1.0 - (len(set(values.tolist())) / float(len(values)))


def _to_python_scalars(obj):
    if isinstance(obj, dict):
        return {key: _to_python_scalars(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_python_scalars(value) for value in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main():
    parser = argparse.ArgumentParser(description="Evaluate CIC-IDS DeepLSH models and baselines.")
    parser.add_argument("--output-dir", default=None, help="Prepared data directory")
    parser.add_argument("--results-dir", default=None, help="Directory to write evaluation results")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--sample-limit", type=int, default=50)
    args = parser.parse_args()

    output_dir = args.output_dir or default_processed_data_dir()
    results_dir = args.results_dir or str((cicids_artifacts_dir() / "results" / "full"))
    os.makedirs(results_dir, exist_ok=True)

    pairs_df = load_prepared_pairs(output_dir)
    token_flows_df = load_prepared_token_flows(output_dir)
    y_true = pairs_df["is_duplicate"].to_numpy(dtype=int)

    baseline_rows = []
    summary = {"results_dir": results_dir}

    token_sequences = token_flows_df["token_sequence"].fillna("").astype(str).tolist()
    md5_hashes = np.asarray([hashlib.md5(sequence.encode("utf-8")).hexdigest() for sequence in token_sequences], dtype=object)
    md5_pair_scores = np.asarray(
        [1.0 if md5_hashes[a] == md5_hashes[b] else 0.0 for a, b in zip(pairs_df["flow_index_1"], pairs_df["flow_index_2"])],
        dtype=np.float32,
    )
    md5_metrics = classification_metrics(y_true, (md5_pair_scores >= 1.0).astype(int))
    md5_metrics.update(
        {
            "model": "exact-md5",
            "threshold": 1.0,
            "compression_rate": _series_collision_rate(md5_hashes),
            "avg_query_latency_ms": 0.0,
        }
    )
    baseline_rows.append(md5_metrics)

    simhash = simhash_signatures(token_sequences, n_bits=64)
    simhash_scores = simhash_pair_scores(simhash, pairs_df, n_bits=64)
    simhash_metrics = best_threshold_metrics(simhash_scores, y_true, thresholds=np.linspace(0.5, 0.98, 17))
    simhash_metrics.update(
        {
            "model": "simhash",
            "compression_rate": _series_collision_rate(simhash),
            "avg_query_latency_ms": simhash_query_latency_ms(simhash, top_k=args.top_k, limit=args.sample_limit),
        }
    )
    baseline_rows.append(simhash_metrics)

    mlp_bundle = load_runtime_bundle("mlp")
    mlp_scores = pair_scores_from_embeddings(mlp_bundle["embeddings"], pairs_df)
    mlp_metrics = best_threshold_metrics(mlp_scores, y_true)
    mlp_metrics.update(
        {
            "model": "baseline-mlp",
            "compression_rate": binary_hash_collision_rate(mlp_bundle["embeddings_hamming"]),
            "avg_query_latency_ms": average_query_latency_ms(mlp_bundle, top_k=args.top_k, limit=args.sample_limit),
        }
    )
    baseline_rows.append(mlp_metrics)

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_path = os.path.join(results_dir, "cicids_baseline_metrics.csv")
    baseline_df.to_csv(baseline_path, index=False)

    bigru_bundle = load_runtime_bundle("bigru")
    bigru_scores = pair_scores_from_embeddings(bigru_bundle["embeddings"], pairs_df)
    bigru_metrics = best_threshold_metrics(bigru_scores, y_true)
    bigru_metrics.update(
        {
            "model": "bigru-deeplsh",
            "compression_rate": binary_hash_collision_rate(bigru_bundle["embeddings_hamming"]),
            "avg_query_latency_ms": average_query_latency_ms(bigru_bundle, top_k=args.top_k, limit=args.sample_limit),
        }
    )
    bigru_df = pd.DataFrame([bigru_metrics])
    bigru_path = os.path.join(results_dir, "cicids_bigru_metrics.csv")
    bigru_df.to_csv(bigru_path, index=False)

    all_metrics = pd.concat([baseline_df, bigru_df], ignore_index=True)
    best_row = _to_python_scalars(
        all_metrics.sort_values(by=["f1", "recall", "precision"], ascending=[False, False, False]).iloc[0].to_dict()
    )
    summary.update(
        {
            "top_k": int(args.top_k),
            "sample_limit": int(args.sample_limit),
            "baseline_metrics_csv": baseline_path,
            "bigru_metrics_csv": bigru_path,
            "best_model": best_row,
        }
    )
    summary_path = os.path.join(results_dir, "cicids_comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_to_python_scalars(summary), f, indent=2)

    print(
        " ".join(
            [
                "done",
                f"baseline_metrics={baseline_path}",
                f"bigru_metrics={bigru_path}",
                f"summary={summary_path}",
                f"best_model={best_row['model']}",
                f"best_f1={float(best_row['f1']):.6f}",
            ]
        )
    )


if __name__ == "__main__":
    main()
