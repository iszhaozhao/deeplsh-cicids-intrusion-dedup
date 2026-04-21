import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deeplsh._paths import cicids_artifacts_dir
from deeplsh.cicids.pipeline import default_processed_data_dir, load_prepared_pairs, load_prepared_token_flows
from deeplsh.cicids.runtime import artifact_paths


def load_pairs_for_plot(output_dir: str, model_type: str) -> tuple[pd.DataFrame, str]:
    heldout_path = cicids_artifacts_dir() / "results" / Path(output_dir).name / f"pairs_validation_{model_type}.csv"
    if os.path.exists(heldout_path):
        return pd.read_csv(heldout_path), str(heldout_path)
    return load_prepared_pairs(output_dir), os.path.join(output_dir, "pairs_train.csv")


def jaccard_similarity(tokens_a: set, tokens_b: set) -> float:
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return float(len(tokens_a & tokens_b) / len(union))


def token_sets_from_sequences(token_sequences: pd.Series) -> list:
    return [set(str(sequence).split()) for sequence in token_sequences.fillna("")]


def hamming_pair_similarity(hash_codes: np.ndarray, index_a: int, index_b: int) -> float:
    return float(np.mean(hash_codes[index_a] == hash_codes[index_b]))


def build_correlation_dataframe(pairs_df: pd.DataFrame, token_flows_df: pd.DataFrame, hash_codes: np.ndarray) -> pd.DataFrame:
    max_pair_index = int(max(pairs_df["flow_index_1"].max(), pairs_df["flow_index_2"].max()))
    if max_pair_index >= token_flows_df.shape[0]:
        raise ValueError(f"Pair index {max_pair_index} exceeds token rows {token_flows_df.shape[0]}")
    if max_pair_index >= hash_codes.shape[0]:
        raise ValueError(f"Pair index {max_pair_index} exceeds hash code rows {hash_codes.shape[0]}")

    token_sets = token_sets_from_sequences(token_flows_df["token_sequence"])
    indices_1 = pairs_df["flow_index_1"].to_numpy(dtype=int)
    indices_2 = pairs_df["flow_index_2"].to_numpy(dtype=int)

    true_sims = np.asarray([jaccard_similarity(token_sets[a], token_sets[b]) for a, b in zip(indices_1, indices_2)], dtype=np.float32)
    pred_sims = np.asarray([hamming_pair_similarity(hash_codes, a, b) for a, b in zip(indices_1, indices_2)], dtype=np.float32)

    if not ((true_sims >= 0).all() and (true_sims <= 1).all()):
        raise ValueError("true_sim values must be in [0, 1]")
    if not ((pred_sims >= 0).all() and (pred_sims <= 1).all()):
        raise ValueError("pred_sim values must be in [0, 1]")

    return pd.DataFrame(
        {
            "flow_index_1": indices_1,
            "flow_index_2": indices_2,
            "label_1": pairs_df["label_1"].astype(str),
            "label_2": pairs_df["label_2"].astype(str),
            "true_sim": true_sims,
            "pred_sim": pred_sims,
            "is_duplicate": pairs_df["is_duplicate"].to_numpy(dtype=int),
        }
    )


def plot_lsh_correlation(correlation_df: pd.DataFrame, metric_name: str, model_name: str, output_png: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(correlation_df["true_sim"], correlation_df["pred_sim"], s=1, alpha=0.1, color="#1f77b4", marker="x")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1)
    plt.title(f"{metric_name} ({model_name})")
    plt.xlabel(f"{metric_name} similarity values")
    plt.ylabel("Hamming similarity")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


def output_paths(results_dir: str, similarity: str, model_type: str) -> tuple:
    stem = f"cicids_lsh_correlation_{similarity}_{model_type}"
    return os.path.join(results_dir, f"{stem}.png"), os.path.join(results_dir, f"{stem}.csv")


def run(output_dir: str, results_dir: str, model_type: str, similarity: str) -> dict:
    if similarity != "jaccard":
        raise ValueError(f"Unsupported similarity: {similarity}")

    os.makedirs(results_dir, exist_ok=True)
    pairs_df, pairs_source = load_pairs_for_plot(output_dir, model_type)
    token_flows_df = load_prepared_token_flows(output_dir)
    hash_path = artifact_paths(model_type)["embeddings_hamming"]
    if not os.path.exists(hash_path):
        raise FileNotFoundError(f"Hash code artifact not found: {hash_path}")
    hash_codes = np.load(hash_path)

    correlation_df = build_correlation_dataframe(pairs_df, token_flows_df, hash_codes)
    output_png, output_csv = output_paths(results_dir, similarity, model_type)
    correlation_df.to_csv(output_csv, index=False)
    model_name = "BiGRU-DeepLSH" if model_type == "bigru" else "MLP-DeepLSH"
    plot_lsh_correlation(correlation_df, "Jaccard", model_name, output_png)

    return {
        "output_png": output_png,
        "output_csv": output_csv,
        "pairs_rows": int(pairs_df.shape[0]),
        "pairs_source": pairs_source,
        "tokens_rows": int(token_flows_df.shape[0]),
        "hash_rows": int(hash_codes.shape[0]),
        "hash_bits": int(hash_codes.shape[1]),
        "max_pair_index": int(max(pairs_df["flow_index_1"].max(), pairs_df["flow_index_2"].max())),
        "csv_rows": int(correlation_df.shape[0]),
        "true_sim_min": float(correlation_df["true_sim"].min()),
        "true_sim_max": float(correlation_df["true_sim"].max()),
        "pred_sim_min": float(correlation_df["pred_sim"].min()),
        "pred_sim_max": float(correlation_df["pred_sim"].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CIC-IDS true similarity versus DeepLSH Hamming similarity.")
    parser.add_argument("--output-dir", default=None, help="Prepared CIC-IDS data directory")
    parser.add_argument("--results-dir", default=None, help="Directory to write correlation PNG/CSV")
    parser.add_argument("--model-type", choices=["mlp", "bigru"], default="bigru")
    parser.add_argument("--similarity", choices=["jaccard"], default="jaccard")
    args = parser.parse_args()

    output_dir = args.output_dir or default_processed_data_dir()
    results_dir = args.results_dir or str(cicids_artifacts_dir() / "results" / "full")
    summary = run(output_dir=output_dir, results_dir=results_dir, model_type=args.model_type, similarity=args.similarity)
    print(
        " ".join(
            [
                "done",
                f"mode=cicids-plot-correlation",
                f"model_type={args.model_type}",
                f"similarity={args.similarity}",
                f"pairs_rows={summary['pairs_rows']}",
                f"pairs_source={summary['pairs_source']}",
                f"tokens_rows={summary['tokens_rows']}",
                f"hash_rows={summary['hash_rows']}",
                f"hash_bits={summary['hash_bits']}",
                f"max_pair_index={summary['max_pair_index']}",
                f"csv_rows={summary['csv_rows']}",
                f"true_sim_range=[{summary['true_sim_min']:.6f},{summary['true_sim_max']:.6f}]",
                f"pred_sim_range=[{summary['pred_sim_min']:.6f},{summary['pred_sim_max']:.6f}]",
                f"output_png={summary['output_png']}",
                f"output_csv={summary['output_csv']}",
            ]
        )
    )


if __name__ == "__main__":
    main()
