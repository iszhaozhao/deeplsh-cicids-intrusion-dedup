import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deeplsh._paths import cicids_artifacts_dir
from deeplsh.cicids.pipeline import default_processed_data_dir


DEFAULT_LSH_GRID: List[Tuple[int, int]] = [(1, 1), (1, 5), (1, 10), (1, 20), (2, 20), (4, 20)]
ORDER_PRESERVING_CONFIG = {"K": 1, "L": 1, "column": "p_lsh_K1_L1"}
FILTERING_CONFIG = {"K": 4, "L": 20, "column": "p_lsh_K4_L20"}
CORRELATION_FILENAME = "cicids_lsh_correlation_jaccard_bigru_paper.csv"
SENSITIVITY_FILENAME = "cicids_lsh_sensitivity_jaccard_bigru_paper.csv"
SENSITIVITY_PNG_FILENAME = "cicids_lsh_sensitivity_jaccard_bigru_paper.png"
CALIBRATION_PNG_FILENAME = "cicids_lsh_calibration_curves_jaccard_bigru_paper.png"
ACADEMIC_PAIR_PNG_FILENAME = "cicids_lsh_academic_pair_jaccard_bigru_paper.png"
SUMMARY_FILENAME = "cicids_lsh_sensitivity_jaccard_bigru_paper_summary.json"


def default_paper_lsh_results_dir(output_dir: str) -> str:
    return str(cicids_artifacts_dir() / "results" / Path(output_dir).name / "paper_lsh")


def _prob_lsh(group_similarity: np.ndarray, k: int, l: int) -> np.ndarray:
    group_similarity = np.clip(group_similarity.astype(np.float32), 0.0, 1.0)
    probabilities = 1.0 - np.power(1.0 - np.power(group_similarity, int(k)), int(l))
    return np.clip(probabilities, 0.0, 1.0).astype(np.float32)


def _column_name(k: int, l: int) -> str:
    return f"p_lsh_K{k}_L{l}"


def _safe_corr(left: pd.Series, right: pd.Series, method: str) -> float:
    value = left.corr(right, method=method)
    return 0.0 if pd.isna(value) else float(value)


def _interval_means(true_values: np.ndarray, pred_values: np.ndarray, mask: np.ndarray) -> Tuple[int, float, float, float]:
    count = int(np.sum(mask))
    if count == 0:
        return 0, 0.0, 0.0, 1.0
    mean_true = float(np.mean(true_values[mask]))
    mean_pred = float(np.mean(pred_values[mask]))
    return count, mean_true, mean_pred, float(abs(mean_pred - mean_true))


def _diagnostics_for_column(df: pd.DataFrame, column: str, k: int, l: int) -> Dict[str, float]:
    true_values = df["true_sim"].to_numpy(dtype=np.float32)
    pred_values = df[column].to_numpy(dtype=np.float32)
    mid_count, mid_true, mid_pred, mid_gap = _interval_means(true_values, pred_values, (true_values >= 0.4) & (true_values < 0.7))
    high_count, high_true, high_pred, high_gap = _interval_means(true_values, pred_values, true_values >= 0.8)
    calibration_mae = float(np.mean(np.abs(pred_values - true_values))) if len(true_values) else 0.0
    recommendation_score = calibration_mae + mid_gap + high_gap
    return {
        "K": int(k),
        "L": int(l),
        "column": column,
        "pearson": _safe_corr(df["true_sim"], df[column], "pearson"),
        "spearman": _safe_corr(df["true_sim"], df[column], "spearman"),
        "calibration_mae": calibration_mae,
        "mid_sim_count": mid_count,
        "mid_sim_mean_true": mid_true,
        "mid_sim_mean_pred": mid_pred,
        "mid_sim_gap": mid_gap,
        "high_sim_count": high_count,
        "high_sim_mean_true": high_true,
        "high_sim_mean_pred": high_pred,
        "high_sim_gap": high_gap,
        "recommendation_score": recommendation_score,
    }


def build_sensitivity_dataframe(correlation_df: pd.DataFrame, grid: Iterable[Tuple[int, int]] = DEFAULT_LSH_GRID) -> pd.DataFrame:
    required = {"true_sim", "pred_sim", "similarity_bin"}
    missing = sorted(required - set(correlation_df.columns))
    if missing:
        raise ValueError(f"Missing required columns in paper LSH correlation CSV: {missing}")

    result = correlation_df.copy()
    result["group_collision_similarity"] = result["pred_sim"].astype(float)
    group_similarity = result["group_collision_similarity"].to_numpy(dtype=np.float32)
    for k, l in grid:
        result[_column_name(k, l)] = _prob_lsh(group_similarity, k=k, l=l)

    probability_columns = [_column_name(k, l) for k, l in grid]
    for column in ["group_collision_similarity", *probability_columns]:
        if not ((result[column] >= 0).all() and (result[column] <= 1).all()):
            raise ValueError(f"{column} values must be in [0, 1]")
    return result


def plot_sensitivity_scatter(sensitivity_df: pd.DataFrame, output_png: str, grid: Iterable[Tuple[int, int]] = DEFAULT_LSH_GRID) -> None:
    grid = list(grid)
    n_cols = 3
    n_rows = int(np.ceil(len(grid) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 7.5), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)
    for axis, (k, l) in zip(axes, grid):
        column = _column_name(k, l)
        axis.scatter(sensitivity_df["true_sim"], sensitivity_df[column], s=4, alpha=0.12, color="#1f77b4", marker="x")
        axis.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1)
        axis.set_title(f"K={k}, L={l}")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(True, linestyle=":", alpha=0.45)
    for axis in axes[len(grid) :]:
        axis.axis("off")
    fig.supxlabel("Jaccard similarity values")
    fig.supylabel("LSH collision probability")
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def _calibration_curve(sensitivity_df: pd.DataFrame, column: str, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    true_values = sensitivity_df["true_sim"].to_numpy(dtype=np.float32)
    pred_values = sensitivity_df[column].to_numpy(dtype=np.float32)
    for index in range(n_bins):
        left = bins[index]
        right = bins[index + 1]
        if index == n_bins - 1:
            mask = (true_values >= left) & (true_values <= right)
        else:
            mask = (true_values >= left) & (true_values < right)
        count = int(np.sum(mask))
        if count == 0:
            continue
        rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "bin_center": float((left + right) / 2.0),
                "count": count,
                "mean_true_sim": float(np.mean(true_values[mask])),
                "mean_pred_sim": float(np.mean(pred_values[mask])),
            }
        )
    return pd.DataFrame.from_records(rows)


def plot_calibration_curves(sensitivity_df: pd.DataFrame, output_png: str, grid: Iterable[Tuple[int, int]] = DEFAULT_LSH_GRID) -> pd.DataFrame:
    rows = []
    plt.figure(figsize=(7, 5.5))
    for k, l in grid:
        column = _column_name(k, l)
        curve = _calibration_curve(sensitivity_df, column)
        if curve.empty:
            continue
        curve.insert(0, "K", int(k))
        curve.insert(1, "L", int(l))
        curve.insert(2, "column", column)
        rows.append(curve)
        plt.plot(curve["mean_true_sim"], curve["mean_pred_sim"], marker="o", linewidth=1.2, markersize=3, label=f"K={k}, L={l}")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1, label="y=x")
    plt.xlabel("Mean Jaccard similarity")
    plt.ylabel("Mean LSH collision probability")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.45)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    if not rows:
        return pd.DataFrame(columns=["K", "L", "column", "bin_left", "bin_right", "bin_center", "count", "mean_true_sim", "mean_pred_sim"])
    return pd.concat(rows, ignore_index=True)


def plot_academic_pair(sensitivity_df: pd.DataFrame, output_png: str) -> None:
    panels = [
        (ORDER_PRESERVING_CONFIG, "Order-preserving view (K=1, L=1)"),
        (FILTERING_CONFIG, "Filtering view (K=4, L=20)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for axis, (config, title) in zip(axes, panels):
        column = config["column"]
        if column not in sensitivity_df.columns:
            raise ValueError(f"Missing required sensitivity column for academic pair plot: {column}")
        axis.scatter(sensitivity_df["true_sim"], sensitivity_df[column], s=5, alpha=0.12, color="#1f77b4", marker="x")
        axis.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(True, linestyle=":", alpha=0.45)
    axes[0].set_ylabel("LSH collision probability")
    for axis in axes:
        axis.set_xlabel("Jaccard similarity values")
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def run(output_dir: str, results_dir: str = None, grid: Iterable[Tuple[int, int]] = DEFAULT_LSH_GRID) -> Dict[str, object]:
    results_dir = results_dir or default_paper_lsh_results_dir(output_dir)
    os.makedirs(results_dir, exist_ok=True)
    correlation_path = os.path.join(results_dir, CORRELATION_FILENAME)
    if not os.path.exists(correlation_path):
        raise FileNotFoundError(f"Paper LSH correlation CSV not found: {correlation_path}")

    correlation_df = pd.read_csv(correlation_path)
    sensitivity_df = build_sensitivity_dataframe(correlation_df, grid=grid)
    sensitivity_path = os.path.join(results_dir, SENSITIVITY_FILENAME)
    sensitivity_png = os.path.join(results_dir, SENSITIVITY_PNG_FILENAME)
    calibration_png = os.path.join(results_dir, CALIBRATION_PNG_FILENAME)
    academic_pair_png = os.path.join(results_dir, ACADEMIC_PAIR_PNG_FILENAME)
    summary_path = os.path.join(results_dir, SUMMARY_FILENAME)

    sensitivity_df.to_csv(sensitivity_path, index=False)
    plot_sensitivity_scatter(sensitivity_df, sensitivity_png, grid=grid)
    calibration_curve_df = plot_calibration_curves(sensitivity_df, calibration_png, grid=grid)
    plot_academic_pair(sensitivity_df, academic_pair_png)

    diagnostics = [_diagnostics_for_column(sensitivity_df, _column_name(k, l), k=k, l=l) for k, l in grid]
    recommended = min(diagnostics, key=lambda item: (item["recommendation_score"], item["calibration_mae"]))
    summary = {
        "source_correlation_csv": correlation_path,
        "sensitivity_csv": sensitivity_path,
        "sensitivity_png": sensitivity_png,
        "calibration_curves_png": calibration_png,
        "academic_pair_png": academic_pair_png,
        "n_pairs": int(sensitivity_df.shape[0]),
        "grid": [{"K": int(k), "L": int(l), "column": _column_name(k, l)} for k, l in grid],
        "diagnostics": diagnostics,
        "recommended_config": recommended,
        "academic_recommended_figures": {
            "order_preserving_config": ORDER_PRESERVING_CONFIG,
            "filtering_config": FILTERING_CONFIG,
            "academic_pair_png": academic_pair_png,
        },
        "calibration_curve_bins": calibration_curve_df.to_dict(orient="records"),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot paper-faithful BiGRU DeepLSH LSH parameter sensitivity.")
    parser.add_argument("--output-dir", default=None, help="Prepared CIC-IDS data directory")
    parser.add_argument("--results-dir", default=None, help="Paper LSH result directory")
    args = parser.parse_args()

    output_dir = args.output_dir or default_processed_data_dir()
    summary = run(output_dir=output_dir, results_dir=args.results_dir)
    recommended = summary["recommended_config"]
    print(
        " ".join(
            [
                "done",
                "mode=cicids-plot-paper-lsh-sensitivity",
                f"pairs={summary['n_pairs']}",
                f"recommended_K={recommended['K']}",
                f"recommended_L={recommended['L']}",
                f"recommendation_score={recommended['recommendation_score']:.6f}",
                f"sensitivity_csv={summary['sensitivity_csv']}",
                f"sensitivity_png={summary['sensitivity_png']}",
                f"calibration_curves_png={summary['calibration_curves_png']}",
                f"academic_pair_png={summary['academic_pair_png']}",
            ]
        )
    )


if __name__ == "__main__":
    main()
