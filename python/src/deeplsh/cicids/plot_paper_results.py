import argparse
import os
from typing import Dict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from deeplsh._paths import cicids_artifacts_dir
from deeplsh.cicids.pipeline import default_processed_data_dir, prepared_paths
from deeplsh.cicids.runtime import artifact_paths


MODEL_ORDER = ["exact-md5", "simhash", "baseline-mlp", "bigru-deeplsh"]
METRIC_ORDER = ["accuracy", "precision", "recall", "f1"]
DISPLAY_NAMES = {
    "exact-md5": "Exact MD5",
    "simhash": "SimHash",
    "baseline-mlp": "MLP-DeepLSH",
    "bigru-deeplsh": "BiGRU-DeepLSH",
}


def _read_required_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def _validate_required_artifacts(output_dir: str) -> None:
    for key in ["pairs", "flows_tokens"]:
        path = prepared_paths(output_dir)[key]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required prepared data not found: {path}")

    for model_type in ["mlp", "bigru"]:
        paths = artifact_paths(model_type)
        for key in ["embeddings", "embeddings_hamming"]:
            if not os.path.exists(paths[key]):
                raise FileNotFoundError(f"Required {model_type} artifact not found: {paths[key]}")


def load_model_metrics(results_dir: str) -> pd.DataFrame:
    baseline_path = os.path.join(results_dir, "cicids_baseline_metrics.csv")
    bigru_path = os.path.join(results_dir, "cicids_bigru_metrics.csv")
    metrics_df = pd.concat([_read_required_csv(baseline_path), _read_required_csv(bigru_path)], ignore_index=True)
    metrics_df["model"] = pd.Categorical(metrics_df["model"], categories=MODEL_ORDER, ordered=True)
    metrics_df = metrics_df.sort_values("model").reset_index(drop=True)
    metrics_df["display_model"] = metrics_df["model"].astype(str).map(DISPLAY_NAMES)
    return metrics_df


def load_correlation(results_dir: str) -> pd.DataFrame:
    correlation_path = os.path.join(results_dir, "cicids_lsh_correlation_jaccard_bigru.csv")
    correlation_df = _read_required_csv(correlation_path)
    for column in ["true_sim", "pred_sim"]:
        if not ((correlation_df[column] >= 0).all() and (correlation_df[column] <= 1).all()):
            raise ValueError(f"{column} values must be in [0, 1]")
    return correlation_df


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def plot_model_metric_comparison(metrics_df: pd.DataFrame, output_path: str) -> None:
    metric_long = metrics_df.melt(
        id_vars=["model", "display_model"],
        value_vars=METRIC_ORDER,
        var_name="metric",
        value_name="score",
    )
    metric_long["metric"] = pd.Categorical(metric_long["metric"], categories=METRIC_ORDER, ordered=True)

    plt.figure(figsize=(8.2, 4.8))
    ax = sns.barplot(data=metric_long, x="metric", y="score", hue="display_model", palette="Set2")
    ax.set_title("Model Performance Comparison")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model", loc="lower right", frameon=True)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix_comparison(metrics_df: pd.DataFrame, output_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.2))
    axes = axes.flatten()
    for ax, (_, row) in zip(axes, metrics_df.iterrows()):
        matrix = np.asarray([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=int)
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Pred non-dup", "Pred dup"],
            yticklabels=["True non-dup", "True dup"],
            ax=ax,
        )
        ax.set_title(DISPLAY_NAMES.get(str(row["model"]), str(row["model"])))
        ax.set_xlabel("")
        ax.set_ylabel("")
    plt.suptitle("Confusion Matrix Comparison", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_lsh_correlation_regplot(correlation_df: pd.DataFrame, pearson: float, spearman: float, output_path: str) -> None:
    plt.figure(figsize=(6.4, 5.2))
    ax = sns.regplot(
        data=correlation_df,
        x="true_sim",
        y="pred_sim",
        scatter_kws={"s": 4, "alpha": 0.08, "color": "#1f77b4", "marker": "x"},
        line_kws={"color": "#d62728", "linewidth": 1.4},
        ci=95,
    )
    ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1, label="y=x")
    ax.text(
        0.03,
        0.97,
        f"Pearson = {pearson:.3f}\nSpearman = {spearman:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
    )
    ax.set_title("Jaccard vs Hamming Similarity")
    ax.set_xlabel("Jaccard similarity values")
    ax.set_ylabel("Hamming similarity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_hamming_similarity_distribution(correlation_df: pd.DataFrame, output_path: str) -> None:
    plot_df = correlation_df.copy()
    plot_df["pair_type"] = np.where(plot_df["is_duplicate"].astype(int) == 1, "Duplicate", "Non-duplicate")

    plt.figure(figsize=(6.8, 4.8))
    ax = sns.violinplot(data=plot_df, x="pair_type", y="pred_sim", inner="quartile", palette="Set2", cut=0)
    sns.stripplot(data=plot_df.sample(n=min(2000, len(plot_df)), random_state=42), x="pair_type", y="pred_sim", color="black", alpha=0.08, size=1, ax=ax)
    ax.set_title("Hamming Similarity Distribution")
    ax.set_xlabel("Pair type")
    ax.set_ylabel("Hamming similarity")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def build_summary(metrics_df: pd.DataFrame, correlation_df: pd.DataFrame) -> pd.DataFrame:
    pearson = float(correlation_df["true_sim"].corr(correlation_df["pred_sim"], method="pearson"))
    spearman = float(correlation_df["true_sim"].corr(correlation_df["pred_sim"], method="spearman"))
    best_f1 = float(metrics_df.loc[metrics_df["model"].astype(str) == "bigru-deeplsh", "f1"].iloc[0])
    simhash_f1 = float(metrics_df.loc[metrics_df["model"].astype(str) == "simhash", "f1"].iloc[0])
    mlp_f1 = float(metrics_df.loc[metrics_df["model"].astype(str) == "baseline-mlp", "f1"].iloc[0])

    summary = metrics_df.copy()
    summary["jaccard_hamming_pearson"] = pearson
    summary["jaccard_hamming_spearman"] = spearman
    summary["best_model"] = "bigru-deeplsh"
    summary["bigru_f1_minus_simhash"] = best_f1 - simhash_f1
    summary["bigru_f1_minus_baseline_mlp"] = best_f1 - mlp_f1
    summary["pred_sim_duplicate_median"] = float(correlation_df.loc[correlation_df["is_duplicate"] == 1, "pred_sim"].median())
    summary["pred_sim_non_duplicate_median"] = float(correlation_df.loc[correlation_df["is_duplicate"] == 0, "pred_sim"].median())
    return summary.drop(columns=["display_model"])


def run(output_dir: str, results_dir: str, figures_dir: str) -> Dict[str, object]:
    _validate_required_artifacts(output_dir)
    os.makedirs(figures_dir, exist_ok=True)
    configure_plotting()

    metrics_df = load_model_metrics(results_dir)
    correlation_df = load_correlation(results_dir)
    pearson = float(correlation_df["true_sim"].corr(correlation_df["pred_sim"], method="pearson"))
    spearman = float(correlation_df["true_sim"].corr(correlation_df["pred_sim"], method="spearman"))

    outputs = {
        "model_metric_comparison": os.path.join(figures_dir, "model_metric_comparison.png"),
        "confusion_matrix_comparison": os.path.join(figures_dir, "confusion_matrix_comparison.png"),
        "lsh_correlation_regplot": os.path.join(figures_dir, "lsh_correlation_regplot.png"),
        "hamming_similarity_distribution": os.path.join(figures_dir, "hamming_similarity_distribution.png"),
        "paper_results_summary": os.path.join(figures_dir, "paper_results_summary.csv"),
    }

    plot_model_metric_comparison(metrics_df, outputs["model_metric_comparison"])
    plot_confusion_matrix_comparison(metrics_df, outputs["confusion_matrix_comparison"])
    plot_lsh_correlation_regplot(correlation_df, pearson, spearman, outputs["lsh_correlation_regplot"])
    plot_hamming_similarity_distribution(correlation_df, outputs["hamming_similarity_distribution"])
    build_summary(metrics_df, correlation_df).to_csv(outputs["paper_results_summary"], index=False)

    return {
        "outputs": outputs,
        "model_count": int(metrics_df.shape[0]),
        "correlation_rows": int(correlation_df.shape[0]),
        "pearson": pearson,
        "spearman": spearman,
        "bigru_f1": float(metrics_df.loc[metrics_df["model"].astype(str) == "bigru-deeplsh", "f1"].iloc[0]),
        "simhash_f1": float(metrics_df.loc[metrics_df["model"].astype(str) == "simhash", "f1"].iloc[0]),
        "baseline_mlp_f1": float(metrics_df.loc[metrics_df["model"].astype(str) == "baseline-mlp", "f1"].iloc[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready CIC-IDS experiment figures.")
    parser.add_argument("--output-dir", default=None, help="Prepared CIC-IDS data directory")
    parser.add_argument("--results-dir", default=None, help="Directory containing full experiment CSV results")
    parser.add_argument("--figures-dir", default=None, help="Directory to write paper figures")
    args = parser.parse_args()

    output_dir = args.output_dir or default_processed_data_dir()
    results_dir = args.results_dir or str(cicids_artifacts_dir() / "results" / "full")
    figures_dir = args.figures_dir or os.path.join(results_dir, "figures")
    summary = run(output_dir=output_dir, results_dir=results_dir, figures_dir=figures_dir)
    output_text = " ".join([f"{key}={value}" for key, value in summary["outputs"].items()])
    print(
        " ".join(
            [
                "done",
                "mode=cicids-plot-paper-results",
                f"model_count={summary['model_count']}",
                f"correlation_rows={summary['correlation_rows']}",
                f"pearson={summary['pearson']:.6f}",
                f"spearman={summary['spearman']:.6f}",
                f"bigru_f1={summary['bigru_f1']:.6f}",
                f"simhash_f1={summary['simhash_f1']:.6f}",
                f"baseline_mlp_f1={summary['baseline_mlp_f1']:.6f}",
                output_text,
            ]
        )
    )


if __name__ == "__main__":
    main()
