import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from deeplsh._paths import cicids_artifacts_dir
from deeplsh.cicids.export_matlab_plot_data import (
    build_lsh_hyperparam_fscore,
    build_multi_similarity_correlation,
    build_retrieval_performance,
)
from deeplsh.cicids.pipeline import default_processed_data_dir


PAPER_MODEL = "BiGRU-DeepLSH-Paper"
DETECTION_BIGRU = "BiGRU-DeepLSH-Detection"
DETECTION_MLP = "MLP-DeepLSH-Detection"
SIMHASH = "SimHash"


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def _paper_paths(output_dir: str, export_dir: str) -> Dict[str, Path]:
    results_root = cicids_artifacts_dir() / "results" / Path(output_dir).name
    return {
        "export_dir": Path(export_dir),
        "paper_pairs": Path(output_dir) / "pairs_similarity_jaccard.csv",
        "paper_correlation": results_root / "paper_lsh" / "cicids_lsh_correlation_jaccard_bigru_paper.csv",
        "paper_metadata": cicids_artifacts_dir() / "models" / "cicids_bigru_jaccard_paper_metadata.json",
        "paper_hamming": cicids_artifacts_dir() / "models" / "cicids_bigru_jaccard_paper_embeddings_hamming.npy",
    }


def load_paper_jaccard(paths: Dict[str, Path]) -> pd.DataFrame:
    _ensure_file(paths["paper_correlation"])
    _ensure_file(paths["paper_metadata"])
    correlation_df = pd.read_csv(paths["paper_correlation"])
    with open(paths["paper_metadata"], "r", encoding="utf-8") as f:
        metadata = json.load(f)

    required = ["flow_index_1", "flow_index_2", "label_1", "label_2", "true_sim", "pred_sim"]
    missing = [column for column in required if column not in correlation_df.columns]
    if missing:
        raise ValueError(f"Paper correlation CSV is missing columns: {missing}")
    for column in ["true_sim", "pred_sim"]:
        if not ((correlation_df[column] >= 0).all() and (correlation_df[column] <= 1).all()):
            raise ValueError(f"{column} values must be in [0, 1]")

    result = correlation_df.copy()
    result["collision_probability"] = result["pred_sim"].astype(float)
    result["M"] = int(metadata["M"])
    result["b"] = int(metadata["b"])
    result["hash_bits"] = int(metadata.get("hash_bits", int(metadata["M"]) * int(metadata["b"])))
    result["method"] = PAPER_MODEL
    result["model_source"] = "paper_faithful_jaccard_mse"
    result["evidence_level"] = "main"
    return result


def build_exploratory_correlation(multi_df: pd.DataFrame, paper_df: pd.DataFrame) -> pd.DataFrame:
    method_specs = [
        (DETECTION_BIGRU, "bigru_hamming_similarity", "detection_bigru_hamming", "exploratory"),
        (DETECTION_MLP, "mlp_hamming_similarity", "detection_mlp_hamming", "exploratory"),
        (SIMHASH, "simhash_similarity", "simhash_baseline", "exploratory"),
    ]
    rows: List[pd.DataFrame] = []
    common_columns = ["metric", "pair_index", "flow_index_1", "flow_index_2", "label_1", "label_2", "true_sim", "is_duplicate"]
    for method, prediction_column, source, level in method_specs:
        frame = multi_df[common_columns].copy()
        frame["pred_sim"] = multi_df[prediction_column].astype(float)
        frame["method"] = method
        frame["model_source"] = source
        frame["evidence_level"] = level
        rows.append(frame)

    paper_frame = paper_df[
        ["flow_index_1", "flow_index_2", "label_1", "label_2", "true_sim", "pred_sim", "method", "model_source", "evidence_level"]
    ].copy()
    paper_frame.insert(0, "metric", "Jaccard")
    paper_frame.insert(1, "pair_index", np.arange(paper_frame.shape[0], dtype=int))
    paper_frame["is_duplicate"] = (paper_frame["label_1"].astype(str) == paper_frame["label_2"].astype(str)).astype(int)
    rows.append(paper_frame)

    result = pd.concat(rows, ignore_index=True)
    if not ((result["true_sim"] >= 0).all() and (result["true_sim"] <= 1).all()):
        raise ValueError("true_sim values must be in [0, 1]")
    if not ((result["pred_sim"] >= 0).all() and (result["pred_sim"] <= 1).all()):
        raise ValueError("pred_sim values must be in [0, 1]")
    return result


def build_kendall_tau_comparison(exploratory_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (metric, method), group in exploratory_df.groupby(["metric", "method"], sort=False):
        tau, p_value = kendalltau(group["true_sim"], group["pred_sim"])
        source = str(group["model_source"].iloc[0])
        level = str(group["evidence_level"].iloc[0])
        rows.append(
            {
                "metric": metric,
                "method": method,
                "kendall_tau": float(tau) if not np.isnan(tau) else 0.0,
                "p_value": float(p_value) if not np.isnan(p_value) else 1.0,
                "model_source": source,
                "evidence_level": level,
            }
        )
    return pd.DataFrame.from_records(rows)


def build_extra_diagnostics(paper_df: pd.DataFrame, hash_codes: np.ndarray) -> pd.DataFrame:
    bit_positive_rate = np.mean(hash_codes > 0, axis=0)
    bit_entropy = -(
        bit_positive_rate * np.log2(np.clip(bit_positive_rate, 1e-12, 1.0))
        + (1.0 - bit_positive_rate) * np.log2(np.clip(1.0 - bit_positive_rate, 1e-12, 1.0))
    )
    rows = [
        {
            "diagnostic_type": "hash_bit_entropy",
            "group": "bit",
            "name": int(bit_index),
            "count": int(hash_codes.shape[0]),
            "value": float(entropy),
            "mean_pred_sim": np.nan,
            "median_pred_sim": np.nan,
            "mean_true_sim": np.nan,
        }
        for bit_index, entropy in enumerate(bit_entropy)
    ]

    label_equal = paper_df["label_1"].astype(str) == paper_df["label_2"].astype(str)
    for group_name, mask in [("same_label", label_equal), ("different_label", ~label_equal)]:
        group = paper_df[mask]
        rows.append(
            {
                "diagnostic_type": "label_pair_pred_sim",
                "group": group_name,
                "name": group_name,
                "count": int(group.shape[0]),
                "value": float(group["pred_sim"].mean()) if not group.empty else np.nan,
                "mean_pred_sim": float(group["pred_sim"].mean()) if not group.empty else np.nan,
                "median_pred_sim": float(group["pred_sim"].median()) if not group.empty else np.nan,
                "mean_true_sim": float(group["true_sim"].mean()) if not group.empty else np.nan,
            }
        )

    bins = pd.cut(paper_df["true_sim"], bins=np.linspace(0, 1, 11), include_lowest=True, labels=False)
    for bin_id in range(10):
        group = paper_df[bins == bin_id]
        rows.append(
            {
                "diagnostic_type": "jaccard_bin_calibration",
                "group": "jaccard_bin",
                "name": int(bin_id),
                "count": int(group.shape[0]),
                "value": float(group["pred_sim"].mean()) if not group.empty else np.nan,
                "mean_pred_sim": float(group["pred_sim"].mean()) if not group.empty else np.nan,
                "median_pred_sim": float(group["pred_sim"].median()) if not group.empty else np.nan,
                "mean_true_sim": float(group["true_sim"].mean()) if not group.empty else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows)


def write_preview_figures(
    export_dir: Path,
    paper_df: pd.DataFrame,
    exploratory_df: pd.DataFrame,
    kendall_df: pd.DataFrame,
    retrieval_df: pd.DataFrame,
    hyper_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
) -> Dict[str, str]:
    figures_dir = export_dir / "preview_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}
    paper_color = "#145f84"
    detection_color = "#e84132"
    simhash_color = "#379094"

    plt.figure(figsize=(6.2, 5.2))
    plt.scatter(paper_df["true_sim"], paper_df["pred_sim"], s=4, alpha=0.12, marker="x", color=paper_color)
    plt.plot([0, 1], [0, 1], "r--", linewidth=1)
    plt.title("Jaccard (BiGRU-DeepLSH Paper)")
    plt.xlabel("True similarity")
    plt.ylabel("Collision probability")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    path = figures_dir / "paper_jaccard_correlation_preview.png"
    plt.savefig(path, dpi=300)
    plt.close()
    outputs["preview_paper_jaccard_correlation"] = str(path)

    metrics = list(exploratory_df["metric"].drop_duplicates())[:6]
    methods = [PAPER_MODEL, DETECTION_BIGRU, SIMHASH]
    colors = {PAPER_MODEL: paper_color, DETECTION_BIGRU: detection_color, SIMHASH: simhash_color}
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), sharex=True, sharey=True)
    for axis, metric in zip(axes.ravel(), metrics):
        for method in methods:
            subset = exploratory_df[(exploratory_df["metric"] == metric) & (exploratory_df["method"] == method)]
            if subset.empty:
                continue
            if subset.shape[0] > 5000:
                subset = subset.sample(n=5000, random_state=42)
            axis.scatter(subset["true_sim"], subset["pred_sim"], s=4, alpha=0.10, marker="x", color=colors[method], label=method)
        axis.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        axis.set_title(metric)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(True, linestyle=":", alpha=0.5)
    for axis in axes[:, 0]:
        axis.set_ylabel("Predicted similarity")
    for axis in axes[-1, :]:
        axis.set_xlabel("True similarity")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    path = figures_dir / "multi_similarity_correlation_exploratory_preview.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    outputs["preview_multi_similarity_correlation"] = str(path)

    pivot = kendall_df.pivot_table(index="metric", columns="method", values="kendall_tau", aggfunc="first")
    ax = pivot.plot(kind="bar", figsize=(11.8, 5.6), ylim=(0, 1), grid=True)
    ax.set_ylabel("Kendall tau coefficient")
    ax.set_xlabel("Similarity measures")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    path = figures_dir / "kendall_tau_comparison_preview.png"
    plt.savefig(path, dpi=300)
    plt.close()
    outputs["preview_kendall_tau_comparison"] = str(path)

    retrieval_plot = retrieval_df.set_index("method")[["rr_at_1", "rr_at_5", "rr_at_10", "mrr"]]
    ax = retrieval_plot.plot(kind="bar", figsize=(9.8, 5.4), ylim=(0, 1), grid=True)
    ax.set_ylabel("Retrieval score")
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = figures_dir / "retrieval_performance_preview.png"
    plt.savefig(path, dpi=300)
    plt.close()
    outputs["preview_retrieval_performance"] = str(path)

    methods_hyper = list(hyper_df["method"].drop_duplicates())
    fig, axes = plt.subplots(1, len(methods_hyper), figsize=(12.5, 5.2), sharey=True)
    if len(methods_hyper) == 1:
        axes = [axes]
    for axis, method in zip(axes, methods_hyper):
        subset = hyper_df[hyper_df["method"] == method].copy()
        labels = [f"({int(row.L)},{int(row.K)})" for row in subset.itertuples(index=False)]
        order = list(dict.fromkeys(labels))
        data = [subset[np.asarray(labels) == label]["f1"].to_numpy() for label in order]
        axis.boxplot(data, labels=order)
        axis.set_title(method)
        axis.set_xlabel("LSH hyperparameters (L,K)")
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(True, linestyle=":", alpha=0.5)
    axes[0].set_ylabel("F1-score")
    fig.tight_layout()
    path = figures_dir / "lsh_hyperparam_fscore_boxplot_preview.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    outputs["preview_lsh_hyperparam_fscore_boxplot"] = str(path)

    calibration = diagnostics_df[diagnostics_df["diagnostic_type"] == "jaccard_bin_calibration"]
    plt.figure(figsize=(7.6, 5.0))
    plt.plot(calibration["mean_true_sim"], calibration["mean_pred_sim"], "-o", color=paper_color)
    plt.plot([0, 1], [0, 1], "r--", linewidth=1)
    plt.xlabel("True similarity")
    plt.ylabel("Collision probability")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    path = figures_dir / "jaccard_bin_calibration_preview.png"
    plt.savefig(path, dpi=300)
    plt.close()
    outputs["preview_jaccard_bin_calibration"] = str(path)

    same = paper_df[paper_df["label_1"].astype(str) == paper_df["label_2"].astype(str)]["pred_sim"].to_numpy()
    different = paper_df[paper_df["label_1"].astype(str) != paper_df["label_2"].astype(str)]["pred_sim"].to_numpy()
    plt.figure(figsize=(6.2, 5.0))
    plt.boxplot([same, different], labels=["Same label", "Different label"])
    plt.ylabel("Collision probability")
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    path = figures_dir / "pred_similarity_by_label_pair_preview.png"
    plt.savefig(path, dpi=300)
    plt.close()
    outputs["preview_pred_similarity_by_label_pair"] = str(path)

    entropy = diagnostics_df[diagnostics_df["diagnostic_type"] == "hash_bit_entropy"]["value"]
    plt.figure(figsize=(7.6, 5.0))
    plt.hist(entropy, bins=30, color=paper_color)
    plt.xlabel("Bit entropy")
    plt.ylabel("Count")
    plt.xlim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    path = figures_dir / "hash_bit_entropy_preview.png"
    plt.savefig(path, dpi=300)
    plt.close()
    outputs["preview_hash_bit_entropy"] = str(path)
    return outputs


def matlab_script(language: str) -> str:
    is_cn = language == "cn"
    labels = {
        "script_comment": "从 CSV 绘制 DeepLSH 论文图。" if is_cn else "Draw DeepLSH paper figures from exported CSV files.",
        "fig1": "图1：Jaccard 论文主相关性图" if is_cn else "Figure 1: Paper Jaccard correlation",
        "fig2": "图2：多相似度相关性散点图（探索性）" if is_cn else "Figure 2: Multi-similarity correlation (Exploratory)",
        "fig3": "图3：Kendall tau 系数对比" if is_cn else "Figure 3: Kendall tau coefficient comparison",
        "fig4": "图4：检索性能对比" if is_cn else "Figure 4: Retrieval performance comparison",
        "fig5": "图5：LSH 参数 F1 箱线图" if is_cn else "Figure 5: LSH hyperparameter F1 boxplot",
        "fig6": "图6：Jaccard 分箱校准曲线" if is_cn else "Figure 6: Jaccard bin calibration",
        "fig7": "图7：正负样本预测相似度分布" if is_cn else "Figure 7: Predicted similarity by label pair",
        "fig8": "图8：哈希位熵分布" if is_cn else "Figure 8: Hash bit entropy distribution",
        "true": "真实相似度" if is_cn else "True similarity",
        "pred": "碰撞概率 / 预测相似度" if is_cn else "Collision probability / predicted similarity",
        "collision": "碰撞概率" if is_cn else "Collision probability",
        "kendall": "Kendall tau 系数" if is_cn else "Kendall tau coefficient",
        "retrieval": "检索得分" if is_cn else "Retrieval score",
        "f1": "F1 值" if is_cn else "F1-score",
        "params": "LSH 参数 (L,K)" if is_cn else "LSH hyperparameters (L,K)",
        "bin": "Jaccard 分箱" if is_cn else "Jaccard bin",
        "entropy": "哈希位熵" if is_cn else "Bit entropy",
    }
    suffix = "_cn" if is_cn else ""
    return f"""% {labels['script_comment']}
% Put this script in the same directory as the generated CSV files, then run it in MATLAB.

clear; clc;

scriptPath = mfilename('fullpath');
if isempty(scriptPath)
    dataDir = pwd;
else
    dataDir = fileparts(scriptPath);
end
figDir = fullfile(dataDir, 'matlab_figures');
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

paper = readtable(fullfile(dataDir, 'paper_jaccard_correlation.csv'));
multi = readtable(fullfile(dataDir, 'multi_similarity_correlation_exploratory.csv'));
kendall = readtable(fullfile(dataDir, 'kendall_tau_comparison.csv'));
retrieval = readtable(fullfile(dataDir, 'retrieval_performance.csv'));
hyper = readtable(fullfile(dataDir, 'lsh_hyperparam_fscore.csv'));
diag = readtable(fullfile(dataDir, 'paper_extra_diagnostics.csv'));

deepColor = [0.91 0.25 0.20];
simhashColor = [0.22 0.56 0.58];
mlpColor = [0.42 0.50 0.62];
paperColor = [0.08 0.38 0.52];

%% {labels['fig1']}
figure('Color', 'w', 'Position', [80 80 620 520]);
scatter(paper.true_sim, paper.pred_sim, 5, 'x', 'MarkerEdgeAlpha', 0.12, 'MarkerEdgeColor', paperColor);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 1);
title('Jaccard (BiGRU-DeepLSH Paper)');
xlabel('{labels['true']}');
ylabel('{labels['collision']}');
xlim([0 1]); ylim([0 1]); grid on;
exportgraphics(gcf, fullfile(figDir, 'paper_jaccard_correlation{suffix}.png'), 'Resolution', 300);

%% {labels['fig2']}
methodsToPlot = {{'BiGRU-DeepLSH-Paper', 'BiGRU-DeepLSH-Detection', 'SimHash'}};
metrics = unique(multi.metric, 'stable');
figure('Color', 'w', 'Position', [60 60 1500 760]);
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:min(numel(metrics), 6)
    nexttile;
    hold on;
    for j = 1:numel(methodsToPlot)
        method = methodsToPlot{{j}};
        idx = strcmp(multi.metric, metrics{{i}}) & strcmp(multi.method, method);
        if any(idx)
            color = paperColor;
            if strcmp(method, 'BiGRU-DeepLSH-Detection'), color = deepColor; end
            if strcmp(method, 'SimHash'), color = simhashColor; end
            scatter(multi.true_sim(idx), multi.pred_sim(idx), 4, 'x', ...
                'MarkerEdgeAlpha', 0.08, 'MarkerEdgeColor', color);
        end
    end
    plot([0 1], [0 1], 'k--', 'LineWidth', 0.8);
    title(sprintf('%s', metrics{{i}}), 'Interpreter', 'none');
    xlabel('{labels['true']}');
    ylabel('{labels['pred']}');
    xlim([0 1]); ylim([0 1]); grid on;
end
legend(methodsToPlot, 'Location', 'southoutside', 'Orientation', 'horizontal', 'Interpreter', 'none');
exportgraphics(gcf, fullfile(figDir, 'multi_similarity_correlation_exploratory{suffix}.png'), 'Resolution', 300);

%% {labels['fig3']}
figure('Color', 'w', 'Position', [100 100 1180 560]);
mainRows = strcmp(kendall.evidence_level, 'main') | strcmp(kendall.method, 'BiGRU-DeepLSH-Detection') | strcmp(kendall.method, 'MLP-DeepLSH-Detection') | strcmp(kendall.method, 'SimHash');
K = kendall(mainRows, :);
metrics = unique(K.metric, 'stable');
methods = unique(K.method, 'stable');
Y = nan(numel(metrics), numel(methods));
for i = 1:numel(metrics)
    for j = 1:numel(methods)
        idx = strcmp(K.metric, metrics{{i}}) & strcmp(K.method, methods{{j}});
        if any(idx), Y(i, j) = K.kendall_tau(find(idx, 1)); end
    end
end
bar(Y);
set(gca, 'XTickLabel', metrics);
xtickangle(35);
ylabel('{labels['kendall']}');
xlabel('Similarity measures');
ylim([0 1]);
legend(methods, 'Location', 'northoutside', 'Orientation', 'horizontal', 'Interpreter', 'none');
grid on;
exportgraphics(gcf, fullfile(figDir, 'kendall_tau_comparison{suffix}.png'), 'Resolution', 300);

%% {labels['fig4']}
retrievalMethods = string(retrieval.method);
retrievalMetrics = [retrieval.rr_at_1, retrieval.rr_at_5, retrieval.rr_at_10, retrieval.mrr];
figure('Color', 'w', 'Position', [120 120 980 540]);
bar(retrievalMetrics);
set(gca, 'XTickLabel', retrievalMethods);
xtickangle(20);
ylabel('{labels['retrieval']}');
ylim([0 1]);
legend({{'RR@1', 'RR@5', 'RR@10', 'MRR'}}, 'Location', 'southoutside', 'Orientation', 'horizontal');
grid on;
exportgraphics(gcf, fullfile(figDir, 'retrieval_performance{suffix}.png'), 'Resolution', 300);

%% {labels['fig5']}
methods = unique(hyper.method, 'stable');
figure('Color', 'w', 'Position', [140 140 1250 520]);
tiledlayout(1, numel(methods), 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:numel(methods)
    nexttile;
    idx = strcmp(hyper.method, methods{{i}});
    H = hyper(idx, :);
    labelsLK = strcat('(', string(H.L), ',', string(H.K), ')');
    orderedLabels = unique(labelsLK, 'stable');
    group = categorical(labelsLK, orderedLabels, 'Ordinal', true);
    if exist('boxchart', 'file') == 2
        boxchart(group, H.f1);
    else
        boxplot(H.f1, group);
    end
    title(sprintf('%s', methods{{i}}), 'Interpreter', 'none');
    xlabel('{labels['params']}');
    ylabel('{labels['f1']}');
    ylim([0 1]);
    xtickangle(35);
    grid on;
end
exportgraphics(gcf, fullfile(figDir, 'lsh_hyperparam_fscore_boxplot{suffix}.png'), 'Resolution', 300);

%% {labels['fig6']}
cal = diag(strcmp(diag.diagnostic_type, 'jaccard_bin_calibration'), :);
figure('Color', 'w', 'Position', [160 160 760 500]);
plot(cal.mean_true_sim, cal.mean_pred_sim, '-o', 'Color', paperColor, 'LineWidth', 1.4, 'MarkerFaceColor', paperColor);
hold on;
plot([0 1], [0 1], 'r--', 'LineWidth', 1);
xlabel('{labels['true']}');
ylabel('{labels['collision']}');
title('{labels['fig6']}');
xlim([0 1]); ylim([0 1]); grid on;
exportgraphics(gcf, fullfile(figDir, 'jaccard_bin_calibration{suffix}.png'), 'Resolution', 300);

%% {labels['fig7']}
figure('Color', 'w', 'Position', [180 180 620 500]);
same = strcmp(paper.label_1, paper.label_2);
group = categorical(same, [true false], {{'Same label', 'Different label'}});
if exist('boxchart', 'file') == 2
    boxchart(group, paper.pred_sim);
else
    boxplot(paper.pred_sim, group);
end
ylabel('{labels['collision']}');
title('{labels['fig7']}');
ylim([0 1]); grid on;
exportgraphics(gcf, fullfile(figDir, 'pred_similarity_by_label_pair{suffix}.png'), 'Resolution', 300);

%% {labels['fig8']}
entropyRows = diag(strcmp(diag.diagnostic_type, 'hash_bit_entropy'), :);
figure('Color', 'w', 'Position', [200 200 760 500]);
histogram(entropyRows.value, 30, 'FaceColor', paperColor, 'EdgeColor', 'none');
xlabel('{labels['entropy']}');
ylabel('Count');
title('{labels['fig8']}');
xlim([0 1]); grid on;
exportgraphics(gcf, fullfile(figDir, 'hash_bit_entropy{suffix}.png'), 'Resolution', 300);

disp('MATLAB figures written to:');
disp(figDir);
"""


def run(output_dir: str, export_dir: str, query_limit: int, folds: int) -> Dict[str, str]:
    paths = _paper_paths(output_dir, export_dir)
    paths["export_dir"].mkdir(parents=True, exist_ok=True)
    _ensure_file(paths["paper_pairs"])
    _ensure_file(paths["paper_hamming"])

    paper_df = load_paper_jaccard(paths)
    multi_df = build_multi_similarity_correlation(output_dir)
    exploratory_df = build_exploratory_correlation(multi_df, paper_df)
    kendall_df = build_kendall_tau_comparison(exploratory_df)
    retrieval_df = build_retrieval_performance(output_dir, query_limit=query_limit, top_ks=[1, 5, 10])
    hyper_df = build_lsh_hyperparam_fscore(output_dir, folds=folds)
    diagnostics_df = build_extra_diagnostics(paper_df, np.load(paths["paper_hamming"]))
    preview_outputs = write_preview_figures(
        export_dir=paths["export_dir"],
        paper_df=paper_df,
        exploratory_df=exploratory_df,
        kendall_df=kendall_df,
        retrieval_df=retrieval_df,
        hyper_df=hyper_df,
        diagnostics_df=diagnostics_df,
    )

    outputs = {
        "paper_jaccard_correlation": paths["export_dir"] / "paper_jaccard_correlation.csv",
        "multi_similarity_correlation_exploratory": paths["export_dir"] / "multi_similarity_correlation_exploratory.csv",
        "kendall_tau_comparison": paths["export_dir"] / "kendall_tau_comparison.csv",
        "retrieval_performance": paths["export_dir"] / "retrieval_performance.csv",
        "lsh_hyperparam_fscore": paths["export_dir"] / "lsh_hyperparam_fscore.csv",
        "paper_extra_diagnostics": paths["export_dir"] / "paper_extra_diagnostics.csv",
        "matlab_script_en": paths["export_dir"] / "draw_paper_figures.m",
        "matlab_script_cn": paths["export_dir"] / "draw_paper_figures_cn.m",
    }
    paper_df.to_csv(outputs["paper_jaccard_correlation"], index=False)
    exploratory_df.to_csv(outputs["multi_similarity_correlation_exploratory"], index=False)
    kendall_df.to_csv(outputs["kendall_tau_comparison"], index=False)
    retrieval_df.to_csv(outputs["retrieval_performance"], index=False)
    hyper_df.to_csv(outputs["lsh_hyperparam_fscore"], index=False)
    diagnostics_df.to_csv(outputs["paper_extra_diagnostics"], index=False)
    outputs["matlab_script_en"].write_text(matlab_script("en"), encoding="utf-8")
    outputs["matlab_script_cn"].write_text(matlab_script("cn"), encoding="utf-8")
    text_outputs = {key: str(value) for key, value in outputs.items()}
    text_outputs.update(preview_outputs)
    return text_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper-faithful MATLAB-ready CIC-IDS DeepLSH figures.")
    parser.add_argument("--output-dir", default=None, help="Prepared CIC-IDS data directory")
    parser.add_argument("--export-dir", default=None, help="Directory to write CSV files and MATLAB scripts")
    parser.add_argument("--query-limit", type=int, default=120, help="Number of evenly spaced queries for retrieval metrics")
    parser.add_argument("--folds", type=int, default=10, help="Number of folds for LSH hyperparameter F1 boxplot data")
    args = parser.parse_args()

    output_dir = args.output_dir or default_processed_data_dir()
    export_dir = args.export_dir or str(cicids_artifacts_dir() / "results" / Path(output_dir).name / "matlab_data_paper")
    outputs = run(output_dir=output_dir, export_dir=export_dir, query_limit=args.query_limit, folds=args.folds)
    print(
        " ".join(
            [
                "done",
                "mode=cicids-export-paper-matlab-data",
                f"query_limit={args.query_limit}",
                f"folds={args.folds}",
                *[f"{key}={value}" for key, value in outputs.items()],
            ]
        )
    )


if __name__ == "__main__":
    main()
