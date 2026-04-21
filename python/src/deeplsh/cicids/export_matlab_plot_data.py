import argparse
import hashlib
import json
import os
import time
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.feature_extraction.text import TfidfVectorizer

from deeplsh._paths import cicids_artifacts_dir
from deeplsh.cicids.pipeline import default_processed_data_dir, load_prepared_flows, load_prepared_pairs, load_prepared_token_flows
from deeplsh.cicids.runtime import artifact_paths, classification_metrics, simhash_pair_scores, simhash_signatures


METRIC_ORDER = ["Jaccard", "Jaccard-bigram", "Cosine-TF", "Cosine-TFIDF", "Flow-cosine", "Levenshtein"]
LSH_PARAMS = [(1, 64), (2, 32), (4, 16), (8, 8), (16, 4), (32, 2), (64, 1)]


def _ensure_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")


def _load_hash_codes(model_type: str) -> np.ndarray:
    path = artifact_paths(model_type)["embeddings_hamming"]
    _ensure_file(path)
    return np.load(path)


def _load_lsh_meta(model_type: str) -> Dict[str, int]:
    path = artifact_paths(model_type)["train_meta"]
    _ensure_file(path)
    with open(path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return {key: int(value) for key, value in metadata["lsh"].items()}


def _token_lists(token_sequences: pd.Series) -> List[List[str]]:
    return [str(sequence).split() for sequence in token_sequences.fillna("")]


def _jaccard(values_a: Iterable, values_b: Iterable) -> float:
    set_a = set(values_a)
    set_b = set(values_b)
    union = set_a | set_b
    if not union:
        return 0.0
    return float(len(set_a & set_b) / len(union))


def _bigrams(tokens: List[str]) -> List[Tuple[str, str]]:
    return list(zip(tokens, tokens[1:]))


def _counter_cosine(counter_a: Counter, counter_b: Counter) -> float:
    if not counter_a or not counter_b:
        return 0.0
    common = set(counter_a) & set(counter_b)
    dot = sum(counter_a[token] * counter_b[token] for token in common)
    norm_a = sum(value * value for value in counter_a.values()) ** 0.5
    norm_b = sum(value * value for value in counter_b.values()) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _cosine_01_pairs(matrix: np.ndarray, idx1: np.ndarray, idx2: np.ndarray) -> np.ndarray:
    vectors_1 = matrix[idx1]
    vectors_2 = matrix[idx2]
    norms = np.linalg.norm(vectors_1, axis=1) * np.linalg.norm(vectors_2, axis=1)
    dots = np.sum(vectors_1 * vectors_2, axis=1)
    scores = np.divide(dots, norms, out=np.zeros_like(dots, dtype=np.float32), where=norms != 0)
    scores = np.clip(scores, -1.0, 1.0)
    return ((scores + 1.0) / 2.0).astype(np.float32)


def _hamming_pair_similarity(hash_codes: np.ndarray, idx1: np.ndarray, idx2: np.ndarray) -> np.ndarray:
    return np.mean(hash_codes[idx1] == hash_codes[idx2], axis=1).astype(np.float32)


def _band_collision_scores(hash_codes: np.ndarray, idx1: np.ndarray, idx2: np.ndarray, L: int, K: int, b: int) -> np.ndarray:
    band_bits = K * b
    scores = np.zeros(idx1.shape[0], dtype=np.float32)
    for band in range(L):
        start = band * band_bits
        end = start + band_bits
        matches = np.all(hash_codes[idx1, start:end] == hash_codes[idx2, start:end], axis=1)
        scores += matches.astype(np.float32)
    return scores / float(L)


def _tfidf_pair_cosine(token_sequences: List[str], idx1: np.ndarray, idx2: np.ndarray) -> np.ndarray:
    vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False)
    tfidf = vectorizer.fit_transform(token_sequences)
    return np.asarray([tfidf[a].multiply(tfidf[b]).sum() for a, b in zip(idx1, idx2)], dtype=np.float32)


def build_multi_similarity_correlation(output_dir: str) -> pd.DataFrame:
    flows_df = load_prepared_flows(output_dir)
    pairs_df = load_prepared_pairs(output_dir)
    token_flows_df = load_prepared_token_flows(output_dir)
    bigru_hash = _load_hash_codes("bigru")
    mlp_hash = _load_hash_codes("mlp")
    bigru_lsh = _load_lsh_meta("bigru")
    mlp_lsh = _load_lsh_meta("mlp")

    idx1 = pairs_df["flow_index_1"].to_numpy(dtype=int)
    idx2 = pairs_df["flow_index_2"].to_numpy(dtype=int)
    sequences = token_flows_df["token_sequence"].fillna("").astype(str).tolist()
    tokens = _token_lists(token_flows_df["token_sequence"])
    counters = [Counter(row) for row in tokens]
    feature_columns = [column for column in flows_df.columns if column not in {"sample_id", "Label", "source_file", "source_row_index"}]
    flow_matrix = flows_df[feature_columns].to_numpy(dtype=np.float32)

    true_scores = {
        "Jaccard": np.asarray([_jaccard(tokens[a], tokens[b]) for a, b in zip(idx1, idx2)], dtype=np.float32),
        "Jaccard-bigram": np.asarray([_jaccard(_bigrams(tokens[a]), _bigrams(tokens[b])) for a, b in zip(idx1, idx2)], dtype=np.float32),
        "Cosine-TF": np.asarray([_counter_cosine(counters[a], counters[b]) for a, b in zip(idx1, idx2)], dtype=np.float32),
        "Cosine-TFIDF": _tfidf_pair_cosine(sequences, idx1, idx2),
        "Flow-cosine": _cosine_01_pairs(flow_matrix, idx1, idx2),
        "Levenshtein": np.asarray([SequenceMatcher(None, sequences[a], sequences[b]).ratio() for a, b in zip(idx1, idx2)], dtype=np.float32),
    }

    simhash = simhash_signatures(sequences, n_bits=64)
    base = pd.DataFrame(
        {
            "pair_index": np.arange(pairs_df.shape[0], dtype=int),
            "flow_index_1": idx1,
            "flow_index_2": idx2,
            "label_1": pairs_df["label_1"].astype(str),
            "label_2": pairs_df["label_2"].astype(str),
            "bigru_hamming_similarity": _hamming_pair_similarity(bigru_hash, idx1, idx2),
            "bigru_collision_probability": _band_collision_scores(bigru_hash, idx1, idx2, bigru_lsh["L"], bigru_lsh["K"], bigru_lsh["b"]),
            "mlp_hamming_similarity": _hamming_pair_similarity(mlp_hash, idx1, idx2),
            "mlp_collision_probability": _band_collision_scores(mlp_hash, idx1, idx2, mlp_lsh["L"], mlp_lsh["K"], mlp_lsh["b"]),
            "simhash_similarity": simhash_pair_scores(simhash, pairs_df, n_bits=64),
            "is_duplicate": pairs_df["is_duplicate"].to_numpy(dtype=int),
        }
    )

    rows = []
    for metric in METRIC_ORDER:
        metric_df = base.copy()
        metric_df.insert(0, "metric", metric)
        metric_df.insert(6, "true_sim", true_scores[metric])
        rows.append(metric_df)
    result = pd.concat(rows, ignore_index=True)
    for column in ["true_sim", "bigru_hamming_similarity", "bigru_collision_probability", "mlp_hamming_similarity", "mlp_collision_probability", "simhash_similarity"]:
        if not ((result[column] >= 0).all() and (result[column] <= 1).all()):
            raise ValueError(f"{column} values must be in [0, 1]")
    return result


def build_kendall_tau(multi_df: pd.DataFrame) -> pd.DataFrame:
    methods = [
        ("BiGRU-DeepLSH", "bigru_hamming_similarity"),
        ("MLP-DeepLSH", "mlp_hamming_similarity"),
        ("SimHash", "simhash_similarity"),
    ]
    rows = []
    for metric in METRIC_ORDER:
        metric_df = multi_df[multi_df["metric"] == metric]
        for method, column in methods:
            tau, p_value = kendalltau(metric_df["true_sim"], metric_df[column])
            rows.append(
                {
                    "metric": metric,
                    "method": method,
                    "kendall_tau": float(tau) if not np.isnan(tau) else 0.0,
                    "p_value": float(p_value) if not np.isnan(p_value) else 1.0,
                    "prediction_column": column,
                }
            )
    return pd.DataFrame.from_records(rows)


def _simhash_scores_for_query(signatures: np.ndarray, query_index: int) -> np.ndarray:
    query_value = int(signatures[query_index])
    return np.asarray([1.0 - (bin(query_value ^ int(value)).count("1") / 64.0) for value in signatures], dtype=np.float32)


def _scores_for_method(method: str, query_index: int, data: Dict[str, object]) -> np.ndarray:
    if method == "Exact-MD5":
        digests = data["md5_digests"]
        return (digests == digests[query_index]).astype(np.float32)
    if method == "SimHash":
        return _simhash_scores_for_query(data["simhash"], query_index)
    if method == "MLP-DeepLSH":
        hash_codes = data["mlp_hash"]
        return np.mean(hash_codes == hash_codes[query_index], axis=1).astype(np.float32)
    if method == "BiGRU-DeepLSH":
        hash_codes = data["bigru_hash"]
        return np.mean(hash_codes == hash_codes[query_index], axis=1).astype(np.float32)
    raise ValueError(f"Unknown retrieval method: {method}")


def build_retrieval_performance(output_dir: str, query_limit: int, top_ks: List[int]) -> pd.DataFrame:
    token_flows_df = load_prepared_token_flows(output_dir)
    pairs_df = load_prepared_pairs(output_dir)
    sequences = token_flows_df["token_sequence"].fillna("").astype(str).to_numpy()
    n_rows = len(token_flows_df)
    candidate_sets: Dict[int, set] = {}
    relevant_sets: Dict[int, set] = {}
    for row in pairs_df.itertuples(index=False):
        index_a = int(row.flow_index_1)
        index_b = int(row.flow_index_2)
        candidate_sets.setdefault(index_a, set()).add(index_b)
        candidate_sets.setdefault(index_b, set()).add(index_a)
        if int(row.is_duplicate) == 1:
            relevant_sets.setdefault(index_a, set()).add(index_b)
            relevant_sets.setdefault(index_b, set()).add(index_a)
    available_queries = np.asarray(sorted(relevant_sets), dtype=int)
    query_count = min(query_limit, available_queries.shape[0])
    query_positions = np.linspace(0, available_queries.shape[0] - 1, query_count, dtype=int)
    query_indices = available_queries[query_positions]
    data = {
        "md5_digests": np.asarray([hashlib.md5(sequence.encode("utf-8")).hexdigest() for sequence in sequences], dtype=object),
        "simhash": simhash_signatures(sequences.tolist(), n_bits=64),
        "mlp_hash": _load_hash_codes("mlp"),
        "bigru_hash": _load_hash_codes("bigru"),
    }

    rows = []
    for method in ["Exact-MD5", "SimHash", "MLP-DeepLSH", "BiGRU-DeepLSH"]:
        hits_at_k = {k: [] for k in top_ks}
        precision_at_k = {k: [] for k in top_ks}
        reciprocal_ranks = []
        durations = []
        for query_index in query_indices:
            relevant = np.zeros(n_rows, dtype=bool)
            relevant[list(relevant_sets[int(query_index)])] = True
            candidate_indices = np.asarray(sorted(candidate_sets[int(query_index)]), dtype=int)
            started = time.perf_counter()
            scores = _scores_for_method(method, int(query_index), data)
            scores[query_index] = -np.inf
            order = candidate_indices[np.argsort(-scores[candidate_indices])]
            durations.append((time.perf_counter() - started) * 1000.0)
            ranked_relevant = relevant[order]
            relevant_positions = np.where(ranked_relevant)[0]
            reciprocal_ranks.append(0.0 if relevant_positions.size == 0 else 1.0 / float(relevant_positions[0] + 1))
            for k in top_ks:
                top_relevant = ranked_relevant[:k]
                hits_at_k[k].append(float(np.any(top_relevant)))
                precision_at_k[k].append(float(np.sum(top_relevant) / k))

        row = {
            "method": method,
            "query_count": query_count,
            "relevance_source": "pairs_train_candidate_ranking",
            "mrr": float(np.mean(reciprocal_ranks)),
            "avg_query_latency_ms": float(np.mean(durations)),
        }
        for k in top_ks:
            row[f"rr_at_{k}"] = float(np.mean(hits_at_k[k]))
            row[f"precision_at_{k}"] = float(np.mean(precision_at_k[k]))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def build_lsh_hyperparam_fscore(output_dir: str, folds: int) -> pd.DataFrame:
    pairs_df = load_prepared_pairs(output_dir)
    idx1 = pairs_df["flow_index_1"].to_numpy(dtype=int)
    idx2 = pairs_df["flow_index_2"].to_numpy(dtype=int)
    y_true = pairs_df["is_duplicate"].to_numpy(dtype=int)
    fold_ids = np.arange(len(pairs_df), dtype=int) % folds
    rows = []
    for method, model_type in [("MLP-DeepLSH", "mlp"), ("BiGRU-DeepLSH", "bigru")]:
        hash_codes = _load_hash_codes(model_type)
        lsh_meta = _load_lsh_meta(model_type)
        b = int(lsh_meta["b"])
        for L, K in LSH_PARAMS:
            scores = _band_collision_scores(hash_codes, idx1, idx2, L=L, K=K, b=b)
            thresholds = np.linspace(0.0, 1.0, L + 1)
            for fold in range(folds):
                mask = fold_ids == fold
                best = None
                for threshold in thresholds:
                    metrics = classification_metrics(y_true[mask], (scores[mask] >= threshold).astype(int))
                    metrics["threshold"] = float(threshold)
                    if best is None or metrics["f1"] > best["f1"] or (metrics["f1"] == best["f1"] and metrics["recall"] > best["recall"]):
                        best = metrics
                rows.append(
                    {
                        "method": method,
                        "L": L,
                        "K": K,
                        "b": b,
                        "fold": fold,
                        "threshold": best["threshold"],
                        "accuracy": best["accuracy"],
                        "precision": best["precision"],
                        "recall": best["recall"],
                        "f1": best["f1"],
                    }
                )
    return pd.DataFrame.from_records(rows)


def matlab_script() -> str:
    return """% Draw DeepLSH paper-style figures from exported CSV files.
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

multi = readtable(fullfile(dataDir, 'multi_similarity_correlation.csv'));
kendall = readtable(fullfile(dataDir, 'kendall_tau_comparison.csv'));
retrieval = readtable(fullfile(dataDir, 'retrieval_performance.csv'));
hyper = readtable(fullfile(dataDir, 'lsh_hyperparam_fscore.csv'));

%% Figure 1: locality-sensitive preserving correlation
metrics = unique(multi.metric, 'stable');
figure('Color', 'w', 'Position', [80 80 1500 760]);
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:numel(metrics)
    nexttile;
    idx = strcmp(multi.metric, metrics{i});
    scatter(multi.true_sim(idx), multi.bigru_hamming_similarity(idx), 5, 'x', ...
        'MarkerEdgeAlpha', 0.12, 'MarkerEdgeColor', [0.10 0.42 0.55]);
    hold on;
    plot([0 1], [0 1], 'r--', 'LineWidth', 1);
    title(sprintf('%s (BiGRU-DeepLSH)', metrics{i}), 'Interpreter', 'none');
    xlabel(sprintf('%s similarity values', metrics{i}), 'Interpreter', 'none');
    ylabel('Predicted Hamming similarity');
    xlim([0 1]); ylim([0 1]);
    grid on;
end
exportgraphics(gcf, fullfile(figDir, 'matlab_multi_similarity_correlation.png'), 'Resolution', 300);

%% Figure 2: Kendall tau coefficient comparison
metrics = unique(kendall.metric, 'stable');
methods = unique(kendall.method, 'stable');
Y = nan(numel(metrics), numel(methods));
for i = 1:numel(metrics)
    for j = 1:numel(methods)
        idx = strcmp(kendall.metric, metrics{i}) & strcmp(kendall.method, methods{j});
        Y(i, j) = kendall.kendall_tau(idx);
    end
end
figure('Color', 'w', 'Position', [120 120 1100 560]);
bar(Y);
set(gca, 'XTickLabel', metrics);
xtickangle(35);
ylabel('Kendall tau coefficient');
xlabel('Similarity measures');
ylim([0 1]);
legend(methods, 'Location', 'northoutside', 'Orientation', 'horizontal', 'Interpreter', 'none');
grid on;
exportgraphics(gcf, fullfile(figDir, 'matlab_kendall_tau_comparison.png'), 'Resolution', 300);

%% Figure 3: retrieval performance comparison
retrievalMethods = string(retrieval.method);
retrievalMetrics = [retrieval.rr_at_1, retrieval.rr_at_5, retrieval.rr_at_10, retrieval.mrr];
figure('Color', 'w', 'Position', [140 140 980 540]);
bar(retrievalMetrics);
set(gca, 'XTickLabel', retrievalMethods);
xtickangle(20);
ylabel('Score');
ylim([0 1]);
legend({'RR@1', 'RR@5', 'RR@10', 'MRR'}, 'Location', 'southoutside', 'Orientation', 'horizontal');
title('Retrieval Performance Comparison');
grid on;
exportgraphics(gcf, fullfile(figDir, 'matlab_retrieval_performance.png'), 'Resolution', 300);

%% Figure 4: F1-score boxplots with different LSH hyperparameters
methods = unique(hyper.method, 'stable');
figure('Color', 'w', 'Position', [160 160 1250 520]);
tiledlayout(1, numel(methods), 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:numel(methods)
    nexttile;
    idx = strcmp(hyper.method, methods{i});
    H = hyper(idx, :);
    labels = strcat('(', string(H.L), ',', string(H.K), ')');
    orderedLabels = unique(labels, 'stable');
    group = categorical(labels, orderedLabels, 'Ordinal', true);
    if exist('boxchart', 'file') == 2
        boxchart(group, H.f1);
    else
        boxplot(H.f1, group);
    end
    title(sprintf('%s', methods{i}), 'Interpreter', 'none');
    xlabel('Hyperparameters of LSH (L,K)');
    ylabel('F1-score');
    ylim([0 1]);
    xtickangle(35);
    grid on;
end
exportgraphics(gcf, fullfile(figDir, 'matlab_lsh_hyperparam_fscore_boxplot.png'), 'Resolution', 300);

disp('MATLAB figures written to:');
disp(figDir);
"""


def run(output_dir: str, export_dir: str, query_limit: int, folds: int) -> Dict[str, str]:
    os.makedirs(export_dir, exist_ok=True)
    multi_df = build_multi_similarity_correlation(output_dir)
    kendall_df = build_kendall_tau(multi_df)
    retrieval_df = build_retrieval_performance(output_dir, query_limit=query_limit, top_ks=[1, 5, 10])
    hyper_df = build_lsh_hyperparam_fscore(output_dir, folds=folds)

    outputs = {
        "multi_similarity_correlation": os.path.join(export_dir, "multi_similarity_correlation.csv"),
        "kendall_tau_comparison": os.path.join(export_dir, "kendall_tau_comparison.csv"),
        "retrieval_performance": os.path.join(export_dir, "retrieval_performance.csv"),
        "lsh_hyperparam_fscore": os.path.join(export_dir, "lsh_hyperparam_fscore.csv"),
        "matlab_script": os.path.join(export_dir, "draw_deeplsh_figures.m"),
    }
    multi_df.to_csv(outputs["multi_similarity_correlation"], index=False)
    kendall_df.to_csv(outputs["kendall_tau_comparison"], index=False)
    retrieval_df.to_csv(outputs["retrieval_performance"], index=False)
    hyper_df.to_csv(outputs["lsh_hyperparam_fscore"], index=False)
    with open(outputs["matlab_script"], "w", encoding="utf-8") as f:
        f.write(matlab_script())
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MATLAB-ready CSV files for CIC-IDS DeepLSH paper figures.")
    parser.add_argument("--output-dir", default=None, help="Prepared CIC-IDS data directory")
    parser.add_argument("--export-dir", default=None, help="Directory to write CSV files and MATLAB script")
    parser.add_argument("--query-limit", type=int, default=120, help="Number of evenly spaced queries for retrieval metrics")
    parser.add_argument("--folds", type=int, default=10, help="Number of folds for LSH hyperparameter F1 boxplot data")
    args = parser.parse_args()

    output_dir = args.output_dir or default_processed_data_dir()
    export_dir = args.export_dir or str(cicids_artifacts_dir() / "results" / "full" / "matlab_data")
    outputs = run(output_dir=output_dir, export_dir=export_dir, query_limit=args.query_limit, folds=args.folds)
    print(
        " ".join(
            [
                "done",
                "mode=cicids-export-matlab-plot-data",
                f"query_limit={args.query_limit}",
                f"folds={args.folds}",
                *[f"{key}={value}" for key, value in outputs.items()],
            ]
        )
    )


if __name__ == "__main__":
    main()
