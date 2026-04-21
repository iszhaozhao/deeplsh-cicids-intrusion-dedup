import argparse
import gc
import json
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from deeplsh._paths import cicids_artifacts_dir
from deeplsh.cicids.pipeline import (
    build_sequence_matrix,
    default_processed_data_dir,
    default_raw_data_dir,
    load_prepared_token_flows,
    load_vocab,
    prepared_paths,
    prepare_cicids_dataset,
)
from deeplsh.cicids.train_bigru import _build_bigru_encoder
from deeplsh.core.deep_hashing_models import intermediate_model_trained, siamese_model, train_siamese_model
from deeplsh.core.lsh_search import convert_to_hamming, create_hash_tables, lsh_hyperparams


def _normalize_max_samples(max_samples):
    if max_samples is None:
        return None
    if max_samples <= 0:
        return None
    return max_samples


PAPER_PAIRS_FILENAME = "pairs_similarity_jaccard.csv"


@dataclass(frozen=True)
class HashConfig:
    m: int
    b: int

    @property
    def n_bits(self) -> int:
        return self.m * self.b

    @property
    def label(self) -> str:
        return f"M{self.m}_b{self.b}"


def _tokens_for_jaccard(token_sequence: object) -> frozenset:
    return frozenset(
        token
        for token in str(token_sequence).split()
        if not token.startswith("label=") and not token.startswith("source=")
    )


def _jaccard(tokens_a: frozenset, tokens_b: frozenset) -> float:
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return float(len(tokens_a & tokens_b) / len(union))


def _similarity_bin(score: float) -> int:
    return min(9, max(0, int(math.floor(float(score) * 10.0))))


def build_jaccard_similarity_pairs(
    token_flows_df: pd.DataFrame,
    max_pairs: int,
    seed: int,
    min_nonempty_bins: int = 8,
) -> pd.DataFrame:
    if max_pairs <= 0:
        raise ValueError("max_pairs must be > 0")
    if "token_sequence" not in token_flows_df.columns:
        raise ValueError("flows_tokens.csv must contain token_sequence")

    rng = np.random.default_rng(seed)
    token_sets = [_tokens_for_jaccard(sequence) for sequence in token_flows_df["token_sequence"].fillna("")]
    token_key_to_indices: Dict[str, List[int]] = {}
    label_to_indices: Dict[str, List[int]] = {}
    token_to_indices: Dict[str, List[int]] = {}
    for index, tokens in enumerate(token_sets):
        token_key_to_indices.setdefault(" ".join(sorted(tokens)), []).append(index)
        label = str(token_flows_df.iloc[index]["Label"])
        label_to_indices.setdefault(label, []).append(index)
        for token in tokens:
            token_to_indices.setdefault(token, []).append(index)

    candidates: Dict[Tuple[int, int], float] = {}
    n_rows = len(token_flows_df)

    def add_candidate(a: int, b: int) -> None:
        pair = tuple(sorted((int(a), int(b))))
        if pair[0] == pair[1] or pair in candidates:
            return
        candidates[pair] = _jaccard(token_sets[pair[0]], token_sets[pair[1]])

    def sample_group_pairs(groups: Iterable[List[int]], attempts_per_group: int) -> None:
        for group in groups:
            if len(group) < 2:
                continue
            indices = np.asarray(group, dtype=int)
            attempts = min(attempts_per_group, max(1, len(indices) * 4))
            for _ in range(attempts):
                a, b = rng.choice(indices, size=2, replace=False).tolist()
                add_candidate(int(a), int(b))

    sample_group_pairs(token_key_to_indices.values(), attempts_per_group=max(1000, max_pairs // 20))
    sample_group_pairs(label_to_indices.values(), attempts_per_group=max(5000, max_pairs // max(1, len(label_to_indices))))
    sample_group_pairs(token_to_indices.values(), attempts_per_group=max(1000, max_pairs // max(1, len(token_to_indices))))

    random_attempts = max(max_pairs * 8, 500000)
    for _ in range(random_attempts):
        a, b = rng.choice(n_rows, size=2, replace=False).tolist()
        add_candidate(int(a), int(b))

    candidate_df = pd.DataFrame.from_records(
        [
            {
                "flow_index_1": int(index_a),
                "flow_index_2": int(index_b),
                "label_1": str(token_flows_df.iloc[index_a]["Label"]),
                "label_2": str(token_flows_df.iloc[index_b]["Label"]),
                "true_sim": float(score),
                "similarity_bin": int(_similarity_bin(score)),
            }
            for (index_a, index_b), score in candidates.items()
        ]
    )
    if candidate_df.empty:
        raise ValueError("No Jaccard similarity pairs could be generated.")

    target_per_bin = int(math.ceil(max_pairs / 10.0))
    selected_indices = []
    for bin_id in range(10):
        group = candidate_df[candidate_df["similarity_bin"] == bin_id]
        if group.empty:
            continue
        selected = group.sample(n=min(target_per_bin, len(group)), random_state=seed + bin_id)
        selected_indices.extend(selected.index.tolist())

    if len(selected_indices) < max_pairs:
        remaining = candidate_df.drop(index=selected_indices, errors="ignore")
        if not remaining.empty:
            extra = remaining.sample(n=min(max_pairs - len(selected_indices), len(remaining)), random_state=seed)
            selected_indices.extend(extra.index.tolist())

    pairs_df = candidate_df.loc[selected_indices].sample(frac=1.0, random_state=seed).head(max_pairs).reset_index(drop=True)

    nonempty_bins = int(pairs_df["similarity_bin"].nunique())
    if nonempty_bins < min_nonempty_bins:
        raise ValueError(f"Expected at least {min_nonempty_bins} non-empty Jaccard bins, found {nonempty_bins}.")
    if not ((pairs_df["true_sim"] >= 0).all() and (pairs_df["true_sim"] <= 1).all()):
        raise ValueError("true_sim values must be in [0, 1]")
    return pairs_df


def group_collision_similarity(hash_codes: np.ndarray, index_a: np.ndarray, index_b: np.ndarray, m: int, b: int) -> np.ndarray:
    if hash_codes.shape[1] != m * b:
        raise ValueError(f"Hash code width {hash_codes.shape[1]} does not match M*b={m * b}")
    grouped = hash_codes.reshape(hash_codes.shape[0], m, b)
    collisions = np.all(grouped[index_a] == grouped[index_b], axis=2)
    return collisions.mean(axis=1).astype(np.float32)


def build_paper_correlation_dataframe(pairs_df: pd.DataFrame, hash_codes: np.ndarray, m: int, b: int) -> pd.DataFrame:
    indices_1 = pairs_df["flow_index_1"].to_numpy(dtype=int)
    indices_2 = pairs_df["flow_index_2"].to_numpy(dtype=int)
    pred_sims = group_collision_similarity(hash_codes, indices_1, indices_2, m=m, b=b)
    true_sims = pairs_df["true_sim"].to_numpy(dtype=np.float32)
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
            "similarity_bin": pairs_df["similarity_bin"].to_numpy(dtype=int),
        }
    )


def plot_paper_correlation(correlation_df: pd.DataFrame, output_png: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(correlation_df["true_sim"], correlation_df["pred_sim"], s=1, alpha=0.12, color="#1f77b4", marker="x")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1)
    plt.title("Jaccard (BiGRU-DeepLSH)")
    plt.xlabel("Jaccard similarity values")
    plt.ylabel("Collision probability")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


def parse_hash_configs(values: List[str]) -> List[HashConfig]:
    configs = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"Hash config must use M:b format, got {value}")
        m_text, b_text = value.split(":", 1)
        config = HashConfig(m=int(m_text), b=int(b_text))
        if config.n_bits != 1024:
            raise ValueError(f"Hash config {value} must keep M*b=1024")
        configs.append(config)
    return configs


def _ensure_prepared(args) -> None:
    paths = prepared_paths(args.output_dir)
    required = [paths["flows_tokens"], paths["vocab"]]
    if args.force_prepare or not all(os.path.exists(path) for path in required):
        prepare_cicids_dataset(
            data_dir=args.data_repo,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            max_pairs=20000,
            seed=args.seed,
        )


def _load_or_build_pairs(args, token_flows_df: pd.DataFrame) -> pd.DataFrame:
    pairs_path = os.path.join(args.output_dir, PAPER_PAIRS_FILENAME)
    if args.force_pairs or not os.path.exists(pairs_path):
        pairs_df = build_jaccard_similarity_pairs(
            token_flows_df=token_flows_df,
            max_pairs=args.max_pairs,
            seed=args.seed,
            min_nonempty_bins=args.min_nonempty_bins,
        )
        pairs_df.to_csv(pairs_path, index=False)
    else:
        pairs_df = pd.read_csv(pairs_path)
    return pairs_df


def _correlation_scores(true_values: np.ndarray, pred_values: np.ndarray) -> Dict[str, float]:
    true_series = pd.Series(true_values)
    pred_series = pd.Series(pred_values)
    pearson = true_series.corr(pred_series, method="pearson")
    spearman = true_series.corr(pred_series, method="spearman")
    kendall = true_series.corr(pred_series, method="kendall")
    return {
        "pearson": 0.0 if pd.isna(pearson) else float(pearson),
        "spearman": 0.0 if pd.isna(spearman) else float(spearman),
        "kendall": 0.0 if pd.isna(kendall) else float(kendall),
    }


def _calibration_scores(true_values: np.ndarray, pred_values: np.ndarray, high_sim_threshold: float = 0.8) -> Dict[str, float]:
    true_values = np.asarray(true_values, dtype=np.float32).reshape(-1)
    pred_values = np.asarray(pred_values, dtype=np.float32).reshape(-1)
    if true_values.shape != pred_values.shape:
        raise ValueError(f"Shape mismatch: true_values={true_values.shape}, pred_values={pred_values.shape}")

    calibration_mae = float(np.mean(np.abs(pred_values - true_values))) if true_values.size else 0.0
    high_mask = true_values >= high_sim_threshold
    high_sim_count = int(np.sum(high_mask))
    if high_sim_count:
        high_sim_mean_true = float(np.mean(true_values[high_mask]))
        high_sim_mean_pred = float(np.mean(pred_values[high_mask]))
        high_sim_gap = abs(high_sim_mean_pred - high_sim_mean_true)
    else:
        high_sim_mean_true = 0.0
        high_sim_mean_pred = 0.0
        high_sim_gap = 1.0

    return {
        "calibration_mae": calibration_mae,
        "high_sim_threshold": float(high_sim_threshold),
        "high_sim_count": high_sim_count,
        "high_sim_mean_true": high_sim_mean_true,
        "high_sim_mean_pred": high_sim_mean_pred,
        "high_sim_gap": float(high_sim_gap),
    }


def _selection_score(metrics: Dict[str, float]) -> float:
    return float(metrics["pearson"] + metrics["spearman"] - metrics["calibration_mae"] - metrics["high_sim_gap"])


def _save_best_artifacts(
    shared_model: tf.keras.Model,
    sequence_matrix: np.ndarray,
    token_flows_df: pd.DataFrame,
    hash_codes: np.ndarray,
    config: HashConfig,
    args,
    metrics: Dict[str, float],
    candidate_summaries: List[Dict[str, float]],
    pairs_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
) -> None:
    base = cicids_artifacts_dir()
    models_dir = base / "models"
    hash_tables_dir = base / "hash_tables"
    results_dir = base / "results" / Path(args.output_dir).name / "paper_lsh"
    models_dir.mkdir(parents=True, exist_ok=True)
    hash_tables_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "model-deep-lsh-cicids-bigru-jaccard-paper.model"
    embeddings_path = models_dir / "cicids_bigru_jaccard_paper_embeddings.npy"
    embeddings_hamming_path = models_dir / "cicids_bigru_jaccard_paper_embeddings_hamming.npy"
    sequences_path = models_dir / "cicids_bigru_jaccard_paper_sequences.npy"
    corpus_path = models_dir / "cicids_bigru_jaccard_paper_tokens.csv"
    metadata_path = models_dir / "cicids_bigru_jaccard_paper_metadata.json"
    hash_tables_path = hash_tables_dir / "hash_tables_deeplsh_cicids_bigru_jaccard_paper.pkl"
    output_csv = results_dir / "cicids_lsh_correlation_jaccard_bigru_paper.csv"
    output_png = results_dir / "cicids_lsh_correlation_jaccard_bigru_paper.png"
    summary_path = results_dir / "cicids_lsh_correlation_jaccard_bigru_paper_summary.json"

    intermediate_model = intermediate_model_trained(shared_model, output_layer=-1, CNN=False)
    embeddings = intermediate_model.predict(sequence_matrix, verbose=0)
    params = lsh_hyperparams(config.m)
    lsh_param_index = min(args.lsh_param_index, len(params) - 1)
    L, K = params[lsh_param_index]
    hash_tables = create_hash_tables(L, K, config.b, hash_codes)

    intermediate_model.save(str(model_path))
    np.save(embeddings_path, embeddings)
    np.save(embeddings_hamming_path, hash_codes)
    np.save(sequences_path, sequence_matrix)
    token_flows_df.to_csv(corpus_path, index=False)
    with open(hash_tables_path, "wb") as f:
        pickle.dump(hash_tables, f)
    correlation_df.to_csv(output_csv, index=False)
    plot_paper_correlation(correlation_df, str(output_png))

    bin_counts = {str(k): int(v) for k, v in pairs_df["similarity_bin"].value_counts().sort_index().to_dict().items()}
    heldout_metrics = _correlation_scores(
        correlation_df["true_sim"].to_numpy(dtype=np.float32),
        correlation_df["pred_sim"].to_numpy(dtype=np.float32),
    )
    heldout_metrics.update(_calibration_scores(
        correlation_df["true_sim"].to_numpy(dtype=np.float32),
        correlation_df["pred_sim"].to_numpy(dtype=np.float32),
    ))
    summary = {
        "objective": "paper_mse_calibrated_selection",
        "similarity": "jaccard",
        "model_type": "bigru-deeplsh-paper",
        "M": int(config.m),
        "b": int(config.b),
        "hash_bits": int(config.n_bits),
        "lsh": {"L": int(L), "K": int(K), "b": int(config.b), "M": int(config.m)},
        "n_flows": int(token_flows_df.shape[0]),
        "n_pairs": int(pairs_df.shape[0]),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "pair_file": os.path.join(args.output_dir, PAPER_PAIRS_FILENAME),
        "similarity_bin_counts": bin_counts,
        "best_validation_pearson": metrics["pearson"],
        "best_validation_spearman": metrics["spearman"],
        "best_validation_kendall": metrics["kendall"],
        "best_validation_calibration_mae": metrics["calibration_mae"],
        "best_validation_high_sim_mean_true": metrics["high_sim_mean_true"],
        "best_validation_high_sim_mean_pred": metrics["high_sim_mean_pred"],
        "best_validation_high_sim_gap": metrics["high_sim_gap"],
        "best_validation_selection_score": metrics["selection_score"],
        "heldout_plot_pearson": heldout_metrics["pearson"],
        "heldout_plot_spearman": heldout_metrics["spearman"],
        "heldout_plot_kendall": heldout_metrics["kendall"],
        "heldout_plot_calibration_mae": heldout_metrics["calibration_mae"],
        "heldout_plot_high_sim_mean_true": heldout_metrics["high_sim_mean_true"],
        "heldout_plot_high_sim_mean_pred": heldout_metrics["high_sim_mean_pred"],
        "heldout_plot_high_sim_gap": heldout_metrics["high_sim_gap"],
        "correlation_split": "heldout_plot",
        "correlation_pairs": int(correlation_df.shape[0]),
        "target_reached": bool(metrics["pearson"] >= args.target_correlation or metrics["spearman"] >= args.target_correlation),
        "target_correlation": float(args.target_correlation),
        "candidate_summaries": candidate_summaries,
        "model_path": str(model_path),
        "embeddings_path": str(embeddings_path),
        "embeddings_hamming_path": str(embeddings_hamming_path),
        "hash_tables_path": str(hash_tables_path),
        "output_csv": str(output_csv),
        "output_png": str(output_png),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    try:
        from deeplsh.cicids.plot_paper_lsh_sensitivity import run as run_lsh_sensitivity

        sensitivity_summary = run_lsh_sensitivity(output_dir=args.output_dir, results_dir=str(results_dir))
        summary["lsh_sensitivity_summary_json"] = str(results_dir / "cicids_lsh_sensitivity_jaccard_bigru_paper_summary.json")
        summary["lsh_sensitivity_csv"] = sensitivity_summary["sensitivity_csv"]
        summary["lsh_sensitivity_png"] = sensitivity_summary["sensitivity_png"]
        summary["lsh_calibration_curves_png"] = sensitivity_summary["calibration_curves_png"]
        summary["lsh_sensitivity_recommended_config"] = sensitivity_summary["recommended_config"]
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception as exc:
        print(f"warning=lsh_sensitivity_failed error={exc}", flush=True)


def run(args) -> Dict[str, object]:
    if args.similarity != "jaccard":
        raise ValueError(f"Unsupported similarity: {args.similarity}")
    _ensure_prepared(args)

    token_flows_df = load_prepared_token_flows(args.output_dir)
    vocab = load_vocab(args.output_dir)
    pairs_df = _load_or_build_pairs(args, token_flows_df)
    sequence_matrix = build_sequence_matrix(token_flows_df, vocab, max_length=vocab["max_sequence_length"])

    pair_indices = np.arange(pairs_df.shape[0], dtype=int)
    plot_stratify = pairs_df["similarity_bin"] if pairs_df["similarity_bin"].nunique() > 1 else None
    development_idx, plot_idx = train_test_split(
        pair_indices,
        test_size=args.plot_fraction,
        random_state=args.seed,
        stratify=plot_stratify,
    )
    development_bins = pairs_df.iloc[development_idx]["similarity_bin"]
    validation_stratify = development_bins if development_bins.nunique() > 1 else None
    train_idx, validation_idx = train_test_split(
        development_idx,
        test_size=args.validation_fraction,
        random_state=args.seed,
        stratify=validation_stratify,
    )

    def build_pair_inputs(indices: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        subset = pairs_df.iloc[indices]
        return {
            "stack_1": sequence_matrix[subset["flow_index_1"].to_numpy(dtype=int)],
            "stack_2": sequence_matrix[subset["flow_index_2"].to_numpy(dtype=int)],
        }, subset["true_sim"].to_numpy(dtype=np.float32)

    X_train, Y_train = build_pair_inputs(train_idx)
    X_validation, Y_validation = build_pair_inputs(validation_idx)
    validation_pairs = pairs_df.iloc[validation_idx]
    validation_i = validation_pairs["flow_index_1"].to_numpy(dtype=int)
    validation_j = validation_pairs["flow_index_2"].to_numpy(dtype=int)

    configs = parse_hash_configs(args.hash_configs)
    best = None
    candidate_summaries: List[Dict[str, float]] = []

    for config in configs:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        shared_model = _build_bigru_encoder(
            vocab_size=max(vocab["token_to_id"].values(), default=1) + 1,
            max_length=sequence_matrix.shape[1],
            embed_dim=args.embed_dim,
            gru_units=args.gru_units,
            dense_dim=args.dense_dim,
            size_hash_vector=config.n_bits,
            attention_pooling=args.attention_pooling,
            layer_norm=args.layer_norm,
        )
        model = siamese_model(
            shared_model,
            input_shape=(sequence_matrix.shape[1],),
            b=config.b,
            m=config.m,
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
        embeddings = intermediate_model.predict(sequence_matrix, verbose=0)
        hash_codes = convert_to_hamming(embeddings)
        validation_pred = group_collision_similarity(hash_codes, validation_i, validation_j, m=config.m, b=config.b)
        metrics = _correlation_scores(Y_validation, validation_pred)
        metrics.update(_calibration_scores(Y_validation, validation_pred))
        metrics["selection_score"] = _selection_score(metrics)
        candidate_summary = {
            "M": int(config.m),
            "b": int(config.b),
            "pearson": metrics["pearson"],
            "spearman": metrics["spearman"],
            "kendall": metrics["kendall"],
            "calibration_mae": metrics["calibration_mae"],
            "high_sim_mean_true": metrics["high_sim_mean_true"],
            "high_sim_mean_pred": metrics["high_sim_mean_pred"],
            "high_sim_gap": metrics["high_sim_gap"],
            "selection_score": metrics["selection_score"],
        }
        candidate_summaries.append(candidate_summary)

        score = metrics["selection_score"]
        is_better = best is None or score > best["score"] or (
            score == best["score"] and metrics["calibration_mae"] < best["metrics"]["calibration_mae"]
        )
        if is_better:
            heldout_correlation_df = build_paper_correlation_dataframe(pairs_df.iloc[plot_idx], hash_codes, m=config.m, b=config.b)
            heldout_metrics = _correlation_scores(
                heldout_correlation_df["true_sim"].to_numpy(dtype=np.float32),
                heldout_correlation_df["pred_sim"].to_numpy(dtype=np.float32),
            )
            heldout_metrics.update(
                _calibration_scores(
                    heldout_correlation_df["true_sim"].to_numpy(dtype=np.float32),
                    heldout_correlation_df["pred_sim"].to_numpy(dtype=np.float32),
                )
            )
            _save_best_artifacts(
                shared_model=shared_model,
                sequence_matrix=sequence_matrix,
                token_flows_df=token_flows_df,
                hash_codes=hash_codes,
                config=config,
                args=args,
                metrics=metrics,
                candidate_summaries=candidate_summaries,
                pairs_df=pairs_df,
                correlation_df=heldout_correlation_df,
            )
            best = {
                "config": config,
                "score": score,
                "metrics": metrics,
                "heldout_metrics": heldout_metrics,
            }

        del model
        del shared_model
        tf.keras.backend.clear_session()
        gc.collect()

    if best is None:
        raise RuntimeError("No paper-faithful DeepLSH candidate was trained.")

    final_metadata_paths = [
        cicids_artifacts_dir() / "models" / "cicids_bigru_jaccard_paper_metadata.json",
        cicids_artifacts_dir()
        / "results"
        / Path(args.output_dir).name
        / "paper_lsh"
        / "cicids_lsh_correlation_jaccard_bigru_paper_summary.json",
    ]
    for metadata_path in final_metadata_paths:
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            metadata["candidate_summaries"] = candidate_summaries
            metadata["target_reached"] = bool(
                best["metrics"]["pearson"] >= args.target_correlation
                or best["metrics"]["spearman"] >= args.target_correlation
            )
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
    return {
        "pairs_rows": int(pairs_df.shape[0]),
        "nonempty_bins": int(pairs_df["similarity_bin"].nunique()),
        "best_m": int(best["config"].m),
        "best_b": int(best["config"].b),
        "best_validation_pearson": best["metrics"]["pearson"],
        "best_validation_spearman": best["metrics"]["spearman"],
        "best_validation_calibration_mae": best["metrics"]["calibration_mae"],
        "best_validation_high_sim_gap": best["metrics"]["high_sim_gap"],
        "heldout_plot_pairs": int(len(plot_idx)),
        "best_heldout_pearson": best["heldout_metrics"]["pearson"],
        "best_heldout_spearman": best["heldout_metrics"]["spearman"],
        "best_heldout_calibration_mae": best["heldout_metrics"]["calibration_mae"],
        "best_heldout_high_sim_gap": best["heldout_metrics"]["high_sim_gap"],
        "best_selection_score": best["score"],
        "target_reached": bool(
            best["metrics"]["pearson"] >= args.target_correlation
            or best["metrics"]["spearman"] >= args.target_correlation
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train paper-faithful BiGRU-DeepLSH for Jaccard collision correlation.")
    parser.add_argument("--data-repo", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--similarity", choices=["jaccard"], default="jaccard")
    parser.add_argument("--max-samples", type=int, default=12000, help="Maximum raw flows to read when preparing; use 0 for all rows")
    parser.add_argument("--max-pairs", type=int, default=120000)
    parser.add_argument("--min-nonempty-bins", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lsh-param-index", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--gru-units", type=int, default=64)
    parser.add_argument("--dense-dim", type=int, default=128)
    parser.add_argument("--hash-configs", nargs="+", default=["128:8", "256:4", "512:2"])
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--plot-fraction", type=float, default=0.2)
    parser.add_argument("--target-correlation", type=float, default=0.75)
    parser.add_argument("--attention-pooling", dest="attention_pooling", action="store_true", default=True)
    parser.add_argument("--no-attention-pooling", dest="attention_pooling", action="store_false")
    parser.add_argument("--layer-norm", dest="layer_norm", action="store_true", default=True)
    parser.add_argument("--no-layer-norm", dest="layer_norm", action="store_false")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--force-pairs", action="store_true")
    args = parser.parse_args()

    if args.data_repo is None:
        args.data_repo = default_raw_data_dir()
    if args.output_dir is None:
        args.output_dir = default_processed_data_dir()
    args.max_samples = _normalize_max_samples(args.max_samples)

    summary = run(args)
    print(
        " ".join(
            [
                "done",
                "mode=cicids-train-paper-lsh",
                f"similarity={args.similarity}",
                f"pairs_rows={summary['pairs_rows']}",
                f"nonempty_bins={summary['nonempty_bins']}",
                f"best_M={summary['best_m']}",
                f"best_b={summary['best_b']}",
                f"best_validation_pearson={summary['best_validation_pearson']:.6f}",
                f"best_validation_spearman={summary['best_validation_spearman']:.6f}",
                f"best_validation_calibration_mae={summary['best_validation_calibration_mae']:.6f}",
                f"best_validation_high_sim_gap={summary['best_validation_high_sim_gap']:.6f}",
                f"heldout_plot_pairs={summary['heldout_plot_pairs']}",
                f"best_heldout_pearson={summary['best_heldout_pearson']:.6f}",
                f"best_heldout_spearman={summary['best_heldout_spearman']:.6f}",
                f"best_heldout_calibration_mae={summary['best_heldout_calibration_mae']:.6f}",
                f"best_heldout_high_sim_gap={summary['best_heldout_high_sim_gap']:.6f}",
                f"best_selection_score={summary['best_selection_score']:.6f}",
                f"target_reached={summary['target_reached']}",
            ]
        )
    )


if __name__ == "__main__":
    main()
