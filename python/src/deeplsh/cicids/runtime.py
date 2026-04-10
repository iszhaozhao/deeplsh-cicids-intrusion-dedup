import hashlib
import json
import os
import pickle
import time
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from deeplsh._paths import cicids_artifacts_dir

def cosine_01(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denom == 0.0:
        return 0.0
    cosine = float(np.dot(vector_a, vector_b) / denom)
    cosine = max(-1.0, min(1.0, cosine))
    return (cosine + 1.0) / 2.0


def _artifact_name_map(model_type: str) -> Dict[str, str]:
    if model_type == "mlp":
        return {
            "model_dir_name": "model-deep-lsh-cicids.model",
            "hash_tables_name": "hash_tables_deeplsh_cicids.pkl",
            "embeddings_name": "cicids_embeddings.npy",
            "embeddings_hamming_name": "cicids_embeddings_hamming.npy",
            "corpus_name": "cicids_flows.csv",
            "train_meta_name": "cicids_train_metadata.json",
            "input_matrix_name": None,
        }
    if model_type == "bigru":
        return {
            "model_dir_name": "model-deep-lsh-cicids-bigru.model",
            "hash_tables_name": "hash_tables_deeplsh_cicids_bigru.pkl",
            "embeddings_name": "cicids_bigru_embeddings.npy",
            "embeddings_hamming_name": "cicids_bigru_embeddings_hamming.npy",
            "corpus_name": "cicids_tokens.csv",
            "train_meta_name": "cicids_bigru_train_metadata.json",
            "input_matrix_name": "cicids_bigru_sequences.npy",
        }
    raise ValueError(f"Unknown model_type: {model_type}")


def artifact_paths(model_type: str) -> Dict[str, str]:
    names = _artifact_name_map(model_type)
    base = cicids_artifacts_dir()
    models_dir = os.path.join(str(base), "models")
    hash_tables_dir = os.path.join(str(base), "hash_tables")
    paths = {
        "model": os.path.join(models_dir, names["model_dir_name"]),
        "hash_tables": os.path.join(hash_tables_dir, names["hash_tables_name"]),
        "embeddings": os.path.join(models_dir, names["embeddings_name"]),
        "embeddings_hamming": os.path.join(models_dir, names["embeddings_hamming_name"]),
        "corpus": os.path.join(models_dir, names["corpus_name"]),
        "train_meta": os.path.join(models_dir, names["train_meta_name"]),
    }
    if names["input_matrix_name"] is not None:
        paths["input_matrix"] = os.path.join(models_dir, names["input_matrix_name"])
    return paths


def load_runtime_bundle(model_type: str) -> Dict[str, object]:
    paths = artifact_paths(model_type)
    required = [paths["hash_tables"], paths["embeddings"], paths["embeddings_hamming"], paths["corpus"], paths["train_meta"]]
    if "input_matrix" in paths:
        required.append(paths["input_matrix"])
    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required artifact not found: {path}")

    corpus_df = pd.read_csv(paths["corpus"])
    embeddings = np.load(paths["embeddings"])
    embeddings_hamming = np.load(paths["embeddings_hamming"])
    with open(paths["train_meta"], "r", encoding="utf-8") as f:
        train_meta = json.load(f)
    with open(paths["hash_tables"], "rb") as f:
        hash_tables = pickle.load(f)

    bundle = {
        "paths": paths,
        "corpus_df": corpus_df,
        "embeddings": embeddings,
        "embeddings_hamming": embeddings_hamming,
        "train_meta": train_meta,
        "hash_tables": hash_tables,
    }
    if "input_matrix" in paths:
        bundle["input_matrix"] = np.load(paths["input_matrix"])
    return bundle


def candidate_hit_counts(query_hamming: np.ndarray, hash_tables: Dict[str, dict], L: int, K: int, b: int) -> Dict[int, int]:
    n_bits = K * b
    hit_counts: Dict[int, int] = {}
    for bucket_index in range(L):
        key = query_hamming[bucket_index * n_bits : (bucket_index + 1) * n_bits].tobytes()
        entry = hash_tables.get(f"entry_{bucket_index}", {})
        if key not in entry:
            continue
        for candidate_index in entry[key]:
            candidate_index = int(candidate_index)
            hit_counts[candidate_index] = hit_counts.get(candidate_index, 0) + 1
    return hit_counts


def query_top_k(bundle: Dict[str, object], query_index: int, top_k: int = 10, label_scope: str = "same") -> pd.DataFrame:
    corpus_df = bundle["corpus_df"]
    embeddings = bundle["embeddings"]
    embeddings_hamming = bundle["embeddings_hamming"]
    train_meta = bundle["train_meta"]
    hash_tables = bundle["hash_tables"]
    lsh = train_meta["lsh"]
    L = int(lsh["L"])
    K = int(lsh["K"])
    b = int(lsh["b"])

    query_hamming = embeddings_hamming[query_index]
    hit_counts = candidate_hit_counts(query_hamming, hash_tables, L=L, K=K, b=b)

    query_label = str(corpus_df.iloc[query_index]["Label"])
    candidate_indices = [idx for idx in hit_counts.keys() if idx != query_index]
    if label_scope == "same":
        candidate_indices = [idx for idx in candidate_indices if str(corpus_df.iloc[idx]["Label"]) == query_label]

    if not candidate_indices:
        candidate_indices = [idx for idx in range(corpus_df.shape[0]) if idx != query_index]
        if label_scope == "same":
            candidate_indices = [idx for idx in candidate_indices if str(corpus_df.iloc[idx]["Label"]) == query_label]

    rows = []
    for candidate_index in candidate_indices:
        rows.append(
            {
                "query_sample_id": corpus_df.iloc[query_index]["sample_id"],
                "candidate_sample_id": corpus_df.iloc[candidate_index]["sample_id"],
                "query_label": query_label,
                "candidate_label": str(corpus_df.iloc[candidate_index]["Label"]),
                "hash_bucket_hits": int(hit_counts.get(candidate_index, 0)),
                "embedding_similarity": cosine_01(embeddings[query_index], embeddings[candidate_index]),
                "is_same_label": int(str(corpus_df.iloc[candidate_index]["Label"]) == query_label),
                "source_file": str(corpus_df.iloc[candidate_index]["source_file"]),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "query_sample_id",
                "candidate_sample_id",
                "query_label",
                "candidate_label",
                "hash_bucket_hits",
                "embedding_similarity",
                "is_same_label",
                "source_file",
            ]
        )

    return (
        pd.DataFrame.from_records(rows)
        .sort_values(by=["hash_bucket_hits", "embedding_similarity", "candidate_sample_id"], ascending=[False, False, True])
        .head(top_k)
    )


def average_query_latency_ms(bundle: Dict[str, object], top_k: int = 10, label_scope: str = "same", limit: int = 50) -> float:
    corpus_size = bundle["corpus_df"].shape[0]
    sample_count = min(limit, corpus_size)
    if sample_count == 0:
        return 0.0
    indices = np.linspace(0, corpus_size - 1, sample_count, dtype=int)
    durations = []
    for index in indices:
        started = time.perf_counter()
        query_top_k(bundle, query_index=int(index), top_k=top_k, label_scope=label_scope)
        durations.append((time.perf_counter() - started) * 1000.0)
    return float(np.mean(durations))


def pair_scores_from_embeddings(embeddings: np.ndarray, pairs_df: pd.DataFrame) -> np.ndarray:
    indices_1 = pairs_df["flow_index_1"].to_numpy(dtype=int)
    indices_2 = pairs_df["flow_index_2"].to_numpy(dtype=int)
    return np.asarray([cosine_01(embeddings[a], embeddings[b]) for a, b in zip(indices_1, indices_2)], dtype=np.float32)


def binary_hash_collision_rate(binary_rows: np.ndarray) -> float:
    if binary_rows.shape[0] == 0:
        return 0.0
    unique = {row.tobytes() for row in binary_rows}
    return 1.0 - (len(unique) / float(binary_rows.shape[0]))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(len(y_true), 1)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def best_threshold_metrics(scores: np.ndarray, y_true: np.ndarray, thresholds: Iterable[float] = None) -> Dict[str, float]:
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.95, 18)
    best = None
    for threshold in thresholds:
        metrics = classification_metrics(y_true, (scores >= threshold).astype(int))
        metrics["threshold"] = float(threshold)
        if best is None or metrics["f1"] > best["f1"] or (metrics["f1"] == best["f1"] and metrics["recall"] > best["recall"]):
            best = metrics
    return best or {"threshold": 0.5, **classification_metrics(y_true, np.zeros_like(y_true))}


def exact_match_scores(token_sequences: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    digests = np.asarray([hashlib.md5(sequence.encode("utf-8")).hexdigest() for sequence in token_sequences])
    return digests, digests


def _simhash_int(token_sequence: str, n_bits: int = 64) -> int:
    weights = Counter(str(token_sequence).split())
    accumulator = np.zeros(n_bits, dtype=np.int32)
    for token, weight in weights.items():
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()[:16]
        token_hash = int(digest, 16)
        for bit in range(n_bits):
            accumulator[bit] += weight if (token_hash >> bit) & 1 else -weight
    value = 0
    for bit, score in enumerate(accumulator):
        if score >= 0:
            value |= (1 << bit)
    return value


def simhash_signatures(token_sequences: List[str], n_bits: int = 64) -> np.ndarray:
    return np.asarray([_simhash_int(sequence, n_bits=n_bits) for sequence in token_sequences], dtype=np.uint64)


def simhash_pair_scores(signatures: np.ndarray, pairs_df: pd.DataFrame, n_bits: int = 64) -> np.ndarray:
    def sim(a: int, b: int) -> float:
        xor = int(a ^ b)
        distance = bin(xor).count("1")
        return 1.0 - (distance / float(n_bits))

    return np.asarray(
        [sim(signatures[a], signatures[b]) for a, b in zip(pairs_df["flow_index_1"].to_numpy(dtype=int), pairs_df["flow_index_2"].to_numpy(dtype=int))],
        dtype=np.float32,
    )


def simhash_query_latency_ms(signatures: np.ndarray, top_k: int = 10, limit: int = 50, n_bits: int = 64) -> float:
    sample_count = min(limit, len(signatures))
    if sample_count == 0:
        return 0.0
    indices = np.linspace(0, len(signatures) - 1, sample_count, dtype=int)
    durations = []
    for index in indices:
        started = time.perf_counter()
        sims = []
        for other_index, signature in enumerate(signatures):
            if other_index == index:
                continue
            distance = bin(int(signatures[index] ^ signature)).count("1")
            sims.append((1.0 - distance / float(n_bits), other_index))
        sims.sort(reverse=True)
        sims[:top_k]
        durations.append((time.perf_counter() - started) * 1000.0)
    return float(np.mean(durations))
