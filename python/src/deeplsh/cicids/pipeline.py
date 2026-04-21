import json
import math
import os
import pickle
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from deeplsh._paths import cicids_processed_dir, cicids_raw_dir

FLOWS_FILENAME = "flows.csv"
FLOWS_TOKENS_FILENAME = "flows_tokens.csv"
PAIRS_FILENAME = "pairs_train.csv"
METADATA_FILENAME = "metadata.json"
PREPROCESSOR_FILENAME = "preprocessor.pkl"
VOCAB_FILENAME = "vocab.json"

META_COLUMNS = ["sample_id", "Label", "source_file", "source_row_index"]


def default_raw_data_dir() -> str:
    return str(cicids_raw_dir())


def default_processed_data_dir() -> str:
    return str(cicids_processed_dir("full"))


def prepared_paths(output_dir: str) -> Dict[str, str]:
    return {
        "flows": os.path.join(output_dir, FLOWS_FILENAME),
        "flows_tokens": os.path.join(output_dir, FLOWS_TOKENS_FILENAME),
        "pairs": os.path.join(output_dir, PAIRS_FILENAME),
        "metadata": os.path.join(output_dir, METADATA_FILENAME),
        "preprocessor": os.path.join(output_dir, PREPROCESSOR_FILENAME),
        "vocab": os.path.join(output_dir, VOCAB_FILENAME),
    }


def list_cicids_csv_files(data_dir: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(data_dir)):
        if name.lower().endswith(".csv"):
            files.append(os.path.join(data_dir, name))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return files


def _normalize_label(label: object) -> str:
    value = str(label).strip()
    replacements = {
        "Web Attack � Brute Force": "Web Attack - Brute Force",
        "Web Attack � Sql Injection": "Web Attack - Sql Injection",
        "Web Attack � XSS": "Web Attack - XSS",
        "Web Attack – Brute Force": "Web Attack - Brute Force",
        "Web Attack – Sql Injection": "Web Attack - Sql Injection",
        "Web Attack – XSS": "Web Attack - XSS",
    }
    return replacements.get(value, value)


def _sample_balanced_rows(df: pd.DataFrame, max_samples: Optional[int], seed: int) -> pd.DataFrame:
    if max_samples is None or max_samples <= 0 or max_samples >= len(df):
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    grouped = {label: group.index.to_numpy() for label, group in df.groupby("Label")}
    labels = sorted(grouped.keys())
    selected: List[int] = []
    remaining = int(max_samples)

    base_quota = max(1, math.ceil(max_samples / max(len(labels), 1)))
    leftovers: List[int] = []
    for label in labels:
        label_indices = grouped[label].copy()
        rng.shuffle(label_indices)
        take = min(base_quota, label_indices.size, remaining)
        selected.extend(label_indices[:take].tolist())
        remaining = max(0, remaining - take)
        leftovers.extend(label_indices[take:].tolist())

    if len(selected) < max_samples and leftovers:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_samples - len(selected)])

    selected = selected[:max_samples]
    return df.loc[selected].reset_index(drop=True)


def load_cicids_raw_flows(data_dir: str, max_samples: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in list_cicids_csv_files(data_dir):
        df = pd.read_csv(path, low_memory=False)
        df.columns = [str(col).strip() for col in df.columns]
        if "Label" not in df.columns:
            raise ValueError(f"Expected 'Label' column in {path}")
        df["Label"] = df["Label"].apply(_normalize_label)

        df["source_file"] = os.path.basename(path)
        df["source_row_index"] = np.arange(df.shape[0], dtype=int)
        df["sample_id"] = df["source_file"] + "#" + df["source_row_index"].astype(str)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = _sample_balanced_rows(combined, max_samples=max_samples, seed=seed)
    return combined


def _clean_numeric_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, float], List[str]]:
    candidate_cols = [col for col in df.columns if col not in META_COLUMNS and col != "Label"]
    numeric = pd.DataFrame(index=df.index)
    for col in candidate_cols:
        numeric[col] = pd.to_numeric(df[col], errors="coerce")

    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    fill_values = numeric.median(numeric_only=True).fillna(0.0)
    numeric = numeric.fillna(fill_values).astype(np.float32)

    dropped = []
    feature_cols = []
    for col in numeric.columns:
        if numeric[col].isna().all():
            dropped.append(col)
        elif numeric[col].nunique(dropna=False) <= 1:
            dropped.append(col)
        else:
            feature_cols.append(col)

    cleaned = numeric[feature_cols].copy()
    return cleaned, feature_cols, {k: float(v) for k, v in fill_values.to_dict().items()}, dropped


def _sequence_feature_candidates(feature_columns: List[str]) -> List[str]:
    preferred = [
        "Destination Port",
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Packet Length Mean",
        "Packet Length Std",
        "SYN Flag Count",
        "PSH Flag Count",
        "ACK Flag Count",
        "Down/Up Ratio",
        "Average Packet Size",
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
        "Active Mean",
        "Idle Mean",
    ]
    selected = [name for name in preferred if name in feature_columns]
    if len(selected) < min(12, len(feature_columns)):
        for name in feature_columns:
            if name not in selected:
                selected.append(name)
            if len(selected) >= 16:
                break
    return selected


def _compute_bin_edges(series: pd.Series, n_bins: int) -> List[float]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float64)
    if values.size == 0:
        return [0.0, 1.0]
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, quantiles).astype(float).tolist()
    deduped = [edges[0]]
    for value in edges[1:]:
        if value > deduped[-1]:
            deduped.append(value)
    if len(deduped) == 1:
        deduped.append(deduped[0] + 1.0)
    return deduped


def _value_to_bin(value: float, edges: List[float]) -> int:
    if value <= edges[0]:
        return 0
    if value >= edges[-1]:
        return len(edges) - 2
    return int(np.searchsorted(edges[1:-1], value, side="right"))


def build_tokenized_flows(
    raw_df: pd.DataFrame,
    feature_columns: List[str],
    n_bins: int = 8,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    sequence_features = _sequence_feature_candidates(feature_columns)
    numeric_raw = pd.DataFrame(index=raw_df.index)
    for col in sequence_features:
        numeric_raw[col] = pd.to_numeric(raw_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    bin_edges = {col: _compute_bin_edges(numeric_raw[col], n_bins=n_bins) for col in sequence_features}

    token_lists: List[List[str]] = []
    for row_index in range(raw_df.shape[0]):
        row = raw_df.iloc[row_index]
        tokens = [
            f"label={_normalize_label(row['Label'])}",
            f"source={str(row['source_file']).replace(' ', '_')}",
        ]
        for feature in sequence_features:
            token_bin = _value_to_bin(float(numeric_raw.iloc[row_index][feature]), bin_edges[feature])
            safe_name = feature.lower().replace(" ", "_").replace("/", "_").replace(".", "_")
            tokens.append(f"{safe_name}=bin_{token_bin}")
        token_lists.append(tokens)

    vocab_tokens = sorted({token for tokens in token_lists for token in tokens})
    token_to_id = {token: index + 2 for index, token in enumerate(vocab_tokens)}
    vocab = {
        "pad_token": "<PAD>",
        "oov_token": "<OOV>",
        "pad_id": 0,
        "oov_id": 1,
        "token_to_id": token_to_id,
        "selected_features": sequence_features,
        "bin_edges": bin_edges,
        "max_sequence_length": max(len(tokens) for tokens in token_lists) if token_lists else 0,
    }

    tokens_df = raw_df[META_COLUMNS].copy()
    tokens_df["Label"] = raw_df["Label"].astype(str)
    tokens_df["token_sequence"] = [" ".join(tokens) for tokens in token_lists]
    tokens_df["sequence_length"] = [len(tokens) for tokens in token_lists]
    return tokens_df, vocab


def _cosine_01(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denom == 0.0:
        return 0.0
    cosine = float(np.dot(vector_a, vector_b) / denom)
    cosine = max(-1.0, min(1.0, cosine))
    return (cosine + 1.0) / 2.0


def _sample_positive_pairs(
    label_to_indices: Dict[str, np.ndarray],
    target_count: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    labels = [label for label, indices in label_to_indices.items() if len(indices) >= 2]
    pairs = set()
    attempts = 0
    max_attempts = max(target_count * 30, 1000)
    while len(pairs) < target_count and attempts < max_attempts and labels:
        label = labels[int(rng.integers(0, len(labels)))]
        a, b = sorted(rng.choice(label_to_indices[label], size=2, replace=False).tolist())
        pairs.add((int(a), int(b)))
        attempts += 1
    return sorted(pairs)


def _sample_negative_pairs(
    label_to_indices: Dict[str, np.ndarray],
    target_count: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    labels = [label for label, indices in label_to_indices.items() if len(indices) >= 1]
    if len(labels) < 2:
        return []

    pairs = set()
    attempts = 0
    max_attempts = max(target_count * 30, 1000)
    while len(pairs) < target_count and attempts < max_attempts:
        label_a, label_b = rng.choice(labels, size=2, replace=False).tolist()
        idx_a = int(rng.choice(label_to_indices[label_a]))
        idx_b = int(rng.choice(label_to_indices[label_b]))
        a, b = sorted((idx_a, idx_b))
        pairs.add((a, b))
        attempts += 1
    return sorted(pairs)


def _tokens_for_similarity(token_sequence: object) -> set:
    return {
        token
        for token in str(token_sequence).split()
        if not token.startswith("label=") and not token.startswith("source=")
    }


def _token_jaccard(token_sets: Optional[List[set]], index_a: int, index_b: int) -> float:
    if token_sets is None:
        return 0.0
    union = token_sets[index_a] | token_sets[index_b]
    if not union:
        return 0.0
    return float(len(token_sets[index_a] & token_sets[index_b]) / len(union))


def _sample_hard_negative_pairs(
    label_to_indices: Dict[str, np.ndarray],
    token_sets: List[set],
    target_count: int,
    rng: np.random.Generator,
    min_jaccard: float,
    max_jaccard: float,
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
    labels = [label for label, indices in label_to_indices.items() if len(indices) >= 1]
    if len(labels) < 2:
        return [], {}

    pairs = set()
    scores: Dict[Tuple[int, int], float] = {}
    attempts = 0
    max_attempts = max(target_count * 500, 10000)
    while len(pairs) < target_count and attempts < max_attempts:
        label_a, label_b = rng.choice(labels, size=2, replace=False).tolist()
        idx_a = int(rng.choice(label_to_indices[label_a]))
        idx_b = int(rng.choice(label_to_indices[label_b]))
        pair = tuple(sorted((idx_a, idx_b)))
        if pair in pairs:
            attempts += 1
            continue
        score = _token_jaccard(token_sets, pair[0], pair[1])
        if min_jaccard <= score <= max_jaccard:
            pairs.add(pair)
            scores[pair] = score
        attempts += 1
    return sorted(pairs), scores


def build_pairs_dataframe(
    flows_df: pd.DataFrame,
    feature_columns: Iterable[str],
    max_pairs: int,
    seed: int,
    token_flows_df: Optional[pd.DataFrame] = None,
    negative_strategy: str = "random",
    hard_negative_min_jaccard: float = 0.3,
    hard_negative_max_jaccard: float = 0.5,
) -> pd.DataFrame:
    if max_pairs <= 0:
        raise ValueError("max_pairs must be > 0")
    if negative_strategy not in {"random", "hard"}:
        raise ValueError(f"Unknown negative_strategy: {negative_strategy}")

    feature_columns = list(feature_columns)
    vectors = flows_df[feature_columns].to_numpy(dtype=np.float32)
    label_to_indices = {label: group.index.to_numpy() for label, group in flows_df.groupby("Label")}
    rng = np.random.default_rng(seed)
    token_sets = None
    if token_flows_df is not None and "token_sequence" in token_flows_df.columns:
        token_sets = [_tokens_for_similarity(sequence) for sequence in token_flows_df["token_sequence"].fillna("")]

    target_positive = max(1, max_pairs // 2)
    target_negative = max(1, max_pairs - target_positive)
    positive_pairs = _sample_positive_pairs(label_to_indices, target_positive, rng)
    hard_negative_scores: Dict[Tuple[int, int], float] = {}
    if negative_strategy == "hard" and token_sets is not None:
        negative_pairs, hard_negative_scores = _sample_hard_negative_pairs(
            label_to_indices,
            token_sets,
            target_negative,
            rng,
            min_jaccard=hard_negative_min_jaccard,
            max_jaccard=hard_negative_max_jaccard,
        )
    else:
        negative_pairs = []
    if len(negative_pairs) < target_negative:
        fallback_pairs = _sample_negative_pairs(label_to_indices, target_negative * 3, rng)
        seen = set(negative_pairs)
        for pair in fallback_pairs:
            if pair in seen:
                continue
            negative_pairs.append(pair)
            seen.add(pair)
            if len(negative_pairs) >= target_negative:
                break

    records = []
    for index_a, index_b in positive_pairs:
        token_jaccard = _token_jaccard(token_sets, index_a, index_b)
        records.append(
            {
                "flow_index_1": int(index_a),
                "flow_index_2": int(index_b),
                "sample_id_1": flows_df.iloc[index_a]["sample_id"],
                "sample_id_2": flows_df.iloc[index_b]["sample_id"],
                "label_1": flows_df.iloc[index_a]["Label"],
                "label_2": flows_df.iloc[index_b]["Label"],
                "target_similarity": _cosine_01(vectors[index_a], vectors[index_b]),
                "token_jaccard": token_jaccard,
                "negative_strategy": "positive",
                "is_duplicate": 1,
            }
        )

    for index_a, index_b in negative_pairs:
        pair = tuple(sorted((int(index_a), int(index_b))))
        token_jaccard = hard_negative_scores.get(pair, _token_jaccard(token_sets, int(index_a), int(index_b)))
        strategy = "hard" if pair in hard_negative_scores else "random"
        records.append(
            {
                "flow_index_1": int(index_a),
                "flow_index_2": int(index_b),
                "sample_id_1": flows_df.iloc[index_a]["sample_id"],
                "sample_id_2": flows_df.iloc[index_b]["sample_id"],
                "label_1": flows_df.iloc[index_a]["Label"],
                "label_2": flows_df.iloc[index_b]["Label"],
                "target_similarity": 0.0,
                "token_jaccard": token_jaccard,
                "negative_strategy": strategy,
                "is_duplicate": 0,
            }
        )

    pairs_df = pd.DataFrame.from_records(records)
    if pairs_df.empty:
        raise ValueError("No training pairs could be generated from the prepared CIC-IDS data.")

    pairs_df = pairs_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return pairs_df


def prepare_cicids_dataset(
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    max_pairs: int = 20000,
    seed: int = 42,
    negative_strategy: str = "random",
    hard_negative_min_jaccard: float = 0.3,
    hard_negative_max_jaccard: float = 0.5,
) -> Dict[str, str]:
    data_dir = data_dir or default_raw_data_dir()
    output_dir = output_dir or default_processed_data_dir()
    os.makedirs(output_dir, exist_ok=True)

    raw_df = load_cicids_raw_flows(data_dir=data_dir, max_samples=max_samples, seed=seed)
    numeric_df, feature_columns, fill_values, dropped_columns = _clean_numeric_features(raw_df)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)
    tokens_df, vocab = build_tokenized_flows(raw_df, feature_columns=feature_columns)

    flows_df = raw_df[META_COLUMNS].copy()
    flows_df["Label"] = raw_df["Label"].astype(str)
    flows_df[feature_columns] = scaled.astype(np.float32)

    pairs_df = build_pairs_dataframe(
        flows_df,
        feature_columns=feature_columns,
        max_pairs=max_pairs,
        seed=seed,
        token_flows_df=tokens_df,
        negative_strategy=negative_strategy,
        hard_negative_min_jaccard=hard_negative_min_jaccard,
        hard_negative_max_jaccard=hard_negative_max_jaccard,
    )

    metadata = {
        "raw_data_dir": data_dir,
        "n_flows": int(flows_df.shape[0]),
        "n_pairs": int(pairs_df.shape[0]),
        "feature_columns": feature_columns,
        "dropped_columns": dropped_columns,
        "label_counts": {str(k): int(v) for k, v in flows_df["Label"].value_counts().sort_index().to_dict().items()},
        "max_samples": max_samples,
        "max_pairs": int(max_pairs),
        "seed": int(seed),
        "negative_strategy": negative_strategy,
        "hard_negative_min_jaccard": float(hard_negative_min_jaccard),
        "hard_negative_max_jaccard": float(hard_negative_max_jaccard),
    }

    preprocessor = {
        "feature_columns": feature_columns,
        "fill_values": {k: fill_values[k] for k in feature_columns},
        "dropped_columns": dropped_columns,
        "scaler": scaler,
        "meta_columns": META_COLUMNS,
    }

    paths = prepared_paths(output_dir)
    flows_df.to_csv(paths["flows"], index=False)
    tokens_df.to_csv(paths["flows_tokens"], index=False)
    pairs_df.to_csv(paths["pairs"], index=False)
    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(paths["preprocessor"], "wb") as f:
        pickle.dump(preprocessor, f)
    with open(paths["vocab"], "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    return paths


def load_prepared_flows(output_dir: Optional[str] = None) -> pd.DataFrame:
    output_dir = output_dir or default_processed_data_dir()
    path = prepared_paths(output_dir)["flows"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prepared flows not found: {path}")
    return pd.read_csv(path)


def load_prepared_pairs(output_dir: Optional[str] = None) -> pd.DataFrame:
    output_dir = output_dir or default_processed_data_dir()
    path = prepared_paths(output_dir)["pairs"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prepared pairs not found: {path}")
    return pd.read_csv(path)


def load_prepared_token_flows(output_dir: Optional[str] = None) -> pd.DataFrame:
    output_dir = output_dir or default_processed_data_dir()
    path = prepared_paths(output_dir)["flows_tokens"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prepared token flows not found: {path}")
    return pd.read_csv(path)


def load_preprocessor(output_dir: Optional[str] = None) -> Dict[str, object]:
    output_dir = output_dir or default_processed_data_dir()
    path = prepared_paths(output_dir)["preprocessor"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessor not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_metadata(output_dir: Optional[str] = None) -> Dict[str, object]:
    output_dir = output_dir or default_processed_data_dir()
    path = prepared_paths(output_dir)["metadata"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_vocab(output_dir: Optional[str] = None) -> Dict[str, object]:
    output_dir = output_dir or default_processed_data_dir()
    path = prepared_paths(output_dir)["vocab"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vocabulary not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_token_sequence(token_sequence: str, vocab: Dict[str, object]) -> List[int]:
    token_to_id = vocab["token_to_id"]
    oov_id = int(vocab.get("oov_id", 1))
    return [int(token_to_id.get(token, oov_id)) for token in str(token_sequence).split() if token]


def build_sequence_matrix(tokens_df: pd.DataFrame, vocab: Dict[str, object], max_length: Optional[int] = None) -> np.ndarray:
    sequence_ids = [encode_token_sequence(token_sequence, vocab) for token_sequence in tokens_df["token_sequence"].fillna("")]
    max_length = int(max_length or vocab.get("max_sequence_length") or max((len(seq) for seq in sequence_ids), default=0))
    matrix = np.zeros((len(sequence_ids), max_length), dtype=np.int32)
    for row_index, sequence in enumerate(sequence_ids):
        length = min(len(sequence), max_length)
        if length:
            matrix[row_index, :length] = np.asarray(sequence[:length], dtype=np.int32)
    return matrix
