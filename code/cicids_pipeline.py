import json
import math
import os
import pickle
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROCESSED_DIRNAME = "cic_ids2017_processed"
FLOWS_FILENAME = "flows.csv"
PAIRS_FILENAME = "pairs_train.csv"
METADATA_FILENAME = "metadata.json"
PREPROCESSOR_FILENAME = "preprocessor.pkl"

META_COLUMNS = ["sample_id", "Label", "source_file", "source_row_index"]


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_raw_data_dir() -> str:
    return os.path.join(project_root(), "MachineLearningCVE")


def default_processed_data_dir() -> str:
    return os.path.join(project_root(), "data", PROCESSED_DIRNAME)


def prepared_paths(output_dir: str) -> Dict[str, str]:
    return {
        "flows": os.path.join(output_dir, FLOWS_FILENAME),
        "pairs": os.path.join(output_dir, PAIRS_FILENAME),
        "metadata": os.path.join(output_dir, METADATA_FILENAME),
        "preprocessor": os.path.join(output_dir, PREPROCESSOR_FILENAME),
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


def build_pairs_dataframe(flows_df: pd.DataFrame, feature_columns: Iterable[str], max_pairs: int, seed: int) -> pd.DataFrame:
    if max_pairs <= 0:
        raise ValueError("max_pairs must be > 0")

    feature_columns = list(feature_columns)
    vectors = flows_df[feature_columns].to_numpy(dtype=np.float32)
    label_to_indices = {label: group.index.to_numpy() for label, group in flows_df.groupby("Label")}
    rng = np.random.default_rng(seed)

    target_positive = max(1, max_pairs // 2)
    target_negative = max(1, max_pairs - target_positive)
    positive_pairs = _sample_positive_pairs(label_to_indices, target_positive, rng)
    negative_pairs = _sample_negative_pairs(label_to_indices, target_negative, rng)

    records = []
    for index_a, index_b in positive_pairs:
        records.append(
            {
                "flow_index_1": int(index_a),
                "flow_index_2": int(index_b),
                "sample_id_1": flows_df.iloc[index_a]["sample_id"],
                "sample_id_2": flows_df.iloc[index_b]["sample_id"],
                "label_1": flows_df.iloc[index_a]["Label"],
                "label_2": flows_df.iloc[index_b]["Label"],
                "target_similarity": _cosine_01(vectors[index_a], vectors[index_b]),
                "is_duplicate": 1,
            }
        )

    for index_a, index_b in negative_pairs:
        records.append(
            {
                "flow_index_1": int(index_a),
                "flow_index_2": int(index_b),
                "sample_id_1": flows_df.iloc[index_a]["sample_id"],
                "sample_id_2": flows_df.iloc[index_b]["sample_id"],
                "label_1": flows_df.iloc[index_a]["Label"],
                "label_2": flows_df.iloc[index_b]["Label"],
                "target_similarity": 0.0,
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
) -> Dict[str, str]:
    data_dir = data_dir or default_raw_data_dir()
    output_dir = output_dir or default_processed_data_dir()
    os.makedirs(output_dir, exist_ok=True)

    raw_df = load_cicids_raw_flows(data_dir=data_dir, max_samples=max_samples, seed=seed)
    numeric_df, feature_columns, fill_values, dropped_columns = _clean_numeric_features(raw_df)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)

    flows_df = raw_df[META_COLUMNS].copy()
    flows_df["Label"] = raw_df["Label"].astype(str)
    flows_df[feature_columns] = scaled.astype(np.float32)

    pairs_df = build_pairs_dataframe(flows_df, feature_columns=feature_columns, max_pairs=max_pairs, seed=seed)

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
    pairs_df.to_csv(paths["pairs"], index=False)
    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(paths["preprocessor"], "wb") as f:
        pickle.dump(preprocessor, f)

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
