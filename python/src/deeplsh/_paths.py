from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("DEEPLSH_REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "python" / "src" / "deeplsh").exists() and (candidate / "datasets").exists():
            return candidate
    return start


def datasets_dir() -> Path:
    return repo_root() / "datasets"


def artifacts_dir() -> Path:
    return repo_root() / "artifacts"


def stacktraces_dataset_dir() -> Path:
    return datasets_dir() / "stacktraces"


def stacktraces_artifacts_dir() -> Path:
    return artifacts_dir() / "stacktraces"


def cicids_raw_dir() -> Path:
    return datasets_dir() / "cicids" / "raw"


def cicids_processed_dir(kind: str = "full") -> Path:
    kind = (kind or "full").strip()
    return datasets_dir() / "cicids" / "processed" / kind


def cicids_artifacts_dir() -> Path:
    return artifacts_dir() / "cicids"

