import sys
from pathlib import Path


def _ensure_python_src_on_path() -> None:
    repo = Path(__file__).resolve().parents[1]
    python_src = repo / "python" / "src"
    if python_src.exists() and str(python_src) not in sys.path:
        sys.path.insert(0, str(python_src))


def main() -> int:
    _ensure_python_src_on_path()
    from deeplsh.cli import main as cli_main

    return int(cli_main())


if __name__ == "__main__":
    raise SystemExit(main())

