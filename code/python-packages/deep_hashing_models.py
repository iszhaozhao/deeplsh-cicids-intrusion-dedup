import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
python_src = repo / "python" / "src"
if python_src.exists() and str(python_src) not in sys.path:
    sys.path.insert(0, str(python_src))

from deeplsh.core.deep_hashing_models import *  # noqa: F401,F403

