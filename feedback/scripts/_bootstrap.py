"""Make the local source package importable without modifying the environment."""

from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
