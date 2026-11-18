"""Recover strict NAVSIM EPDMS targets in the existing main corpus."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "feedback/src"))
sys.path.insert(0, str(ROOT / "clones/navsim"))
from drivesta_feedback.epdms.recovery import main

if __name__ == "__main__":
    main()
