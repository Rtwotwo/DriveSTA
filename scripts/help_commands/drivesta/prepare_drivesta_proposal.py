"""Prepare the immutable target view used by DriveSTA."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "feedback/src")]
from drivesta_feedback.batch.prepare_feedback import main

if __name__ == "__main__":
    main()
