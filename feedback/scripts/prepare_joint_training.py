#!/usr/bin/env python
"""Verify and publish qualified NAVSIM + SimScale training indices."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "feedback/src"), str(ROOT)]
from drivesta_feedback.joint_training import main

if __name__ == "__main__":
    main()
