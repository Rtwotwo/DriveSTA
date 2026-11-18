#!/usr/bin/env python3
"""Merge completed recovery scenes into the original feedback training dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drivesta_feedback.batch.merge_recovered import main

if __name__ == "__main__":
    main()
