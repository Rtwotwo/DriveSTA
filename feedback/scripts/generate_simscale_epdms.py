"""Explicit CLI for synthetic EPDMS banks; importing this file launches no jobs."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "feedback/src"))

if __name__ == "__main__":
    from drivesta_feedback.epdms.simscale import main

    main()
