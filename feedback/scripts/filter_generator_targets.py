"""Create a screened generator-target index; original banks remain unchanged."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drivesta_feedback.batch.filter_targets import main

if __name__ == "__main__":
    main()
