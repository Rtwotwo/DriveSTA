"""Generate EPDMS banks using the isolated NAVSIM v2 runtime."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "feedback/src"))
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
from drivesta_feedback.epdms.dataset import main

if __name__ == "__main__":
    main()
