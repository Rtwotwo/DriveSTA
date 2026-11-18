#!/usr/bin/env python
"""Explicit full-score recovery; high-score shortfalls remain separately recorded."""
import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import _bootstrap
from drivesta_feedback.batch.full_score_backfill import main

if __name__ == "__main__":
    main()
