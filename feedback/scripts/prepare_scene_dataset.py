#!/usr/bin/env python
"""Explicit estimate/status/run entry point for the 0.30 m scene dataset."""
import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import _bootstrap
from drivesta_feedback.batch.production import main

if __name__ == "__main__":
    main()
