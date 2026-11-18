# DriveSTA trajectory mining

This package constructs candidate trajectories, evaluates official NAVSIM PDMS,
preserves full scored banks, screens targets, and supports recovery and training
index generation. The production implementations are in `src/drivesta_feedback/batch/`.

See [the mining guide](../docs/MINING.md) for input caches, configurations,
generation, resume, recovery and label semantics. Start from the repository root:

```bash
source scripts/environment.sh
bash scripts/mine.sh navtrainval estimate
bash scripts/mine.sh navtrainval run
bash scripts/mine.sh simscale run
```

The published trajectory archives are immutable inputs under `data/trajectories/`
after extraction. New runs default to `data/mining/`. EPDMS trajectory banks are
not part of this release.
