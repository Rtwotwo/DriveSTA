# Contributing

Use Python 3.9 or newer and preserve checkpoint parameter names. Changes to sampling,
trajectory coordinates, score components, data splits, or losses require explicit
tests and documentation. Files in `encoders/jepa_vendor` and `nuplan-devkit` retain
their upstream license headers.

Before submitting a change:

```bash
source scripts/environment.sh
python -m black --check navsim feedback/src feedback/scripts scripts tests
python -m ruff check navsim feedback/src feedback/scripts scripts tests
python -m pytest -q
```

Keep generated outputs under `outputs/`. Do not commit checkpoints, sensor data,
credentials, machine-specific configuration, or experiment logs. Store downloadable
release assets as compressed archives under `weights/`; extraction targets are
`checkpoints/` and `data/`.
