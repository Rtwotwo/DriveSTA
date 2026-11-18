# DriveSTA model

`drivesta_agent.py` exposes `DriveSTAAgent`; `drivesta_model.py` exposes `DriveSTAModel`.
All four released profiles instantiate these same classes. Configuration determines whether the temporal residual decoder and JEPA components are constructed.

Model building blocks live in `builders/`, `encoders/`, `decoders/`, `layers/`, `losses/` and `utils/`. Training data and Lightning orchestration live in `navsim/planning/training/drivesta/`; evaluation helpers live in `navsim/evaluate/drivesta/`. All public YAML configuration is centralized in `navsim/planning/script/config/`.

See the repository [README](../../../README.md), [code structure](../../../docs/CODE_STRUCTURE.md), [training guide](../../../docs/TRAINING.md) and [inference guide](../../../docs/INFERENCE.md).
