# DriveSTA release guide

This is the entry index for the complete code, model and trajectory release.

1. [Installation](INSTALL.md): environment, dependencies and import/config checks.
2. [Data preparation](DATA_PREPARATION.md): checkpoints, external observations, trajectory extraction, immutable indexes and initialization backbones.
3. [Training](TRAINING.md): Base/Scale proposal learning, refinement, resume and runtime overrides.
4. [Inference and evaluation](INFERENCE.md): single-scene prediction, metric caching and full NAVTEST PDMS.
5. [Download verification and repair](DOWNLOAD_VERIFY.md): scene lookup, byte checks and recovery.
6. [Code structure](CODE_STRUCTURE.md) and [parameter reference](MODEL_CONFIGS.md).
7. [Validation record](VALIDATION.md).

The asset inventory is `docs/release/files.json`; trajectory archive metadata is `docs/release/archives.json`. Canonical checkpoint names are `base_proposal.ckpt`, `base.ckpt`, `scale_proposal.ckpt` and `scale.ckpt`. Canonical trajectory archives begin with `drivesta_`.

Checkpoint tensors and trajectory archive contents are unchanged. The downloader handles historical remote filenames automatically. Only serialized compatibility boundaries and immutable provenance retain historical identifiers. Full training and complete benchmark reruns are not implied by a release-layout check.
