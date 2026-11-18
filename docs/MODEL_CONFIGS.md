# DriveSTA configuration reference

All public Hydra configurations live under `navsim/planning/script/config/`.

| Group | Files | Responsibility |
| --- | --- | --- |
| `common/agent` | `drivesta.yaml`, `drivesta_proposal.yaml` | Shared architecture, loss, sensors and optimizer defaults |
| `training` | `drivesta_base.yaml`, `drivesta_scale.yaml`, `drivesta_base_proposal.yaml`, `drivesta_scale_proposal.yaml` | Data mixture, runtime, initialization and recipe overrides |
| `pdm_scoring` | `drivesta_eval.yaml`, `drivesta_proposal_eval.yaml` | Official scoring settings, precision and completeness |
| `checkpoints` | `base.yaml`, `scale.yaml`, `base_proposal.yaml`, `scale_proposal.yaml` | Full parameter snapshots matching released weights |

Training recipes compose shared agent defaults and contain only their differences. Checkpoint snapshots are intentionally complete: they document the exact parameters needed to reproduce released inference independently of later training-default changes.

Key settings:

| Setting | Meaning |
| --- | --- |
| `training_mode` | `proposal` or `refinement`; selects the matching training runner |
| `data_profile` | `real` or `mixed`; selects data handling and missing-label policy |
| `agent.config.use_diffusion_decoder` | Enables the temporal residual/JEPA path in the shared model |
| `agent.config.training_phase` | `joint` for the released refined model |
| `agent.loss.rank_weight` | Ranking objective coefficient; unrelated to LoRA rank or distributed rank |
| `agent.checkpoint_path` | Model initialization or strict evaluation checkpoint |
| `train_ckpt_path` | Full Lightning training resume; initialization must be null |
| `feedback.*index*` | Canonical immutable supervision-index aliases |
| `trainer.params.devices`, `agent.num_gpus` | Runtime device count and scheduling metadata |
| `dataloader.params.batch_size`, `agent.batch_size` | Loader batch and effective-batch learning-rate calculation |

Use `bash scripts/training/train_drivesta.sh PROFILE --cfg job --resolve` to inspect the final recipe. Evaluation restores a matching snapshot, checks the sampling protocol and uses deterministic token-conditioned noise. A numerical setting change is an experiment change, even when the file still loads successfully.
