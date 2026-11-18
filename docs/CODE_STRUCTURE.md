# DriveSTA code structure

The release follows the NAVSIM layout used by [ChainFlow-VLA](https://github.com/AFARI-Research/ChainFlow-VLA). The local reference inspected for this organization is commit `91935f3e55db6a321ec44443a8caa819fd30a996`; the public repository HEAD inspected was `0f3565f66546139a0183bc5807d5f7234ae2d4b0`.

| Responsibility | Location |
| --- | --- |
| NAVSIM adapter, optimizer setup and checkpoint initialization | `navsim/agents/drivesta/drivesta_agent.py` |
| Shared proposal model and optional temporal refinement | `navsim/agents/drivesta/drivesta_model.py` |
| Input/target builders | `navsim/agents/drivesta/builders/` |
| DINOv2, JEPA, trajectory/score decoders, shared layers and objectives | `navsim/agents/drivesta/{encoders,decoders,layers,losses}/` |
| Train entry | `navsim/planning/script/run_training.py` |
| Dataset, sampler and Lightning orchestration | `navsim/planning/training/drivesta/` |
| Single-scene prediction | `navsim/planning/script/run_inference.py` |
| Distributed NAVSIM v1 PDMS entry | `navsim/planning/script/run_pdm_score_drivesta_multi_gpu.py` |
| Official online labels and validation | `navsim/evaluate/drivesta/` |
| Shared agent parameters | `navsim/planning/script/config/common/agent/` |
| Training recipes | `navsim/planning/script/config/training/` |
| Benchmark settings | `navsim/planning/script/config/pdm_scoring/` |
| Released checkpoint snapshots | `navsim/planning/script/config/checkpoints/` |
| Public launchers | `scripts/training/`, `scripts/evaluation/` |
| Serialized-name, archive-index and historical-path adapters | `navsim/common/compatibility/` |
| Trajectory mining | `feedback/src/drivesta_feedback/`, `navsim/tools/trajectory_mining/` |

Both proposal-only and refined profiles instantiate `DriveSTAAgent` and `DriveSTAModel`. The model constructs JEPA and the residual path only when `use_diffusion_decoder=true`. Proposal/refinement objective and data implementations remain separately named by responsibility, outside the model's orchestration path. Base/Scale differences belong to YAML recipes.

Third-party names such as NAVSIM, nuPlan, DINOv2 and Drive-JEPA identify actual dependencies. Copyright/license attribution and immutable serialized metadata are retained. Historical checkpoint identifiers are translated in memory; there are no old-name model import aliases in the public agent package.
