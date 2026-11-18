# Train DriveSTA

Complete [installation](INSTALL.md) and [data preparation](DATA_PREPARATION.md), then source `scripts/environment.sh`. Public entry points are under `scripts/training/`; the Hydra runner is `navsim/planning/script/run_training.py`. The network and NAVSIM adapter remain in `navsim/agents/drivesta/`.

## Recipes and optimization order

| Profile | Supervision | Data mixture | Epochs | Default per-device batch | Default devices |
| --- | --- | --- | ---: | ---: | ---: |
| `base_proposal` | Proposal and score learning | 103,263 real | 25 | 6 | 8 |
| `base` | Joint temporal refinement | 103,263 real | 40 | 6 | 8 |
| `scale_proposal` | Proposal and score learning | 232,672 mixed | 25 | 6 | 8 |
| `scale` | Joint temporal refinement | 232,672 mixed | 40 | 6 | 8 |

Inspect the resolved recipe for authoritative values, including any overrides:

```bash
bash scripts/training/train_drivesta.sh base_proposal --cfg job --resolve
bash scripts/training/train_drivesta.sh base --cfg job --resolve
```

The two objectives use the same `DriveSTAAgent` / `DriveSTAModel`. `training_mode=proposal` disables temporal refinement; `training_mode=refinement` enables the residual decoder and JEPA objective. The runner checks that model capability and recipe match. Changing the mode alone is not a complete recipe change: select the corresponding profile.

## Train Base

```bash
bash scripts/training/train_drivesta.sh base_proposal
```

The proposal recipe starts from the pretrained image backbone. Find the generated checkpoint under the run's output directory. Then initialize refinement with the matching trained proposal checkpoint:

```bash
bash scripts/training/train_drivesta.sh base \
  agent.checkpoint_path=/absolute/path/to/proposal_checkpoint.ckpt
```

Alternatively, download `base_proposal` and use the refinement recipe's default initialization. A proposal initialization requires the Drive-JEPA pretrained artifact described in the data guide. Refinement jointly learns the residual planner and score adaptation while JEPA uses its expert-future objective; the planning adapters receive detached JEPA features.

## Train Scale

```bash
bash scripts/training/train_drivesta.sh scale_proposal
bash scripts/training/train_drivesta.sh scale \
  agent.checkpoint_path=/absolute/path/to/scale_proposal_checkpoint.ckpt
```

Use the matching Scale proposal checkpoint, the immutable mixed-data specification and adapted synthetic scoring caches. The mixed recipe masks unavailable long-horizon expert labels. Do not substitute the complete 204,831-scene synthetic archive for the fixed 129,409-scene synthetic training selection.

## Resume an interrupted run

Initialization and full resume are different operations. `agent.checkpoint_path` initializes model weights; `train_ckpt_path` restores the Lightning optimizer/scheduler/training state. Select exactly one:

```bash
bash scripts/training/train_drivesta.sh base \
  train_ckpt_path=/absolute/path/to/last.ckpt \
  agent.checkpoint_path=null
```

Keep the original profile, data specification and supervision settings when resuming. The runtime checks checkpoint metadata, sampler epoch and RNG state to prevent silent changes to the continuation protocol.

## Runtime overrides

```bash
# Example of a smaller run; this changes the effective batch and recipe.
bash scripts/training/train_drivesta.sh base_proposal \
  trainer.params.devices=1 agent.num_gpus=1 \
  dataloader.params.batch_size=2 agent.batch_size=2 \
  dataloader.params.num_workers=4

# Explicit output folder and loss override:
bash scripts/training/train_drivesta.sh base \
  output_dir=/path/to/experiments/my_run agent.loss.rank_weight=0.0
```

Keep the loader batch and agent batch settings synchronized, and likewise the trainer and agent GPU counts; the learning-rate schedule uses the effective optimizer batch. All remaining arguments are Hydra overrides. Use a full quoted Hydra assignment for paths containing spaces or `=`. `RANK_WEIGHT` is also supported by the training launcher, but the equivalent `agent.loss.rank_weight` override makes the resolved configuration explicit.

The supplied full-trainval recipe disables overlapping validation during optimization. Use the held-out NAVTEST evaluation entry for reporting. Output directories contain resolved Hydra configurations and Lightning logs/checkpoints. Always retain these together when recording results.
