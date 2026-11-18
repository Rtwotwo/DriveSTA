# Inference and evaluation

Complete [INSTALL.md](INSTALL.md), obtain a checkpoint and observations using [DATA_PREPARATION.md](DATA_PREPARATION.md), and source `scripts/environment.sh`.

## Predict a trajectory for one scene

```bash
python -m navsim.planning.script.run_inference \
  --profile base --token SCENE_TOKEN --split test --device cuda \
  --output outputs/prediction.json
```

Profiles are `base`, `scale`, `base_proposal` and `scale_proposal`. Use `--checkpoint /path/to/model.ckpt` for another checkpoint with the same architecture/protocol. `--data-root` overrides `OPENSCENE_DATA_ROOT`, and `--log-name LOG_BASENAME` restricts loading to a known log. `--device cpu` is supported but slower.

The command reads four historical ego frames and the camera observations selected by the profile. It never loads future target supervision. It initializes with strict tensor checks, switches the model to evaluation mode and writes JSON with the scene token, profile, coordinate convention and eight predicted `(x_m, y_m, heading_rad)` poses. Sampling is 0.5 seconds and the horizon is four seconds. This is inference only; it does not calculate PDMS.

## Prepare NAVTEST metric caches

Official scoring needs maps and precomputed metric caches matching the split. Reuse an existing NAVSIM v1 cache at `$NAVSIM_CACHE_ROOT/metric_cache`, or build it once:

```bash
python -m navsim.planning.script.run_metric_caching \
  train_test_split=navtest \
  cache.cache_path="$NAVSIM_CACHE_ROOT/metric_cache"
```

Inspect `python -m navsim.planning.script.run_metric_caching --cfg job --resolve` before an expensive cache job. Maps, logs and cache versions must agree. See [DATA_LAYOUT.md](DATA_LAYOUT.md) for the distinction between evaluation and training scorer caches.

## Run the full NAVTEST benchmark

```bash
GPUS_PER_NODE=1 bash scripts/evaluation/run_drivesta_pdm_score_evaluation.sh base
GPUS_PER_NODE=8 bash scripts/evaluation/run_drivesta_pdm_score_evaluation.sh scale
```

The shell launcher starts distributed inference, restores the released checkpoint's parameter snapshot, enforces strict loading and uses FP32 evaluation. The default NAVTEST recipe requires all **12,146** expected scenes and complete metrics; a partial run must not be reported as the full benchmark.

```bash
CHECKPOINT_PATH=/absolute/path/to/model.ckpt \
METRIC_CACHE_PATH=/absolute/path/to/navtest_metric_cache \
OUTPUT_DIR=/absolute/path/to/evaluation_output \
GPUS_PER_NODE=2 \
bash scripts/evaluation/run_drivesta_pdm_score_evaluation.sh base
```

For a custom trained checkpoint, override `training_config_path` with that run's saved Hydra configuration so evaluation restores its actual model settings. The selected snapshot/profile must match the checkpoint; generator and supervision protocol checks remain enforced.

## One-scene scoring smoke test

Use a real NAVTEST token for which observations and metric cache are present:

```bash
bash scripts/evaluation/run_drivesta_pdm_score_evaluation.sh base \
  'train_test_split.scene_filter.tokens=["SCENE_TOKEN"]' \
  train_test_split.scene_filter.max_scenes=1 expected_eval_scenes=1
```

Keep the double quotes around each token inside the Hydra list: a token containing `e` can otherwise be parsed as a floating-point number.

This validates the inference-to-scoring chain for one scene. It is not a benchmark score. Results and completeness summaries are written under `OUTPUT_DIR`, or the default `$NAVSIM_EXP_ROOT/evaluation/<profile>_<timestamp>/`.

## Other benchmarks

EPDMS/NAVHARD definitions are retained under `navsim/planning/script/config/epdms/` and `navsim/v2/`. Their preparation is described in [EPDMS_MINING.md](EPDMS_MINING.md). Do not treat the NAVSIM v1 PDMS launcher as an EPDMS or HUGSIM evaluator: those metrics require their corresponding external benchmark runtime and data. The project-page results describe the reported experiments; this release guide validates the included NAVSIM v1 training/inference path.
