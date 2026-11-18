# Trajectory mining

The miner constructs candidate paths, evaluates the official NAVSIM v1 PDMS,
retains full scored banks and high-score banks, and screens auxiliary training
targets. The release includes generation and recovery code; EPDMS-mined data is
outside this release.

## Inputs

Source `scripts/environment.sh` after setting your data roots. Provide official
Navtrainval logs, maps and metric caches, or the five SimScale PDM groups
`synthetic_reaction_pdm_v1.0-0` through `-4`. SimScale source logs, images and
invalid-token lists must share the same generation. The supplied synthetic
filters and real train/val manifest preserve the original scene selection.

For a different local cache layout, first regenerate the real manifest:

```bash
python feedback/scripts/prepare_scene_manifest.py \
  --metadata "$NAVSIM_CACHE_ROOT/train_metric_cache/metadata/train_metric_cache_metadata_node_0.csv" \
  --cache-root "$NAVSIM_CACHE_ROOT/train_metric_cache" \
  --split-config navsim/planning/script/config/training/default_train_val_test_log_split.yaml \
  --output feedback/data/dense_gt_v1/manifests/navtrainval_v1 --check-files
```

The NAVSIM cache entry points are `navsim.planning.script.run_metric_caching`,
`run_train_metric_caching`, `run_train_metric_caching_synt`, and
`run_dataset_caching`. Their `--help` and `--cfg job` options describe Hydra
settings; choose the matching split and agent before generating large caches.

## Generate or resume

```bash
bash scripts/mine.sh navtrainval estimate
MINING_WORKERS=16 bash scripts/mine.sh navtrainval run
bash scripts/mine.sh navtrainval status

bash scripts/mine.sh simscale manifest
MINING_WORKERS=16 bash scripts/mine.sh simscale run
bash scripts/mine.sh simscale status
```

New output defaults to `data/mining/<dataset>`. Set `MINING_OUTPUT` to use another
new run directory. Published trajectory archives are read-only inputs and should
not be used as a mining work directory. The generation code checks existing run
contracts before resuming. For a small pilot, append `--limit 8` and choose a
separate `MINING_OUTPUT`; a full run cannot resume a different limited manifest.

The real profile uses a 3,072-query base budget, retains all scores, and targets a
256-member high-score bank. `--query-budget 8192` or `32768` selects the declared
larger budget. SimScale uses explicit physical constraints and an inferred-actor
tail extension for full-window NAVSIM v1 TTC. It does not silently replace this
protocol with within-window TTC or EPDMS.

## Recovery and target preparation

The published banks include recovery and consolidation beyond the initial pass.
A new initial run is not guaranteed to reproduce the final recovered banks merely
by using the same nominal candidate count. Inspect `PARTIAL.json`, `status.json`,
shortfall tokens and per-scene results before selecting the appropriate recovery.

```bash
python feedback/scripts/recover_shortfalls.py --help
python feedback/scripts/recover_simscale.py --help
python feedback/scripts/repair_simscale_geometry.py --help
python feedback/scripts/merge_recovered_scenes.py --help
python feedback/scripts/publish_recovery.py --help
python feedback/scripts/backfill_full_scores.py --help
```

The corresponding explicit policies are under `feedback/configs/`. Recovery output
is audited before publication. Do not replace partial states with success markers.
The complete published metadata records the exact recovery revisions and shortfalls.

To build a new screened target index from an eligible scene index:

```bash
python scripts/help_commands/drivesta/prepare_chainflow_feedback_v2.py \
  --source-index /path/to/eligible_scene_index.json \
  --cache-root "$OFFICIAL_PDM_CACHE" \
  --policy feedback/configs/generator_targets_feedback32.yaml \
  --output outputs/targets32 --workers 16
```

The miner's `training_splits.py` and `joint_training.py` publish qualified synthetic
membership and data lineage. For the released checkpoints, use the supplied
immutable indexes: rebuilding changes provenance and must be treated as a new
training experiment. Real and synthetic video-index builders are retained under
`scripts/help_commands/drivesta/prepare_chainflow_feedback_refinement_*.py`.

## Continue a new run with insufficient high-score trajectories

For a new real-data run, use the built-in continuation with the same output and
scene selection. Full scored rows remain available even if fewer than 256 paths
reach the high-score threshold:

```bash
bash scripts/mine.sh navtrainval run --query-budget 8192
bash scripts/mine.sh navtrainval run --query-budget 32768
```

For a new synthetic run, the basic bounded recovery profile reads
`data/mining/simscale`. Edit its `source` when using another mining directory:

```bash
python feedback/scripts/recover_simscale.py \
  --config feedback/configs/simscale_recovery.yaml \
  --tokens data/mining/simscale/shortfall_tokens.txt \
  --output data/mining/simscale_recovery --workers 8
```

This command writes a separate recovery output. Add `--merge` only when ready to
publish the validated replacements back into that new mining run. Geometry/TTO
profiles additionally require the matching cache-repair products; their
`cache_repairs` entries are explicit configurable input locations. The
`shortfall690.yaml` workflow reproduces a historical 690-scene retention upgrade
and expects its original contract artifacts. It is not the default continuation
for a newly generated bank.

The default mining pipeline produces the seven-component PDMS corpus.
[Optional EPDMS mining code](EPDMS_MINING.md) is also retained, with separate
inputs and output directories. EPDMS-mined banks are deliberately excluded. The independent v2 replay
evaluator remains available for a supplied NAVTEST/NAVHARD submission and official
v2 caches; its results must not be labeled PDMS:

```bash
python -m navsim.evaluate.epdms.evaluate --benchmark navtest \
  --submission /path/to/submission.pkl --metric-cache /path/to/navtest_metric_cache_v2 \
  --output-dir outputs/epdms_navtest --workers 8
```
