# Optional EPDMS mining implementation

The release retains the real-data pair-conditioned generator, bounded recovery,
independent official replay validator and SimScale generator under
`feedback/src/drivesta_feedback/epdms/`. These use the isolated `navsim.v2`
namespace; the published PDMS banks and four checkpoints keep their original v1
protocol. No EPDMS-mined bank or EPDMS base cache is included in `weights/`.

## Navtrainval prerequisites

The retained real-data generator is a second-pass pipeline. Set `EPDMS_BASE_ROOT`
to the original first-pass 103,288-scene EPDMS base directory. It must contain
`scene_summary.csv.gz`, per-scene `summary.json`, and official v2 caches under
`metric_cache/<log_name>/unknown/<token>/metric_cache.pkl`. The code checks the
complete census and keeps adjacent-frame ordering. A v1 PDMS archive cannot
replace this input. The historical server input is
`feedback/runs/navtrain_epdms_20261027/full103288_v1` in the research checkout.
This prerequisite is intentionally not part of the data release.

```bash
source scripts/environment.sh
export EPDMS_BASE_ROOT=/path/to/original/epdms_base
EPDMS_OUTPUT=data/mining/navtrainval_epdms_pilot \
  bash scripts/mine_epdms.sh navtrainval run --logs 1 --frames 2
EPDMS_OUTPUT=data/mining/navtrainval_epdms_pilot \
  python -m drivesta_feedback.epdms.validation data/mining/navtrainval_epdms_pilot --max-logs 1

# A full run uses a different output directory and no pilot limits.
bash scripts/mine_epdms.sh navtrainval run
bash scripts/mine_epdms.sh navtrainval recover --rounds 8
```

## SimScale prerequisites

Set `EPDMS_SIMSCALE_SOURCE` to the complete PDMS SimScale bank (default
`data/trajectories/simscale`). Its manifest and per-scene contracts are supplied
in the PDMS archives. The matching original source logs, maps and adapted
full-horizon actor-tail caches remain external inputs, as described in
`DATA_LAYOUT.md`. `SIMSCALE_OFFICIAL_CACHE_ROOT` and `OPENSCENE_DATA_ROOT` relocate
the historical paths while preserving file contents and provenance checks.

```bash
export EPDMS_SIMSCALE_SOURCE=/path/to/pdms/simscale
EPDMS_OUTPUT=data/mining/simscale_epdms_pilot \
  bash scripts/mine_epdms.sh simscale run --pilot-per-source 1
bash scripts/mine_epdms.sh simscale run
```

EPDMS outputs contain eight components plus explicit EC and qualification fields.
Synthetic clips have no valid adjacent-frame EC; unavailable EC is not replaced
with a perfect score. The inferred actor-tail TTC gate remains an additional
target admission rule. These outputs must not be read as seven-label PDMS
training tensors.

Historical cohort-specific adaptive pilot scripts are not public launchers.
The supported recovery entry point is `recover_navsim_epdms.py`, which retains
transactional publication and independent validation. Changing code or run
contracts requires a new output directory; do not resume an old research job with
the reformatted release sources.
