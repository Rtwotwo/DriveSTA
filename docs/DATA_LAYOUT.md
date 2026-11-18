# Data and cache layout

The trajectory archives contain scored alternatives and their per-scene metadata.
Camera images, original driving logs, maps, and simulator/feature caches are
separate inputs. This distinction keeps the trajectory release readable with
NumPy alone.

```text
DriveSTA/
  weights/
    Trajectories/
      Navtrainval/       # 16 independent tar.gz shards
      SimScale/          # 32 independent tar.gz shards
      Metadata/          # scene manifests and immutable training indexes
    Checkpoints/
      base_proposal.ckpt
      base.ckpt
      scale_proposal.ckpt
      scale.ckpt
    Backbones/
      dinov2/model.safetensors
      drive_jepa/vitl_merge_3dataset_e50.pt
  data/
    trajectories/                  # created by extraction
      navtrainval/scenes/{train,val}/<token>/
      simscale/scenes/train/<token>/
      simscale/training_splits/
    indexes/research/              # immutable model-training indexes
    mining/                        # new generation runs
    navsim_logs/                   # external logs
      trainval/
      test/
      synthetic_reaction_pdm_v1.0-{0,1,2,3,4}/
    sensor_blobs/                  # external images
      trainval/
      trainval_all/                # full 4-history + 10-future video for Refinement
      test/
      synthetic_reaction_pdm_v1.0-{0,1,2,3,4}/
    maps/                          # external nuPlan map files
    navsim_cache/                  # external cache products
      training_cache/<log>/<token>/
      train_metric_cache/
      metric_cache/                # NAVTEST v1
      official_metric_cache_v1/    # real-data online training scorer
      train_metric_synthetic_reaction_pdm_v1.0-{0,1,2,3,4}/
```

Environment variables support data on separate volumes:

| Variable | Meaning |
| --- | --- |
| `OPENSCENE_DATA_ROOT` | Parent of `navsim_logs`, `sensor_blobs`, and default `maps` |
| `NAVSIM_CACHE_ROOT` | Real-data training and evaluation caches |
| `OFFICIAL_PDM_CACHE` | Exact real-data online-scoring caches |
| `DRIVESTA_INDEX_ROOT` | Optional separate immutable research-index root; public aliases still work |
| `DRIVESTA_TRAJECTORY_ROOT` | Parent of extracted `navtrainval` and `simscale` banks |
| `SIMSCALE_SOURCE_CACHE_ROOT` | Original synthetic input caches, if stored separately |
| `SIMSCALE_OFFICIAL_CACHE_ROOT` | Published adapted synthetic caches, preserving source-group and `published/` paths |
| `NUPLAN_MAPS_ROOT` | Optional explicit map directory |
| `NAVSIM_EXP_ROOT` | Output location for new experiments |
| `DRIVESTA_BACKBONE_ROOT` | Optional alternative root for backbone files |
| `PYTHON_BIN` | Python interpreter used by shell entry points |

Set these before sourcing `scripts/environment.sh`. A new synthetic mining run
uses `NAVSIM_CACHE_ROOT` as its synthetic input cache root; point it at the
synthetic cache volume for that mining invocation. Training metadata can use the
separate `SIMSCALE_SOURCE_CACHE_ROOT` above.

The Scale checkpoints were trained with published, adapted full-horizon actor-tail
caches, including repaired scenes. Their cache references are retained in the
training manifest. The original four-second synthetic caches cannot replace those
online-scoring inputs. The release preserves all trajectory data but does not
bundle these external cache files or image/log datasets.
