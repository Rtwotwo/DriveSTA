# Prepare DriveSTA assets and driving data

## 1. Download the required profile

Run from the code checkout. The downloader reads this release's SHA-256 inventory, resolves a fixed Hugging Face revision for the entire operation, verifies each download and writes canonical DriveSTA filenames. It also supports existing Hub filenames through the compatibility registry; do not rename or edit checkpoint tensors yourself.

```bash
# Inference only: choose base, scale, base_proposal or scale_proposal.
python scripts/download_assets.py --profiles base
# All four checkpoints:
python scripts/download_assets.py --profiles base scale base_proposal scale_proposal
# Preview selected files without a network transfer:
python scripts/download_assets.py --profiles base --plan
```

If `weights/Checkpoints/` is already populated in a server release, this step is unnecessary. Inference from a complete checkpoint does not require separate DINOv2 or JEPA pretraining files.

## 2. Prepare external observations

Obtain NAVSIM/OpenScene logs, sensor images and nuPlan maps following their dataset access terms. These external assets are not included in the trajectory banks.

```text
$OPENSCENE_DATA_ROOT/
  navsim_logs/trainval/
  navsim_logs/test/
  sensor_blobs/trainval/
  sensor_blobs/test/
  maps/                         # or set NUPLAN_MAPS_ROOT separately
```

The proposal planner reads current multi-camera observations and ego history. Temporal refinement also needs the preceding front-camera observation. Refinement training requires the full historical/future expert video referenced by the supplied history and future indexes, including `sensor_blobs/trainval_all/`. Single-scene inference does not consume future labels or a metric cache.

## 3. Prepare trajectory supervision for training

```bash
# Base training data and shared immutable indexes:
python scripts/download_assets.py --trajectories navtrainval
# Also required for Scale:
python scripts/download_assets.py --trajectories simscale
python scripts/extract_release.py weights/Trajectories/*/*.tar.gz \
  --destination . --workers 4
python scripts/verify_trajectories.py --root . --workers 4
```

Each of the 48 trajectory shards and the one metadata shard is an independent gzip-compressed tar archive. Never concatenate shards. Extraction verifies the archive hash, rejects unsafe members and refuses to overwrite existing files. Use `verify_trajectories.py --repair` for interrupted extraction, as explained in [DOWNLOAD_VERIFY.md](DOWNLOAD_VERIFY.md).

Metadata extraction also creates stable public index aliases under `data/indexes/drivesta/`: `proposals.json`, `mixed.json`, `history.json`, `future.json` and `synthetic_video.json`. Their contents are copied byte-for-byte from the archived indexes. Historical internal paths are translated by the IO compatibility layer without changing tokens, labels, order or scientific checksums.

The archives total about 135.3 GiB compressed and 273.8 GiB unpacked. Allow additional space for checkpoints, external images, source caches and experiment outputs. The full released bank contains 308,119 scenes; use the supplied indexes for the smaller fixed Base/Scale training mixtures.

## 4. Prepare caches and initialization backbones

Training requires feature/target caches and official online-scoring caches. Set `NAVSIM_CACHE_ROOT`, `OFFICIAL_PDM_CACHE` and, for Scale, `SIMSCALE_SOURCE_CACHE_ROOT` / `SIMSCALE_OFFICIAL_CACHE_ROOT` as described in [DATA_LAYOUT.md](DATA_LAYOUT.md). The adapted synthetic actor-tail caches used by the fixed Scale recipe cannot be replaced with shorter source caches. They are external prerequisites, not included in the trajectory archives.

For proposal training from scratch, place the DINOv2 small image encoder weights at `weights/Backbones/dinov2/model.safetensors`:

```bash
hf download timm/vit_small_patch14_reg4_dinov2.lvd142m model.safetensors \
  --local-dir weights/Backbones/dinov2
```

Refinement training initialized from a proposal checkpoint also needs the pretrained encoder/predictor/EMA artifact `vitl_merge_3dataset_e50.pt` from [Drive-JEPA](https://github.com/linhanwang/Drive-JEPA). Follow that project's pretrained-weight instructions and place it at `weights/Backbones/drive_jepa/vitl_merge_3dataset_e50.pt`. This server release already includes both initialization artifacts. Their expected hashes are in `docs/release/files.json`; follow upstream asset licensing.

## 5. Validate the training data before optimization

```bash
bash scripts/training/train_drivesta.sh base_proposal preflight_only=true
bash scripts/training/train_drivesta.sh base preflight_only=true
# For the full Scale inputs:
bash scripts/training/train_drivesta.sh scale_proposal preflight_only=true
bash scripts/training/train_drivesta.sh scale preflight_only=true
```

These checks enforce scene membership/counts and profile-specific supervision requirements. They do not replace a complete training run. Continue with [TRAINING.md](TRAINING.md).
