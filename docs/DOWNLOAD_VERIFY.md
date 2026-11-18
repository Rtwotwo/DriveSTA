# Download, scene lookup, verification and recovery

The release contains **103,288 Navtrainval scenes in 16 independent shards** and
**204,831 SimScale scenes in 32 independent shards**. These are trajectory banks;
images, logs, maps, evaluation caches and EPDMS-mined banks are not included.
Each `.tar.gz` is a complete archive. Do not concatenate shards.
The four trained models are directly usable `.ckpt` files under
`weights/Checkpoints/` and require no extraction.

## Download

Keep the code checkout and its inventories together. Use the release downloader so public filenames and the local checksums match:

```bash
python scripts/download_assets.py --profiles base scale base_proposal scale_proposal
python scripts/download_assets.py --trajectories all
```

Add `--revision HUB_COMMIT` to pin a published Hub revision, or `--plan` to inspect the file selection without downloading. The downloader verifies SHA-256 before installing each asset and handles historical Hub filenames inside its compatibility registry. Do not replace this checkout's scripts/manifests with older copies from an asset-only snapshot.

The verification helpers require Python 3.9 or later and use only the standard library. Reading trajectory arrays additionally requires NumPy. Downloading uses `huggingface_hub`, included in the requirements.

## Inventories

| File | Purpose |
| --- | --- |
| `manifests/scene_to_shard.csv` | One row for every released scene: dataset, split, scene token, archive filename, extraction directory, file count and high-bank status marker |
| `manifests/shard_summary.csv` | Compact 49-row archive table with scene/file counts, sizes, SHA-256 and file-manifest paths |
| `manifests/index.json` | Dataset totals, per-shard scene/file counts, archive sizes and SHA-256, and checksum-manifest locations and SHA-256 |
| `manifests/shards/*.files.jsonl.gz` | Every original extracted file's relative path, byte size and SHA-256, exported from the checksums embedded in its archive |
| `docs/release/archives.json` | Full inventory of the 48 trajectory archives and metadata archive |
| `docs/release/SHA256SUMS` | Checksums of downloadable assets, inventories and verification helpers; run from the download root |

The scene identity is `(dataset, split, token)`; always retain the dataset and
split when joining with other indexes. `COMPLETE.json` and `SHORTFALL.json` in the
CSV describe the original mining result, not download health. A shortfall is an
intentional source-data state and does not indicate a broken archive.

To find the shard containing a scene, replace `SCENE_TOKEN` below:

```bash
python scripts/verify_trajectories.py --root . \
  --dataset navtrainval --token SCENE_TOKEN --locate
```

Use `--dataset simscale` for synthetic scenes. The CSV is also readable directly
with Python's `csv.DictReader` or spreadsheet applications.
When importing into a spreadsheet, set the `token` column to text before import
to preserve leading zeros and long numeric identifiers. `csv.DictReader` and the
provided lookup helper preserve tokens as strings automatically.

## Verify downloads and extract

For a full download, check all listed checksums before extraction:

```bash
sha256sum -c docs/release/SHA256SUMS
python scripts/extract_release.py weights/Trajectories/*/*.tar.gz \
  --destination . --workers 4
```

For a partial download, verify or extract only the selected archive:

```bash
python scripts/extract_release.py \
  weights/Trajectories/Navtrainval/drivesta_navtrainval_001-of-016.tar.gz \
  --verify-only
# Remove --verify-only and add --destination . to extract it.
```

The extractor checks each archive's complete SHA-256 before writing. It refuses
to overwrite existing files. Use an empty destination for first extraction, or
the repair command below when recovering an interrupted extraction. Retain the
archives until the extracted files have passed verification.

## Check extracted files

After a full extraction:

```bash
python scripts/verify_trajectories.py --root . --workers 4
python scripts/verify_trajectories.py --root . --shard drivesta_metadata.tar.gz
```

After a partial extraction, select the corresponding shard or scene:

```bash
python scripts/verify_trajectories.py --root . \
  --shard drivesta_navtrainval_001-of-016.tar.gz
python scripts/verify_trajectories.py --root . \
  --dataset simscale --token SCENE_TOKEN
```

Checks include file presence, byte size and SHA-256. Exit status is `0` only when
all selected files pass. `trajectory_integrity_report.json` lists the precise
broken/missing files and their shards; use `--report PATH` to keep separate
reports. `--quick` checks only existence and size, so it cannot detect corruption
that preserves file size and must not replace the full final check.

## Repair only the affected files

```bash
python scripts/verify_trajectories.py --root . \
  --shard drivesta_navtrainval_001-of-016.tar.gz --repair
# Alternatively target one scene:
python scripts/verify_trajectories.py --root . \
  --dataset simscale --token SCENE_TOKEN --repair
```

Repair verifies the entire local archive first, then replaces only missing or
incorrect files, using temporary files and atomic replacement. Every restored
file is checked against its original SHA-256 and read back. Files that already
pass verification are retained. This also recovers an interrupted extraction
without extracting the remaining 47 shards again.

If an archive is missing or fails its checksum, rerun the downloader for the relevant collection. Already-correct assets are verified and retained. Then repeat the targeted repair command. Keep helpers and inventories from this code release together.

## 中文速查

- `manifests/scene_to_shard.csv` 列出全部 308,119 个场景与分片的对应关系。
- 用 Excel 等表格软件导入时，将 `token` 列设为文本，保留前导零和长编号；核对脚本会自动保留原始字符串。
- 用 `--token 场景TOKEN --locate` 查分片；用 `--shard 分片名` 检查解压文件。
- 默认逐文件检查 SHA-256，能发现同等大小的内容损坏；`--quick` 仅作快速检查。
- 添加 `--repair` 只恢复缺失或损坏的文件，并保留已经正确的文件。
- 压缩包自身损坏时，只需重新下载报告指出的分片，再运行 `--repair`。
- 初次解压用 `extract_release.py`；中断后继续恢复用 `verify_trajectories.py --repair`。
