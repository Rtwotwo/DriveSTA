"""Export immutable, token-exact training subsets from published SimScale banks."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def partition(records, excluded, groups):
    tokens = [r["token"] for r in records]
    if len(tokens) != len(set(tokens)):
        raise ValueError("Duplicate scene tokens in source manifest")
    if excluded - set(tokens):
        raise ValueError("Exclusion list contains unknown scene tokens")
    selected = [r for r in records if r["source_group"] in groups]
    good = [r for r in selected if r["token"] not in excluded]
    bad = [r for r in selected if r["token"] in excluded]
    return selected, good, bad


def inspect_published(root, record, excluded):
    token = record["token"]
    folder = root / "scenes" / record["split"] / token
    try:
        full = json.loads((folder / "ALL_SCORES.json").read_text())
        if not (
            full.get("passed") is True
            and full.get("candidates") == 3072
            and full.get("token") == token
        ):
            raise ValueError("Invalid full-bank completion marker")
        for name in ("candidates.npy", "contract.json"):
            if (folder / name).stat().st_size == 0:
                raise ValueError(f"Empty {name}")
        complete = folder / "COMPLETE.json"
        if token in excluded:
            if complete.exists():
                raise ValueError("Excluded token has a COMPLETE marker")
        else:
            high = json.loads(complete.read_text())
            if not (
                high.get("passed") is True
                and high.get("candidates") == 256
                and high.get("format") == "high_score_256"
                and high.get("token") == token
            ):
                raise ValueError("Invalid high-bank completion marker")
            if high.get("minimum_pdms", 95.0) < 95.0:
                raise ValueError("High-bank score below 95")
            if (folder / "high_score.npz").stat().st_size == 0:
                raise ValueError("Empty high-score bank")
        return None
    except Exception as exc:
        return {"token": token, "error": f"{type(exc).__name__}: {exc}"}


def write_tokens(path, records):
    path.write_text("".join(r["token"] + "\n" for r in records))


def export(root, output, source_configs, workers=8, details=None):
    import yaml

    root, output = Path(root).resolve(), Path(output).resolve()
    if root not in output.parents:
        raise ValueError("Snapshot must be inside the canonical dataset root")
    records = [json.loads(s) for s in (root / "scenes.jsonl").read_text().splitlines()]
    exclusions = (root / "shortfall_tokens.txt").read_text().splitlines()
    if len(exclusions) != len(set(exclusions)) or any(not t for t in exclusions):
        raise ValueError("Blank or duplicate exclusions")
    excluded = set(exclusions)
    source_hashes = {
        str(root / n): digest(root / n)
        for n in ("scenes.jsonl", "shortfall_tokens.txt", "status.json")
    }
    subsets = {}
    for name, config_path in source_configs.items():
        config_path = Path(config_path)
        cfg = yaml.safe_load(config_path.read_text())
        sources = cfg["synthetic_sources"]
        groups = [int(s["data_split"].rsplit("-", 1)[1]) for s in sources.values()]
        all_rows, good, bad = partition(records, excluded, groups)
        counts = Counter(r["source"] for r in all_rows)
        for source in sources.values():
            if counts[source["data_split"]] != source["expected_num_samples"]:
                raise ValueError(f"{name}: source count mismatch")
        subsets[name] = (groups, all_rows, good, bad)
        source_hashes[str(config_path)] = digest(config_path)
    if set(r["token"] for r in subsets["simscale205k"][1]) != set(r["token"] for r in records):
        raise ValueError("205k configuration does not cover the canonical manifest")
    output.mkdir(parents=True, exist_ok=False)
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        checks = pool.map(lambda r: inspect_published(root, r, excluded), records)
        for i, error in enumerate(checks, 1):
            if error:
                errors.append(error)
            if i % 10000 == 0:
                print(
                    json.dumps({"checked": i, "total": len(records), "errors": len(errors)}),
                    flush=True,
                )
    (output / "publication_errors.json").write_text(json.dumps(errors, indent=2))
    if errors:
        raise ValueError(f"{len(errors)} publication errors; no training manifest exported")
    for path, value in source_hashes.items():
        if digest(path) != value:
            raise ValueError(f"Source changed during inventory: {path}")
    shutil.copy2(root / "shortfall_tokens.txt", output / "shortfall_tokens.txt")
    if details:
        import csv

        with Path(details).open() as f:
            detail_rows = list(csv.DictReader(f))
        if len(detail_rows) != len(excluded) or {r["token"] for r in detail_rows} != excluded:
            raise ValueError("Per-token details differ from current exclusions")
        shutil.copy2(details, output / "shortfall_details.csv")
    summary = {
        "schema": "simscale-training-splits/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "source_hashes": source_hashes,
        "validation": {
            "published_metadata_and_nonempty_banks": len(records),
            "errors": 0,
            "all_banks_freshly_rescored": False,
        },
        "scoring_protocol": "navsim_v1_full_ttc_actor_tail_v1",
        "cache_resolution": "Read official_cache_path from each scene contract.identity; source_cache_path is not the scoring cache.",
        "training_launched": False,
        "subsets": {},
    }
    for name, (groups, all_rows, good, bad) in subsets.items():
        write_tokens(output / f"{name}_all_tokens.txt", all_rows)
        write_tokens(output / f"{name}_train_tokens.txt", good)
        write_tokens(output / f"{name}_excluded_tokens.txt", bad)
        with (output / f"{name}_train.jsonl").open("w") as f:
            for r in good:
                folder = root / "scenes" / r["split"] / r["token"]
                item = {
                    key: r[key] for key in ("token", "log_name", "split", "source", "source_group")
                }
                item.update(scene_dir=str(folder), contract_path=str(folder / "contract.json"))
                f.write(json.dumps(item, separators=(",", ":")) + "\n")
        summary["subsets"][name] = {
            "source_groups": groups,
            "total": len(all_rows),
            "excluded": len(bad),
            "eligible": len(good),
            "excluded_fraction": len(bad) / len(all_rows),
            "data_size_simscale_only": len(good),
            "by_source": {
                s: {
                    "total": sum(r["source"] == s for r in all_rows),
                    "excluded": sum(r["source"] == s for r in bad),
                    "eligible": sum(r["source"] == s for r in good),
                }
                for s in sorted({r["source"] for r in all_rows})
            },
        }
    summary["file_sha256"] = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["subsets"], indent=2), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config134k", type=Path, required=True)
    p.add_argument("--config205k", type=Path, required=True)
    p.add_argument("--details", type=Path)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    export(
        a.root,
        a.output,
        {"simscale134k": a.config134k, "simscale205k": a.config205k},
        a.workers,
        a.details,
    )


if __name__ == "__main__":
    main()
