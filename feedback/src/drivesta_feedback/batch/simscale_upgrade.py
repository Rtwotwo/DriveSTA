"""Explicit sampler upgrades preserve existing accepted data and failed attempts."""

import os
from pathlib import Path
import shutil
import time
from .store import read_json, write_json, lock


def compatible_labels(old, new):
    for name in (
        "scoring_protocol",
        "actor_tail",
        "policy",
        "full_count",
        "high_count",
        "source_counts",
        "filter_root",
    ):
        if old.get(name) != new.get(name):
            raise ValueError("Cannot reuse prior banks: " + name + " changed")
    for name in (
        "batched_official.py",
        "actor_tail.py",
        "filter_targets.py",
        "simscale_source.py",
        "physical_sampling.py",
    ):
        if old["implementation"][name] != new["implementation"][name]:
            raise ValueError("Cannot reuse prior banks: " + name + " changed")


def upgrade_run(output, config, limit):
    output = Path(output)
    with lock(output / "runner"):
        previous = read_json(output / "run.json")
        if previous["limit"] != limit:
            raise ValueError("Upgrade must preserve scene manifest")
        compatible_labels(previous["config"], config)
        archive = output / "previous_attempts" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        archive.mkdir(parents=True, exist_ok=False)
        for name in (
            "run.json",
            "status.json",
            "results.jsonl",
            "shortfall_tokens.txt",
            "error_tokens.txt",
            "PARTIAL.json",
            "COMPLETE.json",
        ):
            if (output / name).exists():
                shutil.copy2(output / name, archive / name)
        moved = []
        reused = []
        for split in (output / "scenes").iterdir():
            if not split.is_dir():
                continue
            for folder in split.iterdir():
                if not folder.is_dir():
                    continue
                result = (
                    read_json(folder / "RESULT.json") if (folder / "RESULT.json").exists() else {}
                )
                if result.get("complete_high") and result.get("complete_full"):
                    reused.append(folder.name)
                else:
                    destination = archive / "scenes" / split.name / folder.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(folder, destination)
                    moved.append(folder.name)
        accepted = previous.get("compatible_digests", []) + [previous["config"]["digest"]]
        current = dict(previous, config=config, compatible_digests=sorted(set(accepted)))
        write_json(
            archive / "upgrade.json",
            dict(
                reused=reused,
                archived=moved,
                previous_digest=previous["config"]["digest"],
                new_digest=config["digest"],
            ),
        )
        write_json(output / "run.json", current)
        return current
