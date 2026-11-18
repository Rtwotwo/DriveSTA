"""Remove only a verified recovery run's disposable outputs after publication."""

import argparse
import json
import os
from pathlib import Path
import shutil
import time
from .store import checksum, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "apply"))
    p.add_argument("--project", type=Path, required=True)
    a = p.parse_args()
    project = a.project.resolve()
    run = project / "feedback/runs/recovery548_20261019"
    bank = project / "exp/drivesta_feedback"
    report = bank / "reports/merge523_20261019"
    txn = bank / "replacements/recovery523_20261019_v1"
    verified = json.loads((report / "VALIDATED.json").read_text())
    if not verified["passed"] or verified["replaced_scenes"] != 523 or verified["remaining"] != 25:
        raise ValueError("Publication readback must pass before cleanup")
    if json.loads((bank / "status.json").read_text())["completed_scenes"] != 103263:
        raise ValueError("Production status changed")
    if (
        checksum(project / "exp/chainflow_vla/drivesta/chainflow_feedback/targets_v1/index.json")
        != verified["active_training_index_digest"]
    ):
        raise ValueError("Active training index changed")
    run.resolve().relative_to((project / "feedback/runs").resolve())
    if run.is_symlink() or bank.is_symlink() or txn.is_symlink():
        raise ValueError("Cleanup roots must be real directories")
    plan_path = report / "cleanup_plan.json"
    if a.action == "plan":
        # Keep small root-level evidence; archive run configurations/summaries and
        # the two route figures before deleting candidate trials and caches.
        entries = [x for x in run.iterdir() if x.is_dir() or x.name.endswith(".tar.gz")]
        entries.append(txn / "backup")
        if (txn / "staged").exists():
            entries.append(txn / "staged")
        archive = report / "experiment_summaries"
        archive.mkdir(exist_ok=True)
        for entry in entries:
            if entry.parent == run and entry.is_dir() and not entry.is_symlink():
                for name in ("config.json", "summary.json", "COMPLETE.json"):
                    source = entry / name
                    if source.is_file():
                        destination = archive / entry.name / name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, destination)
        figures = report / "route_figures"
        figures.mkdir(exist_ok=True)
        for source in (run / "route_anchor2_reference/scenes").glob("*.png"):
            shutil.copy2(source, figures / source.name)
        before = json.loads((txn / "root_before/PARTIAL.json").read_text())
        (report / "original_shortfall548_tokens.txt").write_text(
            "".join(x["token"] + "\n" for x in before["shortfalls"])
        )
        readback = json.loads((run / "readback.json").read_text())
        plan = json.loads((txn / "plan.json").read_text())
        identity = {x["record"]["token"]: x["record"] for x in plan["scenes"]}
        published = []
        for row in readback["records"]:
            token = row["token"]
            record = identity[token]
            folder = bank / "scenes" / record["split"] / token
            if (
                not (folder / "COMPLETE.json").is_file()
                or not (folder / "ALL_SCORES.json").is_file()
            ):
                raise ValueError("Published data is missing")
            published.append(
                dict(row, directory=str(folder), former_trial_directory=row["directory"])
            )
        write_json(
            report / "published523.json", dict(published_to_live_bank=True, records=published)
        )
        inventory = []
        for entry in entries:
            parent = run if entry.parent == run else txn
            entry.resolve().relative_to(parent.resolve())
            if entry.is_symlink():
                raise ValueError("Unexpected cleanup symlink")
            files = (
                [entry]
                if entry.is_file()
                else [
                    Path(base) / n
                    for base, _, names in os.walk(entry, followlinks=False)
                    for n in names
                ]
            )
            inventory.append(
                dict(path=str(entry), files=len(files), bytes=sum(f.lstat().st_size for f in files))
            )
        write_json(
            plan_path,
            dict(
                created_unix=time.time(),
                verified_digest=checksum(report / "VALIDATED.json"),
                entries=inventory,
                scope="trial directories, duplicate review bundles and verified replacement backups",
            ),
        )
        print(
            json.dumps(dict(paths=len(inventory), bytes=sum(x["bytes"] for x in inventory))),
            flush=True,
        )
    else:
        plan = json.loads(plan_path.read_text())
        if checksum(report / "VALIDATED.json") != plan["verified_digest"]:
            raise ValueError("Verification changed")
        allowed = {str(x) for x in run.iterdir() if x.is_dir() or x.name.endswith(".tar.gz")}
        allowed |= {str(txn / "backup"), str(txn / "staged")}
        for item in plan["entries"]:
            path = Path(item["path"])
            parent = run if path.parent == run else txn
            path.resolve().relative_to(parent.resolve())
            if str(path) not in allowed and path.exists():
                raise ValueError("Cleanup target changed")
            if path.is_symlink():
                raise ValueError("Cleanup target became a symlink")
        removed = []
        for item in plan["entries"]:
            path = Path(item["path"])
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            removed.append(item)
            write_json(
                report / "cleanup_progress.json",
                dict(removed=len(removed), total=len(plan["entries"])),
            )
        for row in json.loads((report / "published523.json").read_text())["records"]:
            if not all(
                (Path(row["directory"]) / n).is_file()
                for n in ("candidates.npy", "high_score.npz", "COMPLETE.json", "ALL_SCORES.json")
            ):
                raise ValueError("Production data missing after cleanup")
        result = dict(
            passed=True,
            completed_unix=time.time(),
            removed_paths=len(removed),
            removed_files=sum(x["files"] for x in removed),
            removed_file_bytes=sum(x["bytes"] for x in removed),
            production_data_preserved=True,
            token_lists_and_small_reports_preserved=True,
            training_and_official_caches_preserved=True,
        )
        write_json(report / "CLEANUP.json", result)
        brief = json.loads((run / "brief.json").read_text())
        brief.update(
            published_to_live_bank=True,
            canonical_shortfall_count=25,
            production_high_scenes=103263,
            intermediate_outputs_removed=True,
            publication_record=str(report / "published523.json"),
        )
        write_json(run / "brief.json", brief)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
