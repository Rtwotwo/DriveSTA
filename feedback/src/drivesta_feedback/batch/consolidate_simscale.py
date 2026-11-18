"""Verify published SimScale banks, retain receipts, and remove finished trial dirs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def descendants(value, key=""):
    if isinstance(value, dict):
        for name, child in value.items():
            yield from descendants(child, key + "." + name)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from descendants(child, key + "." + str(i))
    elif isinstance(value, str):
        yield key, value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "apply"))
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve(strict=True)
    exp = project / "exp"
    bank = exp / "drivesta_feedback_simscale"
    report = bank / "reports/consolidation_20261021"
    report.mkdir(parents=True, exist_ok=True)
    from .store import checksum, write_json
    import psutil

    def guard(path):
        path = Path(path)
        if path.parent != exp or not path.name.startswith("drivesta_feedback_simscale_"):
            raise ValueError("Cleanup target outside the explicit sibling prefix: " + str(path))
        if path.is_symlink() or path.resolve() != path or path == bank:
            raise ValueError("Symlink or noncanonical cleanup target: " + str(path))
        if bank.is_symlink() or bank.resolve() != bank or exp.resolve() != exp:
            raise ValueError("Production roots are not canonical directories")
        return path

    def idle(paths):
        busy = []
        prefixes = [str(p) for p in paths]
        for proc in psutil.process_iter(["pid", "cmdline"]):
            if proc.pid == os.getpid():
                continue
            command = " ".join(proc.info["cmdline"] or [])
            if "recover_simscale.py" in command or any(
                p + "/" in command or command.endswith(p) for p in prefixes
            ):
                busy.append(proc.info)
        if busy:
            raise RuntimeError("A process still uses the trial outputs: " + json.dumps(busy))

    if args.action == "plan":
        paths = sorted(guard(p) for p in exp.glob("drivesta_feedback_simscale_*"))
        if not paths:
            raise ValueError("No trial directories to consolidate")
        idle(paths)
        status = read(bank / "status.json")
        successful = set()
        entries = []
        for path in paths:
            state = read(path / "status.json")
            if state["status"] != "finished" or state.get("errors"):
                raise ValueError("Trial run is incomplete or has errors: " + str(path))
            rows = {
                r["token"]: r
                for r in map(json.loads, (path / "results.jsonl").read_text().splitlines())
            }
            successful.update(t for t, r in rows.items() if r.get("recovered"))
            target = report / "source_runs" / path.name
            target.mkdir(parents=True, exist_ok=True)
            receipts = {}
            for p in path.iterdir():
                if p.is_file() and p.suffix in (".json", ".jsonl", ".txt"):
                    shutil.copy2(p, target / p.name)
                    receipts[p.name] = checksum(p)
                    assert checksum(target / p.name) == receipts[p.name]
            # Exact file-size totals are not a cleanup precondition. Avoid a slow
            # recursive metadata pass over this network filesystem. rmtree does
            # not follow child symlinks; the deletion root is separately guarded.
            entries.append(dict(path=str(path), root_receipts=receipts))
            print(json.dumps(dict(inventoried=path.name, status=state["status"])), flush=True)
        expected_missing = set((bank / "shortfall_tokens.txt").read_text().split())
        if len(expected_missing) != status["shortfall"] or successful & expected_missing:
            raise ValueError("Recovered scene missing from the production high bank")
        assert len(successful) == status["recovered"] == 4373
        prefixes = [str(p) + "/" for p in paths]

        def verify(token):
            import numpy as np
            from .high_score_bank import read_scene
            from .physical_sampling import physical_mask, score_mask
            from .filter_targets import TargetPolicy

            try:
                folder = bank / "scenes/train" / token
                result = read(folder / "RESULT.json")
                for name, digest in result["files"].items():
                    assert checksum(folder / name) == digest, str(folder / name)
                identity, high = read_scene(folder, verify=True)
                contract = read(folder / "contract.json")
                c = contract["protocol"]["constructor"]
                policy = TargetPolicy(**c["policy"])
                full = np.load(folder / "candidates.npy", allow_pickle=False)
                assert len(full) == 3072 and len(high["labels"]) == 256
                q = high["query_indices"]
                for key in ("keys", "trajectories", "labels"):
                    assert np.array_equal(full[key][q], high[key]), key
                assert score_mask(high["labels"], policy).all()
                valid, _ = physical_mask(
                    full["trajectories"],
                    np.asarray(c["expert"], np.float32),
                    policy,
                    c["initial_velocity"],
                    c["initial_acceleration"],
                )
                assert valid.all()
                with np.load(folder / "construction.npz", allow_pickle=False) as z:
                    assert (
                        z["eligible"][q].all()
                        and score_mask(z["alternative_labels"][q], policy).all()
                    )
                    assert (
                        z["tracking"][q]
                        <= [
                            policy.max_tracking_ade_m,
                            policy.max_tracking_error_m,
                            policy.max_tracking_heading_error_rad,
                        ]
                    ).all()
                with np.load(folder / "extra_queries.npz", allow_pickle=False) as z:
                    ids = np.r_[full["keys"][:, 1], z["source_query_indices"]]
                    assert len(ids) == result["scored_queries"]
                    assert np.array_equal(np.sort(ids), np.arange(len(ids)))
                    assert len(z["labels"]) == len(z["source_query_indices"])
                dependencies = []
                for key, value in descendants(contract):
                    if any(value.startswith(p) for p in prefixes):
                        assert key == ".recovery.query_archive", (key, value)
                    if (
                        key.startswith(".identity")
                        and key.endswith("_path")
                        and value.startswith("/")
                    ):
                        assert Path(value).exists(), (key, value)
                        dependencies.append(value)
                source = Path(contract["recovery"]["query_archive"])
                assert any(str(source).startswith(p) for p in prefixes)
                if (source / "checkpoint.npz").exists():
                    with np.load(source / "checkpoint.npz", allow_pickle=False) as z:
                        for key in ("trajectories", "labels"):
                            assert np.array_equal(full[key], z[key][full["keys"][:, 1]])
                else:
                    with np.load(source / "relabeled.npz", allow_pickle=False) as z:
                        for key in ("trajectories", "labels"):
                            assert np.array_equal(full[key], z[key][full["keys"][:, 1]])
                receipt = result["official_verification"]
                assert receipt["official_max_error"] == 0 and receipt["verified_high"] == 256
                backup = Path(contract["recovery"]["original_backup"])
                assert (
                    backup.is_relative_to(bank / "previous_attempts")
                    and (backup / "candidates.npy").is_file()
                )
                before = report / "metadata_before" / token
                before.mkdir(parents=True, exist_ok=True)
                for name in ("contract.json", "RESULT.json"):
                    shutil.copy2(folder / name, before / name)
                    assert checksum(folder / name) == checksum(before / name)
                return dict(
                    token=token,
                    passed=True,
                    result_digest=checksum(folder / "RESULT.json"),
                    array_digests={
                        k: v for k, v in result["files"].items() if k.endswith((".npz", ".npy"))
                    },
                    before_contract_digest=checksum(folder / "contract.json"),
                    query_archive=str(source),
                    original_backup=str(backup),
                    cache_dependencies=dependencies,
                    minimum_pdms=float(high["labels"][:, -1].min() * 100),
                )
            except Exception:
                return dict(token=token, error=traceback.format_exc())

        checked = []
        with ThreadPoolExecutor(16) as pool:
            for i, row in enumerate(pool.map(verify, sorted(successful)), 1):
                checked.append(row)
                if i % 256 == 0:
                    print(json.dumps(dict(verified=i, total=len(successful))), flush=True)
        failures = [row for row in checked if "error" in row]
        write_json(
            report / "validation.json",
            dict(passed=not failures, scenes=len(checked), failures=failures, rows=checked),
        )
        if failures:
            raise ValueError(
                "Published bank checks failed; no cleanup permitted: " + json.dumps(failures[:3])
            )
        for name in ("status.json", "PARTIAL.json", "shortfall_tokens.txt"):
            shutil.copy2(bank / name, report / ("before_" + name))
        plan = dict(
            version=1,
            created_unix=time.time(),
            main=str(bank),
            entries=entries,
            recovered_scenes=len(successful),
            status_before=status,
            validation_digest=checksum(report / "validation.json"),
            shortfall_digest=checksum(bank / "shortfall_tokens.txt"),
            scope="Remove explicitly requested sibling trial directories after canonical 3072/256 publication checks. "
            "Keep all production data, extra_queries, original scene backups, official and repaired caches. "
            "Preserve run summaries; discarded trial queries/checkpoints are not needed for training.",
        )
        write_json(report / "plan.json", plan)
        print(
            json.dumps(
                dict(
                    plan=str(report / "plan.json"),
                    targets=len(entries),
                    verified_scenes=len(checked),
                )
            ),
            flush=True,
        )
        return

    plan = read(report / "plan.json")
    if checksum(report / "validation.json") != plan["validation_digest"]:
        raise ValueError("Verification receipt changed")
    validation = read(report / "validation.json")
    assert validation["passed"] and not validation["failures"]
    paths = [guard(Path(e["path"])) for e in plan["entries"]]
    idle(paths)
    assert checksum(bank / "shortfall_tokens.txt") == plan["shortfall_digest"]
    current = read(bank / "status.json")
    for key in ("full3072", "high256", "shortfall", "recovered"):
        assert current[key] == plan["status_before"][key]

    # Preserve each original metadata file, then update only source-location fields.
    # Array files and all scoring/physical contracts remain byte-identical.
    def update_metadata(row):
        token = row["token"]
        folder = bank / "scenes/train" / token
        receipt = read(folder / "RESULT.json")
        contract = read(folder / "contract.json")
        for name, digest in row["array_digests"].items():
            assert checksum(folder / name) == digest, str(folder / name)
        if (
            contract["recovery"].get("query_archive_layout")
            != "published_candidates_and_extra_queries_v1"
        ):
            assert checksum(folder / "RESULT.json") == row["result_digest"]
            assert checksum(folder / "contract.json") == row["before_contract_digest"]
            contract["recovery"]["former_query_archive"] = contract["recovery"]["query_archive"]
            contract["recovery"]["query_archive"] = str(folder)
            contract["recovery"][
                "query_archive_layout"
            ] = "published_candidates_and_extra_queries_v1"
            contract["recovery"]["source_index_fields"] = [
                "candidates.npy:keys[:,1]",
                "extra_queries.npz:source_query_indices",
            ]
            contract["recovery"]["consolidation_receipt"] = str(report / "plan.json")
            write_json(folder / "contract.json", contract)
        receipt["files"]["contract.json"] = checksum(folder / "contract.json")
        write_json(folder / "RESULT.json", receipt)

    with ThreadPoolExecutor(16) as pool:
        for i, _ in enumerate(pool.map(update_metadata, validation["rows"]), 1):
            if i % 512 == 0:
                print(
                    json.dumps(dict(metadata_updated=i, total=len(validation["rows"]))), flush=True
                )
    for name in ("status.json", "PARTIAL.json"):
        data = read(bank / name)
        if data.get("recovery_output") and data["recovery_output"] != str(bank):
            data["former_recovery_output"] = data["recovery_output"]
            data["recovery_output"] = str(bank)
        data["consolidation_report"] = str(report)
        write_json(bank / name, data)
    # Final reviewable gate before any recursive deletion. No symlinks or live jobs.
    idle(paths)
    for entry, path in zip(plan["entries"], paths):
        guard(path)
        if path.exists():
            for name, digest in entry["root_receipts"].items():
                if (path / name).exists():
                    assert checksum(path / name) == digest, str(path / name)
                elif not (report / "cleanup_progress.json").exists():
                    raise ValueError(
                        "Trial receipt missing before first cleanup: " + str(path / name)
                    )
                assert checksum(report / "source_runs" / path.name / name) == digest

    def published_markers(row):
        folder = bank / "scenes/train" / row["token"]
        assert all(
            (folder / name).is_file()
            for name in ("candidates.npy", "high_score.npz", "COMPLETE.json", "ALL_SCORES.json")
        )

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(published_markers, validation["rows"]))
    removed = []

    def remove_entry(entry):
        path = guard(Path(entry["path"]))
        if path.exists():
            # Query archives contain thousands of independent scene directories;
            # parallelize those network-filesystem operations within this root.
            queries = path / "queries"
            if queries.is_dir() and not queries.is_symlink():

                def remove_query(child):
                    if child.parent != queries:
                        raise ValueError("Query path escaped its cleanup root")
                    if child.is_symlink() or not child.is_dir():
                        child.unlink()
                    else:
                        shutil.rmtree(child)

                with ThreadPoolExecutor(16) as pool:
                    list(pool.map(remove_query, list(queries.iterdir())))
            shutil.rmtree(path)
        return entry

    with ThreadPoolExecutor(4) as pool:
        futures = [pool.submit(remove_entry, entry) for entry in plan["entries"]]
        for future in as_completed(futures):
            entry = future.result()
            removed.append(entry)
            write_json(
                report / "cleanup_progress.json",
                dict(removed=len(removed), total=len(paths), last_removed=entry["path"]),
            )
            print(
                json.dumps(
                    dict(removed=Path(entry["path"]).name, completed=len(removed), total=len(paths))
                ),
                flush=True,
            )
    if list(exp.glob("drivesta_feedback_simscale_*")):
        raise RuntimeError("Unexpected sibling trial directory remains")
    from .high_score_bank import read_scene, collate_high_score
    from .retained_bank import read_full_scene
    import numpy as np

    def readback(item):
        i, row = item
        folder = bank / "scenes/train" / row["token"]
        identity, high = read_scene(folder, verify=True)
        _, full = read_full_scene(folder, verify=True)
        for key in ("trajectories", "labels"):
            assert np.array_equal(full[key][high["query_indices"]], high[key])
        assert Path(identity["official_cache_path"]).is_file()
        if i <= 4:
            assert tuple(collate_high_score([(identity, high)])["scorer_trajectories"].shape) == (
                1,
                256,
                8,
                3,
            )
            assert tuple(collate_high_score([(identity, full)])["scorer_trajectories"].shape) == (
                1,
                3072,
                8,
                3,
            )

    with ThreadPoolExecutor(16) as pool:
        for i, _ in enumerate(pool.map(readback, enumerate(validation["rows"], 1)), 1):
            if i % 512 == 0:
                print(
                    json.dumps(dict(post_cleanup_readback=i, total=len(validation["rows"]))),
                    flush=True,
                )
    assert checksum(bank / "shortfall_tokens.txt") == plan["shortfall_digest"]
    final = dict(
        passed=True,
        completed_unix=time.time(),
        removed_directories=len(removed),
        removed_files=None,
        removed_logical_bytes=None,
        verified_published_scenes=len(validation["rows"]),
        main=str(bank),
        full3072=current["full3072"],
        high256=current["high256"],
        shortfall=current["shortfall"],
        raw_trajectory_and_label_files_unchanged=True,
        original_scene_backups_preserved=True,
        official_and_repaired_caches_preserved=True,
        downstream_readback=True,
    )
    write_json(report / "COMPLETE.json", final)
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
