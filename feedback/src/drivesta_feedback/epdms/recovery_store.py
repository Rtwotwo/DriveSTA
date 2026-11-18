"""Journalled in-place EPDMS publication, with immutable previous-frame context."""

from pathlib import Path
import json
import shutil
import time
import os
import numpy as np
from .dataset import (
    read_scene,
    atomic_npy,
    atomic_npz,
    cache_for,
    FastArchiveScorer,
    admission,
    ec_pair,
    score_from_components,
)
from .sampling import unique_ranked
from ..batch.store import write_json, checksum

REVISION = "recovery_20261028"


def workspace(root):
    return Path(root) / "work" / REVISION


def snapshot_original(root):
    target = workspace(root) / "original_summary.json"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".preparing.json")
        shutil.copy2(Path(root) / "summary.json", tmp)
        value = json.loads(tmp.read_text())
        if (
            not value["complete"]
            or value["processed"] != 103288
            or value["high256"] != 71723
            or value["errors"]
        ):
            raise ValueError("Unexpected original corpus snapshot")
        with tmp.open("rb") as stream:
            os.fsync(stream.fileno())
        tmp.replace(target)
    return target


def original_folder(root, split, token):
    saved = workspace(root) / "original" / split / token
    return saved if (saved / "summary.json").exists() else Path(root) / "scenes" / split / token


def preserve(root, split, token):
    source = original_folder(root, split, token)
    target = workspace(root) / "original" / split / token
    if source == target:
        read_scene(target)
        return target
    read_scene(source)
    stage = target.with_name(token + ".preparing")
    if stage.exists():
        # Only a private, unpublished interrupted snapshot is cleared.
        if stage.resolve().parent != target.resolve().parent:
            raise ValueError("Unsafe temporary path")
        shutil.rmtree(stage)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, stage)
    read_scene(stage)
    stage.rename(target)
    return target


def make_stage(root, source, stage):
    """Rebuild only an unpublished private stage, including interrupted copies."""
    stage = Path(stage)
    base = workspace(root).resolve()
    for p in (stage, stage.with_name(stage.name + ".preparing")):
        if base not in p.resolve().parents:
            raise ValueError("Stage is outside recovery workspace")
        if p.exists():
            shutil.rmtree(p)
    temporary = stage.with_name(stage.name + ".preparing")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, temporary)
    read_scene(temporary)
    temporary.rename(stage)
    return stage


def install(root, staged, split, token, operation):
    """Two renames with a durable journal; storage does not support RENAME_EXCHANGE."""
    root = Path(root)
    staged = Path(staged)
    dest = root / "scenes" / split / token
    txn = workspace(root) / "transactions" / split / (token + "_" + operation)
    txn.mkdir(parents=True, exist_ok=True)
    plan = txn / "plan.json"
    done = txn / "DONE.json"
    old = txn / "previous"
    if done.exists():
        info, _, _ = read_scene(dest)
        receipt = json.loads(done.read_text())
        # A later recovery may supersede a metadata-only context publication.
        return info
    if not plan.exists():
        info, _, _ = read_scene(staged)
        write_json(
            plan,
            dict(
                target=str(dest),
                stage=str(staged),
                summary_sha256=checksum(staged / "summary.json"),
            ),
        )
    record = json.loads(plan.read_text())
    if Path(record["stage"]) != staged or Path(record["target"]) != dest:
        raise ValueError("Transaction identity changed")
    if dest.exists() and checksum(dest / "summary.json") == record["summary_sha256"]:
        read_scene(dest)
        write_json(done, dict(committed=True, time=time.time()))
        return read_scene(dest)[0]
    if not staged.exists():
        raise ValueError("Missing prepared transaction")
    read_scene(staged)
    if checksum(staged / "summary.json") != record["summary_sha256"]:
        raise ValueError("Changed prepared transaction")
    if dest.exists():
        if old.exists():
            raise ValueError("Ambiguous interrupted transaction")
        dest.rename(old)
    elif not old.exists():
        raise ValueError("Missing destination and rollback version")
    staged.rename(dest)
    read_scene(dest)
    write_json(done, dict(committed=True, time=time.time()))
    return read_scene(dest)[0]


def resume_transactions(root):
    for p in (workspace(root) / "transactions").glob("*/*/plan.json"):
        if (p.parent / "DONE.json").exists():
            continue
        r = json.loads(p.read_text())
        dest = Path(r["target"])
        split = dest.parent.name
        token = dest.name
        suffix = p.parent.name[len(token) + 1 :]
        install(root, Path(r["stage"]), split, token, suffix)


def freeze_successor(root, row, next_row):
    if next_row is None:
        return
    folder = Path(root) / "scenes" / next_row["split"] / next_row["token"]
    info, _, _ = read_scene(folder)
    if info["previous_token"] != row["token"]:
        return
    previous = preserve(root, row["split"], row["token"])
    preserve(root, next_row["split"], next_row["token"])
    c = json.loads((folder / "contract.json").read_text())
    relative = str(previous.relative_to(root))
    if c.get("previous_context") == relative:
        return
    stage = workspace(root) / "staged_context" / next_row["split"] / next_row["token"]
    make_stage(root, folder, stage)
    c.update(
        previous_context=relative, previous_context_sha256=checksum(previous / "candidates.npy")
    )
    write_json(stage / "contract.json", c)
    info["files"]["contract.json"] = checksum(stage / "contract.json")
    write_json(stage / "summary.json", info)
    install(root, stage, next_row["split"], next_row["token"], "context")


def verify_result(root, row, work, independent_count=8):
    """Re-score every selected current plan; independently replay a fixed subset."""
    from .scoring import (
        ArchiveScorer,
        pdm_score,
        Trajectory,
        OUTPUT_SAMPLING,
        SAMPLING,
        component_rows,
    )

    work = Path(work)
    bank = np.load(work / "best.npy", allow_pickle=False)
    ids = unique_ranked(bank["trajectories"], bank["epdms"], bank["admissible"])
    cache, _ = cache_for(row)
    engine = FastArchiveScorer(cache)
    paths = bank["trajectories"][ids]
    raw, comp, states = engine.score(paths, return_states=True)
    previous = None
    prev_cache = None
    source = original_folder(root, row["split"], row["token"])
    info, _, _ = read_scene(source)
    if info["previous_token"]:
        prevfolder = original_folder(root, row["split"], info["previous_token"])
        pi, pb, _ = read_scene(prevfolder)
        prev_cache, _ = cache_for(pi)
        previous = FastArchiveScorer(prev_cache).simulate(pb["trajectories"][ids])
        gap = info["start_time"] - pi["start_time"]
        ec = ec_pair(states, previous, gap)
        scores = score_from_components(comp, ec)
    else:
        ec = np.full(len(ids), np.nan)
        scores = score_from_components(comp)
    errors = []
    for k, value in [("components_raw", raw), ("components", comp), ("ec", ec), ("epdms", scores)]:
        np.testing.assert_allclose(value, bank[k][ids], atol=1e-10, rtol=0, equal_nan=True)
    ok, phys, track, space, tracking, funnel = admission(
        cache, paths, raw, states, scores, cache.human_trajectory.poses
    )
    if not ok.all():
        raise ValueError("Selected targets failed fresh safety validation")
    official = ArchiveScorer(cache)
    prev_official = ArchiveScorer(prev_cache) if prev_cache else None
    selected = np.unique(np.linspace(0, len(ids) - 1, min(independent_count, len(ids)), dtype=int))
    maximum = 0.0
    for j in selected:
        result, state = pdm_score(
            cache,
            Trajectory(paths[j].astype(float), OUTPUT_SAMPLING),
            SAMPLING,
            official.sim,
            official.official,
            official.traffic,
        )
        c = component_rows([result])
        maximum = max(maximum, float(abs(c - comp[j]).max()))
        if previous is not None:
            _, ps = pdm_score(
                prev_cache,
                Trajectory(pb["trajectories"][ids[j]].astype(float), OUTPUT_SAMPLING),
                SAMPLING,
                prev_official.sim,
                prev_official.official,
                prev_official.traffic,
            )
            # Official one-stage aggregation pairs these same simulated states.
            e = ec_pair(state[None], ps[None], gap)
            np.testing.assert_equal(e, ec[j : j + 1])
            v = score_from_components(c, e)
        else:
            v = score_from_components(c)
        maximum = max(maximum, float(abs(v[0] - scores[j])))
    if maximum > 1e-10:
        raise ValueError("Official independent replay disagrees")
    audit = dict(
        selected_count=len(ids),
        all_selected_replayed=True,
        independent_current_calls=len(selected),
        independent_previous_calls=len(selected) if previous is not None else 0,
        max_official_error=maximum,
        minimum_epdms=float(scores.min() * 100) if len(ids) else None,
        safety_passed=bool(ok.all()),
        best_file_sha256=checksum(work / "best.npy"),
    )
    atomic_npz(work / "verified_targets.npz", query_indices=ids, tracking_errors=tracking)
    write_json(work / "validation.json", audit)
    return audit


def publish(root, row, next_row, work):
    root = Path(root)
    work = Path(work)
    result = json.loads((work / "result.json").read_text())
    audit = json.loads((work / "validation.json").read_text())
    if not result["recovered"] or audit["selected_count"] != 256:
        raise ValueError("Only complete recovered scenes are published")
    if checksum(work / "best.npy") != audit["best_file_sha256"]:
        raise ValueError("Changed audited bank")
    dest = root / "scenes" / row["split"] / row["token"]
    current, _, _ = read_scene(dest)
    if current.get("recovery_revision") == REVISION:
        return current
    old = preserve(root, row["split"], row["token"])
    info, _, _ = read_scene(old)
    if checksum(old / "candidates.npy") != result["source_digest"]:
        raise ValueError("Original scene changed")
    # Preserve the original successor label's predecessor BEFORE this replacement.
    freeze_successor(root, row, next_row)
    contract = json.loads((old / "contract.json").read_text())
    if info["previous_token"]:
        prev = preserve(root, row["split"], info["previous_token"])
        contract.update(
            previous_context=str(prev.relative_to(root)),
            previous_context_sha256=checksum(prev / "candidates.npy"),
        )
    stage = workspace(root) / "staged_recovery" / row["split"] / row["token"]
    make_stage(root, old, stage)
    bank = np.load(work / "best.npy", allow_pickle=False)
    with np.load(work / "verified_targets.npz") as z:
        ids = z["query_indices"]
        tracking = z["tracking_errors"]
    atomic_npy(stage / "candidates.npy", bank)
    atomic_npz(
        stage / "high_score.npz",
        query_indices=ids,
        **{
            k: bank[k][ids]
            for k in ("trajectories", "components_raw", "components", "ec", "epdms", "family_id")
        },
        previous_family_id=ids if info["previous_token"] else np.full(len(ids), -1),
        tracking_errors=tracking,
    )
    contract.update(
        recovery_revision=REVISION,
        recovery_queries=str(work.relative_to(root)),
        pairing="same original family slot with immutable previous-context version; no post-score rematching",
        original_scene=str(old.relative_to(root)),
    )
    write_json(stage / "contract.json", contract)
    info.update(
        recovery_revision=REVISION,
        selected_count=256,
        min_selected_epdms=float(bank["epdms"][ids].min() * 100),
        best_epdms=float(bank["epdms"].max() * 100),
        recovery_queries=result["queries"],
        recovery_audit=audit,
        ec_pass_count=int(np.nansum(bank["ec"])) if info["ec_defined"] else None,
        median_nearest_target_ade_m=float(
            np.median(
                (
                    np.linalg.norm(
                        bank["trajectories"][ids, None, :, :2]
                        - bank["trajectories"][None, ids, :, :2],
                        axis=-1,
                    ).mean(-1)
                    + np.eye(256) * 1e10
                ).min(1)
            )
        ),
    )
    raw = bank["components_raw"]
    safe = (raw[:, [0, 1, 2, 3, 5, 6, 7]] == 1).all(1)
    score = bank["epdms"] > 0.95
    info["funnel"] = dict(
        raw_score_gt95=int(score.sum()),
        raw_safety_gt95=int((score & safe).sum()),
        physical_gt95=int((score & safe & bank["physical"]).sum()),
        tracking_gt95=int((score & safe & bank["physical"] & bank["tracking"]).sum()),
        space_gt95=int(bank["admissible"].sum()),
        physical_all=int(bank["physical"].sum()),
    )
    info["files"] = {
        n: checksum(stage / n)
        for n in ("candidates.npy", "high_score.npz", "contract.json", "expert.json")
    }
    write_json(stage / "summary.json", info)
    read_scene(stage)
    installed = install(root, stage, row["split"], row["token"], "recovery")
    write_json(
        work / "PUBLISHED.json",
        dict(
            token=row["token"],
            destination=str(dest),
            selected_count=256,
            committed_unix=time.time(),
        ),
    )
    return installed
