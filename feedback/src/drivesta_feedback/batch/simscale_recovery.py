"""Bounded GT-neighborhood recovery, official verification and preserving merge."""

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import copy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

import numpy as np

from .store import atomic, write_json, read_json, write_npz, checksum, digest, lock
from .simscale_dataset import path_key, validate_physical_rows
from .physical_sampling import physical_mask, score_mask, select_bank


def gt_proposals(reference, coefficients):
    """Observed-window time warp and bounded offsets; zero preserves reference."""
    from .recovery_gt_local import perturb_expert

    poses = np.vstack([np.zeros(3), reference])
    times = np.arange(9) * 0.5
    return np.asarray([perturb_expert(poses, times, c) for c in coefficients], np.float32)


def local_parent_indices(paths, priorities, reference, max_deviation, count, min_ade, ids=None):
    """Rank within the feasible GT neighborhood, preserving original row IDs."""
    from .adaptive_sampling import diverse_indices

    ids = np.arange(len(paths)) if ids is None else np.asarray(ids, dtype=int)
    if not len(ids):
        return np.empty(0, dtype=int)
    near = (
        np.linalg.norm(paths[ids, :, :2] - reference[None, :, :2], axis=-1).max(1) <= max_deviation
    )
    ids = ids[near]
    if not len(ids):
        return np.empty(0, dtype=int)
    return ids[diverse_indices(paths[ids], priorities[ids], count, min_ade)]


def load_existing(folder):
    rows = np.load(folder / "candidates.npy", allow_pickle=False)
    with np.load(folder / "construction.npz", allow_pickle=False) as z:
        arrays = dict(
            trajectories=rows["trajectories"].copy(),
            labels=rows["labels"].copy(),
            alternative_labels=z["alternative_labels"].copy(),
            eligible=z["eligible"].copy(),
            tracking=z["tracking"].copy(),
        )
        seeds = z["seeds"].copy()
    if (folder / "extra_queries.npz").exists():
        with np.load(folder / "extra_queries.npz", allow_pickle=False) as z:
            for k in arrays:
                arrays[k] = np.concatenate([arrays[k], z[k]])
    # Every path has exactly one retained label record.
    keep = []
    seen = set()
    for i, t in enumerate(arrays["trajectories"]):
        key = path_key(t)
        if key not in seen:
            keep.append(i)
            seen.add(key)
    return {k: v[keep] for k, v in arrays.items()}, seeds


def source_folder(token, cfg):
    root = Path(cfg["source"])
    current = root / "scenes/train" / token
    backup = root / "previous_attempts" / cfg["run_id"] / "train" / token
    return current if current.exists() else backup


def repaired_cache_path(token, cfg, contract):
    """Opt in only to an audited, isolated cache for this exact source."""
    old = contract["identity"]["official_cache_path"]
    if not cfg.get("cache_repairs"):
        return old, None
    record = read_json(Path(cfg["cache_repairs"]) / "scenes" / token / "repair.json")
    if record["token"] != token or record["provenance"]["source_cache_path"] != old:
        raise ValueError("Cache repair does not describe the source scene")
    return record["cache_path"], record["provenance"]


def relabel_existing(
    old, out, cfg, evaluator, space, expert, policy, velocity, acceleration, origin
):
    """Old labels cannot be reused after changing map precision or route."""
    from .filter_targets import tracking_errors

    target = out / "relabeled.npz"
    if target.exists():
        with np.load(target, allow_pickle=False) as z:
            if z["digest"].tobytes().decode() != cfg["digest"]:
                raise ValueError("Relabel configuration changed")
            if not np.array_equal(z["trajectories"], old["trajectories"]):
                raise ValueError("Relabel source paths changed")
            return {k: z[k].copy() for k in old}
    result = {k: v.copy() for k, v in old.items()}
    limits = [
        policy.max_tracking_ade_m,
        policy.max_tracking_error_m,
        policy.max_tracking_heading_error_rad,
    ]
    for start in range(0, len(old["labels"]), cfg["score_batch"]):
        ids = slice(start, start + cfg["score_batch"])
        paths = old["trajectories"][ids]
        y, simulated = evaluator.score(paths)
        alt = evaluator.alternative_labels.copy()
        track = np.asarray([tracking_errors(t, s, origin) for t, s in zip(paths, simulated)])
        physical, _ = physical_mask(paths, expert, policy, velocity, acceleration)
        result["labels"][ids] = y
        result["alternative_labels"][ids] = alt
        result["tracking"][ids] = track
        result["eligible"][ids] = (
            physical
            & score_mask(y, policy)
            & score_mask(alt, policy)
            & (track <= limits).all(1)
            & space.batch(paths)
        )
    write_npz(target, **result, digest=np.frombuffer(cfg["digest"].encode(), np.uint8))
    return result


def verify_stage(folder, cache, policy):
    """Fresh evaluation of all 256, with the unchanged full driving admission."""
    from .batched_official import BatchedOfficial
    from .adaptive_sampling import PreparedSpace
    from .filter_targets import tracking_errors
    from .high_score_bank import read_scene, collate_high_score

    contract = read_json(folder / "contract.json")
    constructor = contract["protocol"]["constructor"]
    rows = np.load(folder / "candidates.npy", allow_pickle=False)
    if len(rows) != 3072:
        raise ValueError("Full bank must contain 3072 records")
    validate_physical_rows(rows, constructor)
    identity, high = read_scene(folder, verify=True)
    q = high["query_indices"]
    for k in ("keys", "trajectories", "labels"):
        if not np.array_equal(rows[k][q], high[k]):
            raise ValueError("High/full mapping differs")
    with np.load(folder / "construction.npz", allow_pickle=False) as z:
        alt = z["alternative_labels"].copy()
        eligible = z["eligible"].copy()
    if not eligible[q].all():
        raise ValueError("High bank includes rejected paths")
    independent = BatchedOfficial(cache)
    space = PreparedSpace(cache)
    origin = identity["ego_pose"]
    limits = [
        policy.max_tracking_ade_m,
        policy.max_tracking_error_m,
        policy.max_tracking_heading_error_rad,
    ]
    error = 0.0
    for start in range(0, 256, 31):
        ids = q[start : start + 31]
        paths = rows["trajectories"][ids]
        labels, tracked = independent.score(paths)
        error = max(
            error,
            float(np.abs(labels - rows["labels"][ids]).max()),
            float(np.abs(independent.alternative_labels - alt[ids]).max()),
        )
        errors = np.asarray([tracking_errors(t, s, origin) for t, s in zip(paths, tracked)])
        if not (
            score_mask(labels, policy).all()
            and score_mask(independent.alternative_labels, policy).all()
            and (errors <= limits).all()
            and space.batch(paths).all()
        ):
            raise ValueError("Fresh official evaluation or driving admission failed")
    for i in q[[0, 127, 255]]:
        error = max(
            error,
            float(
                np.abs(independent.runtime.score(rows["trajectories"][i]) - rows["labels"][i]).max()
            ),
        )
    if error > 1e-8:
        raise ValueError("Official score differs from saved label")
    batch = collate_high_score([(identity, high)])
    if tuple(batch["scorer_trajectories"].shape) != (1, 256, 8, 3):
        raise ValueError("Training reader shape")
    return dict(official_max_error=error, verified_high=256, full_rows=3072, reader_passed=True)


def publish_stage(stage, token, cfg):
    """Keep original directory intact in previous_attempts, then publish stage."""
    root = Path(cfg["source"])
    target = root / "scenes/train" / token
    backup = root / "previous_attempts" / cfg["run_id"] / "train" / token
    with lock(root / "merge_locks" / token):
        if (
            target.exists()
            and read_json(target / "RESULT.json").get("recovery_id") == cfg["run_id"]
        ):
            return str(backup)
        if target.exists():
            if backup.exists():
                raise FileExistsError("Refuse to overwrite original backup")
            backup.parent.mkdir(parents=True, exist_ok=True)
            os.replace(target, backup)
        elif not backup.exists():
            raise FileNotFoundError("Both original and backup missing")
        try:
            os.replace(stage, target)
        except Exception:
            if not target.exists():
                os.replace(backup, target)
            raise
        return str(backup)


def make_stage(token, cfg, original, arrays, parameters, seeds, cache, policy, recovery_stats):
    from .scene_store import ROW_DTYPE, row_digest
    from .high_score_bank import validate_bank

    full_ids, high_ids = select_bank(arrays["trajectories"], arrays["labels"], arrays["eligible"])
    if len(full_ids) != 3072 or len(high_ids) != 256:
        raise ValueError("Incomplete stage")
    stage = Path(cfg["output"]) / "staged/train" / token
    stage.mkdir(parents=True, exist_ok=True)
    rows = np.zeros(3072, dtype=ROW_DTYPE)
    rows["keys"][:, 1] = full_ids
    rows["trajectories"] = arrays["trajectories"][full_ids]
    rows["labels"] = arrays["labels"][full_ids]
    contract = copy.deepcopy(read_json(original / "contract.json"))
    cache_path, repair = repaired_cache_path(token, cfg, contract)
    if repair:
        contract["identity"]["official_cache_path"] = cache_path
        contract["identity"]["geometry_repair"] = repair
    contract["protocol"]["constructor"]["config_digest"] = cfg["digest"]
    contract["protocol"]["sampling"] = dict(
        mode="gt_local_tangent_recovery" if cfg.get("gt_local_sampling") else "bounded_gt_recovery",
        recovery_id=cfg["run_id"],
        max_gt_deviation_m=cfg["max_gt_deviation_m"],
        coefficient_bounds=cfg["coefficient_bounds"],
    )
    contract["recovery"] = dict(
        source_scene_before_publication=str(original),
        original_backup=str(
            Path(cfg["source"]) / "previous_attempts" / cfg["run_id"] / "train" / token
        ),
        query_archive=str(Path(cfg["output"]) / "queries" / token),
        retained_original=True,
        scoring_protocol_unchanged=True,
    )
    validate_physical_rows(rows, contract["protocol"]["constructor"])
    expert_record = read_json(original / "expert.json")
    if repair:
        from .batched_official import BatchedOfficial

        evaluator = BatchedOfficial(cache)
        labels, _ = evaluator.score(np.asarray(expert_record["trajectory"], np.float32)[None])
        expert_record = dict(
            expert_record,
            labels=labels[0].tolist(),
            alternative_labels=evaluator.alternative_labels[0].tolist(),
            geometry_repair=repair,
        )
    write_json(stage / "contract.json", contract)
    write_json(stage / "expert.json", expert_record)
    atomic(stage / "candidates.npy", lambda f: np.save(f, rows, allow_pickle=False))
    write_npz(
        stage / "construction.npz",
        seeds=seeds,
        parameters=parameters[full_ids],
        tracking=arrays["tracking"][full_ids],
        eligible=arrays["eligible"][full_ids],
        alternative_labels=arrays["alternative_labels"][full_ids],
    )
    write_json(
        stage / "seeds.json",
        dict(
            metadata=[dict(parent="recovery_reference", index=i) for i in range(len(seeds))],
            parameter_semantics=(
                "parent,index,time_shift,terminal_along,terminal_across,middle_along,middle_across,early_late_across"
                if cfg.get("gt_local_sampling")
                else "parent,index,time_shift,terminal_along,terminal_across,middle_along,middle_across,middle_heading"
            ),
            scored_seed_count=0,
            selected_parents=list(range(len(seeds))),
        ),
    )
    write_json(
        stage / "state.json",
        dict(
            count=3072,
            pending=None,
            completed_stages=["bounded_gt_recovery"],
            blocks=[dict(start=0, end=3072, digest=row_digest(rows))],
        ),
    )
    write_json(
        stage / "ALL_SCORES.json",
        dict(
            format="all_scored_candidates_v1",
            passed=True,
            candidates=3072,
            token=token,
            split="train",
            data_digest=checksum(stage / "candidates.npy"),
        ),
    )
    high = {k: rows[k][high_ids] for k in ("keys", "trajectories", "labels")}
    high["query_indices"] = high_ids
    validate_bank(high, np.asarray(contract["identity"]["expert_gt"]), contract["protocol"])
    write_npz(stage / "high_score.npz", **high)
    # Completion is only externally visible after verification and rename.
    write_json(
        stage / "COMPLETE.json",
        dict(
            format="high_score_256",
            passed=True,
            candidates=256,
            token=token,
            split="train",
            data_digest=checksum(stage / "high_score.npz"),
            minimum_pdms=float(high["labels"][:, -1].min() * 100),
        ),
    )
    extra = np.setdiff1d(np.arange(len(arrays["labels"])), full_ids)
    write_npz(
        stage / "extra_queries.npz",
        **{k: v[extra] for k, v in arrays.items()},
        source_query_indices=extra,
        parameters=parameters[extra],
    )
    verification = verify_stage(stage, cache, policy)
    complete = read_json(stage / "COMPLETE.json")
    complete.update(verification)
    write_json(stage / "COMPLETE.json", complete)
    result = dict(
        token=token,
        source=contract["identity"]["source"],
        config_digest=cfg["digest"],
        recovery_id=cfg["run_id"],
        full_count=3072,
        high_count=256,
        complete_full=True,
        complete_high=True,
        scored_queries=len(arrays["labels"]),
        initial_space_valid=True,
        official_verification=verification,
        recovery_stats=recovery_stats,
        files={},
    )
    names = [
        "contract.json",
        "expert.json",
        "candidates.npy",
        "construction.npz",
        "seeds.json",
        "state.json",
        "ALL_SCORES.json",
        "high_score.npz",
        "COMPLETE.json",
        "extra_queries.npz",
    ]
    result["files"] = {n: checksum(stage / n) for n in names}
    write_json(stage / "RESULT.json", result)
    return stage, verification


def recover(token, cfg):
    import torch

    torch.set_num_threads(1)
    from drivesta_feedback.audit.simscale_recovery import load_cache
    from .filter_targets import TargetPolicy, tracking_errors, MissingPlannedGeometry
    from .adaptive_sampling import PreparedSpace, diverse_indices, quality, shuffled_parameters
    from .batched_official import BatchedOfficial
    from .recovery_seeds import to_local

    begin = time.time()
    out = Path(cfg["output"]) / "queries" / token
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        row = read_json(out / "result.json")
        if row["config_digest"] != cfg["digest"]:
            raise ValueError("Resume config changed")
        if row.get("recovered") and cfg.get("merge") and not row.get("merged"):
            stage = Path(cfg["output"]) / "staged/train" / token
            target = Path(cfg["source"]) / "scenes/train" / token
            check_folder = stage if stage.exists() else target
            if read_json(check_folder / "RESULT.json").get("config_digest") != cfg["digest"]:
                raise ValueError("Staged result uses another configuration")
            contract = read_json(check_folder / "contract.json")
            policy = TargetPolicy(**cfg["policy"])
            verify_stage(
                check_folder, load_cache(contract["identity"]["official_cache_path"]), policy
            )
            if stage.exists():
                old = contract["recovery"].pop("source_scene", None)
                if old:
                    contract["recovery"]["source_scene_before_publication"] = old
                contract["recovery"]["original_backup"] = str(
                    Path(cfg["source"]) / "previous_attempts" / cfg["run_id"] / "train" / token
                )
                write_json(stage / "contract.json", contract)
                published = read_json(stage / "RESULT.json")
                published["recovery_stats"] = {
                    k: row[k]
                    for k in (
                        "queries",
                        "qualified",
                        "proposed",
                        "reason",
                        "elapsed_s",
                        "relabeled_queries",
                        "seed_scored_queries",
                        "seed_duplicate_queries",
                        "actual_additional_scoring_queries",
                    )
                    if k in row
                }
                published["files"]["contract.json"] = checksum(stage / "contract.json")
                write_json(stage / "RESULT.json", published)
            row["backup"] = publish_stage(stage, token, cfg)
            row["merged"] = True
            write_json(out / "result.json", row)
        return dict(row, reused=True)
    original = source_folder(token, cfg)
    old_result = read_json(original / "RESULT.json")
    if old_result.get("recovery_id") == cfg["run_id"]:
        if old_result["config_digest"] != cfg["digest"]:
            raise ValueError("Already published under a different configuration")
        for name, value in old_result["files"].items():
            if checksum(original / name) != value:
                raise ValueError("Published recovery file changed")
        row = dict(
            old_result["recovery_stats"],
            token=token,
            merged=True,
            recovered=True,
            reused=True,
            verification=old_result["official_verification"],
            config_digest=cfg["digest"],
            backup=str(Path(cfg["source"]) / "previous_attempts" / cfg["run_id"] / "train" / token),
        )
        write_json(out / "result.json", row)
        return row
    contract = read_json(original / "contract.json")
    constructor = contract["protocol"]["constructor"]
    if constructor["policy"] != cfg["policy"]:
        raise ValueError("Safety policy mismatch")
    policy = TargetPolicy(**cfg["policy"])
    expert = np.asarray(constructor["expert"], np.float32)
    cache_path, repair = repaired_cache_path(token, cfg, contract)
    cache = load_cache(cache_path)
    try:
        space = PreparedSpace(cache)
    except MissingPlannedGeometry:
        space = None
    if space is None or not space(np.zeros((8, 3), np.float32)):
        row = dict(
            token=token,
            merged=False,
            recovered=False,
            reason="immutable_initial_space",
            queries=0,
            config_digest=cfg["digest"],
            elapsed_s=time.time() - begin,
        )
        write_json(out / "result.json", row)
        return row
    old, seeds = load_existing(original)
    evaluator = BatchedOfficial(cache)
    origin = contract["identity"]["ego_pose"]
    limits = [
        policy.max_tracking_ade_m,
        policy.max_tracking_error_m,
        policy.max_tracking_heading_error_rad,
    ]
    velocity, acceleration = constructor["initial_velocity"], constructor["initial_acceleration"]
    if repair:
        old = relabel_existing(
            old, out, cfg, evaluator, space, expert, policy, velocity, acceleration, origin
        )
        if repair.get("route_repair"):
            from .simscale_source import seed_paths

            # A changed reference supplies new search parents, never old labels.
            seeds = seed_paths(cache, expert)[0]
    initial_size = len(old["labels"])
    params = np.zeros((initial_size, 8))
    params[:, 0] = -1
    params[:, 1] = np.arange(initial_size)
    checkpoint = out / "checkpoint.npz"
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as z:
            if z["digest"].tobytes().decode() != cfg["digest"]:
                raise ValueError("Checkpoint changed")
            arrays = {k: z[k].copy() for k in old}
            parents = z["parents"].copy()
            params = z["parameters"].copy()
            cursor = int(z["cursor"])
    else:
        arrays = old
        cursor = 0
        # Parent labels are only search hints. All child paths get new official labels.
        gt_labels, sim = evaluator.score(expert[None])
        if cfg.get("gt_local_sampling"):
            gt_physical, _ = physical_mask(expert[None], expert, policy, velocity, acceleration)
            gt_track = np.asarray(tracking_errors(expert, sim[0], origin))
            write_json(
                out / "gt_reference.json",
                dict(
                    labels=gt_labels[0].tolist(),
                    alternative_labels=evaluator.alternative_labels[0].tolist(),
                    physical=bool(gt_physical[0]),
                    tracking=gt_track.tolist(),
                    planned_space=bool(space(expert)),
                    reference_only=True,
                    counts_as_new_target=False,
                ),
            )
        projected = to_local(sim[0, :, :3], origin)[5::5]
        priorities = quality(
            old["labels"], old["alternative_labels"], old["tracking"], limits, old["eligible"]
        )
        if cfg.get("parent_selection") == "gt_feasible_first_v1":
            ids = local_parent_indices(
                old["trajectories"], priorities, expert, cfg["max_gt_deviation_m"], 24, 0.1
            )
        else:
            ids = diverse_indices(old["trajectories"], priorities, 24, 0.1)
        candidates = np.concatenate(
            [expert[None], projected[None], seeds, old["trajectories"][ids]]
        )
        parents = []
        for t in candidates:
            if np.linalg.norm(t[:, :2] - expert[:, :2], axis=1).max() <= cfg["max_gt_deviation_m"]:
                if not any(path_key(t) == path_key(p) for p in parents):
                    parents.append(t)
        parents = np.asarray(parents, np.float32)
        if cfg.get("gt_local_sampling"):
            parents = expert[None].copy()
        if (cfg.get("constrained_seeds") or cfg.get("tto_seeds")) and np.count_nonzero(
            arrays["eligible"]
        ) < 256:
            if cfg.get("tto_seeds"):
                from .simscale_tto_seeds import search_seeds as optimize_seeds
            else:
                from .simscale_constrained_seeds import optimize_seeds
            raw_expert = np.asarray(contract["identity"]["raw_expert_gt"], np.float32)
            optimized = optimize_seeds(
                out,
                cfg,
                evaluator,
                space,
                expert,
                policy,
                velocity,
                acceleration,
                origin,
                arrays,
                parents,
                raw_expert,
            )
            known = {path_key(t) for t in arrays["trajectories"]}
            known.update((path_key(expert), path_key(raw_expert)))
            ids = []
            for i, t in enumerate(optimized["trajectories"]):
                key = path_key(t)
                if key not in known:
                    ids.append(i)
                    known.add(key)
            if ids:
                arrays = {k: np.concatenate([v, optimized[k][ids]]) for k, v in arrays.items()}
                provenance = np.zeros((len(ids), 8))
                provenance[:, 0] = -2
                provenance[:, 1] = ids
                params = np.concatenate([params, provenance])
            write_npz(
                checkpoint,
                **arrays,
                parents=parents,
                parameters=params,
                cursor=np.array(cursor),
                digest=np.frombuffer(cfg["digest"].encode(), np.uint8),
            )
    seen = {path_key(t) for t in arrays["trajectories"]}
    seen.add(path_key(expert))
    seen.add(path_key(np.asarray(contract["identity"]["raw_expert_gt"], np.float32)))
    random_seed = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) ^ 20261020
    if cfg.get("gt_local_sampling"):
        from .gt_neighborhood import gt_local_proposals, neighborhood_parameters

        designs = neighborhood_parameters(cfg["max_proposals"], random_seed)
    else:
        designs = shuffled_parameters(cfg["max_proposals"], random_seed)
    bounds = np.asarray(cfg["coefficient_bounds"])
    proposed = cursor
    while (
        cursor < cfg["max_proposals"] and len(arrays["labels"]) - initial_size < cfg["max_queries"]
    ):
        if np.count_nonzero(arrays["eligible"]) >= 256:
            break
        # Feedback updates once per committed block; checkpoint includes parents.
        qualified = np.flatnonzero(arrays["eligible"])
        if len(qualified):
            if cfg.get("parent_selection") == "gt_feasible_first_v1":
                ids = local_parent_indices(
                    arrays["trajectories"],
                    arrays["labels"][:, -1],
                    expert,
                    cfg["max_gt_deviation_m"],
                    12,
                    0.01,
                    qualified,
                )
            else:
                ids = qualified[
                    diverse_indices(
                        arrays["trajectories"][qualified], arrays["labels"][qualified, -1], 12, 0.01
                    )
                ]
        elif not cfg.get("gt_local_sampling"):
            priorities = quality(
                arrays["labels"],
                arrays["alternative_labels"],
                arrays["tracking"],
                limits,
                arrays["eligible"],
            )
            if cfg.get("parent_selection") == "gt_feasible_first_v1":
                ids = local_parent_indices(
                    arrays["trajectories"], priorities, expert, cfg["max_gt_deviation_m"], 12, 0.05
                )
            else:
                ids = np.asarray(diverse_indices(arrays["trajectories"], priorities, 12, 0.05), int)
        else:
            ids = []
        local = []
        for i in ids:
            t = arrays["trajectories"][i]
            if np.linalg.norm(t[:, :2] - expert[:, :2], axis=1).max() > cfg["max_gt_deviation_m"]:
                continue
            matches = [j for j, p in enumerate(parents) if path_key(p) == path_key(t)]
            if matches:
                local.append(matches[0])
            else:
                local.append(len(parents))
                parents = np.concatenate([parents, t[None]])
        stop = min(cursor + 256, cfg["max_proposals"])
        indices = np.arange(cursor, stop)
        provenance = []
        for j in indices:
            if cfg.get("gt_local_sampling"):
                parent = local[(j // 2) % len(local)] if local and j % 2 else 0
                scale = (0.05, 0.15, 0.4)[(j // 2) % 3] if parent else 1.0
                coefficient = designs[j] * bounds * scale
                provenance.append([parent, j, *coefficient])
                continue
            parent = 0 if j % 4 < 2 else (j // 4) % len(parents)
            scale = (0.15, 0.4, 1.0)[(j // 4) % 3]
            if local and j % 4 != 0:
                parent = local[(j // 4) % len(local)]
                scale = (
                    (0.01, 0.03, 0.08)[j % 4 - 1] if len(qualified) else (0.2, 0.5, 1.0)[j % 4 - 1]
                )
            coefficient = np.round(designs[j] * bounds * scale, 4)
            provenance.append([parent, j, *coefficient])
        provenance = np.asarray(provenance, float)
        candidates = np.zeros((len(indices), 8, 3), np.float32)
        for parent in np.unique(provenance[:, 0]).astype(int):
            ids = np.flatnonzero(provenance[:, 0] == parent)
            proposer = gt_local_proposals if cfg.get("gt_local_sampling") else gt_proposals
            candidates[ids] = proposer(parents[parent], provenance[ids, 2:])
        physical, _ = physical_mask(candidates, expert, policy, velocity, acceleration)
        near = (
            np.linalg.norm(candidates[:, :, :2] - expert[None, :, :2], axis=-1).max(1)
            <= cfg["max_gt_deviation_m"]
        )
        keep = []
        for i in np.flatnonzero(physical & near):
            k = path_key(candidates[i])
            if k not in seen:
                seen.add(k)
                keep.append(i)
        keep = keep[: cfg["max_queries"] - (len(arrays["labels"]) - initial_size)]
        for start in range(0, len(keep), cfg["score_batch"]):
            ids = keep[start : start + cfg["score_batch"]]
            paths = candidates[ids]
            y, sim = evaluator.score(paths)
            alt = evaluator.alternative_labels.copy()
            track = np.asarray([tracking_errors(t, s, origin) for t, s in zip(paths, sim)])
            ok = (
                score_mask(y, policy)
                & score_mask(alt, policy)
                & (track <= limits).all(1)
                & space.batch(paths)
            )
            new = dict(
                trajectories=paths, labels=y, alternative_labels=alt, tracking=track, eligible=ok
            )
            arrays = {k: np.concatenate([v, new[k]]) for k, v in arrays.items()}
            params = np.concatenate([params, provenance[ids]])
        cursor = stop
        proposed = cursor
        write_npz(
            checkpoint,
            **arrays,
            parents=parents,
            parameters=params,
            cursor=np.array(cursor),
            digest=np.frombuffer(cfg["digest"].encode(), np.uint8),
        )
    # All new scored candidates are kept even when no usable high bank is found.
    write_npz(
        out / "queries.npz",
        **{k: v[initial_size:] for k, v in arrays.items()},
        parameters=params[initial_size:],
        parents=parents,
    )
    recovered = int(np.count_nonzero(arrays["eligible"])) >= 256
    verification = None
    backup = None
    recovery_stats = dict(
        queries=len(arrays["labels"]) - initial_size,
        relabeled_queries=initial_size if repair else 0,
        qualified=int(np.count_nonzero(arrays["eligible"])),
        proposed=proposed,
        reason="complete" if recovered else "bounded_search_exhausted",
        elapsed_s=time.time() - begin,
    )
    if (out / "tto_seeds.json").exists():
        seed_stats = read_json(out / "tto_seeds.json")
        recovery_stats["seed_scored_queries"] = seed_stats["queries"]
        recovery_stats["seed_duplicate_queries"] = seed_stats["duplicate_existing_queries"]
        recovery_stats["actual_additional_scoring_queries"] = (
            recovery_stats["queries"] + seed_stats["duplicate_existing_queries"]
        )
    if recovered:
        stage, verification = make_stage(
            token, cfg, original, arrays, params, parents, cache, policy, recovery_stats
        )
        backup = publish_stage(stage, token, cfg) if cfg.get("merge") else None
        if cfg.get("merge"):
            # Read final location with the real downstream reader as well.
            from .high_score_bank import read_scene

            read_scene(Path(cfg["source"]) / "scenes/train" / token, verify=True)
    row = dict(
        recovery_stats,
        token=token,
        recovered=recovered,
        merged=bool(backup),
        backup=backup,
        verification=verification,
        config_digest=cfg["digest"],
        elapsed_s=time.time() - begin,
    )
    write_json(out / "result.json", row)
    return row


def worker(token, cfg):
    try:
        return recover(token, cfg)
    except Exception:
        return dict(token=token, error=traceback.format_exc())


def run(cfg, tokens, workers):
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    with lock(output / "runner"):
        run_path = output / "run.json"
        if run_path.exists():
            previous_run = read_json(run_path)
            new_run = dict(config=cfg, tokens=tokens)
            # The only permitted phase transition is publishing the same
            # previously validated pilot, with unchanged tokens and semantics.
            publication = copy.deepcopy(previous_run)
            publication["config"]["merge"] = True
            if previous_run != new_run and not (cfg["merge"] and publication == new_run):
                raise ValueError("Resume run differs")
        write_json(run_path, dict(config=cfg, tokens=tokens))
        previous = {}
        journal = output / "results.jsonl"
        if journal.exists():
            previous = {r["token"]: r for r in map(json.loads, journal.read_text().splitlines())}
        results = {
            t: r
            for t, r in previous.items()
            if t in tokens
            and "error" not in r
            and not (cfg.get("merge") and r.get("recovered") and not r.get("merged"))
        }
        todo = [t for t in tokens if t not in results]
        start = time.time()
        last = 0.0

        def report(final=False):
            recovered = [t for t, r in results.items() if r.get("recovered")]
            merged = [t for t, r in results.items() if r.get("merged")]
            errors = [r for r in results.values() if "error" in r]
            summary = dict(
                status="finished" if final else "running",
                pid=os.getpid(),
                workers=workers,
                scenes=len(tokens),
                processed=len(results),
                recovered=len(recovered),
                merged=len(merged),
                errors=errors,
                queries=sum(
                    r.get("actual_additional_scoring_queries", r.get("queries", 0))
                    for r in results.values()
                ),
                relabeled_queries=sum(r.get("relabeled_queries", 0) for r in results.values()),
                elapsed_s=time.time() - start,
                updated_unix=time.time(),
            )
            write_json(output / "status.json", summary)
            atomic(
                output / "recovered_tokens.txt",
                lambda f: f.write(("\n".join(recovered) + "\n").encode()),
            )
            atomic(
                output / "remaining_tokens.txt",
                lambda f: f.write(
                    ("\n".join(t for t in tokens if t not in recovered) + "\n").encode()
                ),
            )
            if cfg.get("merge"):
                root = Path(cfg["source"])
                snapshot = read_json(output / "source_snapshot.json")
                baseline, original = snapshot["status"], snapshot["shortfall_tokens"]
                all_merged = set(merged) & set(original)
                remaining = [t for t in original if t not in all_merged]
                global_status = dict(
                    baseline,
                    status="partial" if final else "recovery_running",
                    high256=baseline["high256"] + len(all_merged),
                    shortfall=len(remaining),
                    recovery_processed=baseline.get("recovery_processed", 0) + len(results),
                    recovery_errors=baseline.get("recovery_errors", 0) + len(errors),
                    recovered=baseline.get("recovered", 0) + len(all_merged),
                    scored_queries=baseline["scored_queries"]
                    + summary["queries"]
                    + summary["relabeled_queries"],
                    recovery_scored_queries=baseline.get("recovery_scored_queries", 0)
                    + summary["queries"]
                    + summary["relabeled_queries"],
                    first_pass_completed=True,
                    updated_unix=time.time(),
                    pid=os.getpid(),
                    workers=0 if final else workers,
                    recovery_output=str(output),
                    eta_hours=0.0 if final else None,
                    scene_per_hour=None,
                    recovery_elapsed_s=summary["elapsed_s"],
                )
                write_json(root / "status.json", global_status)
                atomic(
                    root / "shortfall_tokens.txt",
                    lambda f: f.write(("\n".join(remaining) + "\n").encode()),
                )
                write_json(root / "PARTIAL.json", dict(global_status, passed=False))
            print(json.dumps(summary), flush=True)

        with (
            ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool,
            journal.open("a") as stream,
        ):
            iterator = iter(todo)
            pending = {}

            def fill():
                while len(pending) < workers * 2:
                    try:
                        t = next(iterator)
                    except StopIteration:
                        break
                    pending[pool.submit(worker, t, cfg)] = t

            fill()
            report()
            while pending:
                done, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for f in done:
                    t = pending.pop(f)
                    try:
                        r = f.result()
                    except Exception:
                        r = dict(token=t, error=traceback.format_exc())
                    results[t] = r
                    stream.write(json.dumps(r) + "\n")
                    stream.flush()
                fill()
                if time.time() - last > 30:
                    report()
                    last = time.time()
            report(True)


def main():
    import yaml
    from drivesta_feedback.paths import bootstrap_navsim

    root = bootstrap_navsim()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--tokens", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=128)
    p.add_argument("--merge", action="store_true")
    a = p.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    if cfg.get("parent_selection") not in (None, "gt_feasible_first_v1"):
        raise ValueError("Unknown parent selection policy")
    cfg["source"] = str(root / cfg["source"])
    cfg["output"] = str(a.output.resolve())
    cfg["merge"] = a.merge
    if cfg.get("cache_repairs"):
        cfg["cache_repairs"] = str(root / cfg["cache_repairs"])
    cfg["digest"] = digest({k: v for k, v in cfg.items() if k not in ("output", "merge")})
    tokens = a.tokens.read_text().split()
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError("Invalid tokens")
    a.output.mkdir(parents=True, exist_ok=True)
    if a.merge and not (a.output / "source_snapshot.json").exists():
        write_json(
            a.output / "source_snapshot.json",
            dict(
                status=read_json(Path(cfg["source"]) / "status.json"),
                shortfall_tokens=(Path(cfg["source"]) / "shortfall_tokens.txt").read_text().split(),
            ),
        )
    run(cfg, tokens, a.workers)


if __name__ == "__main__":
    main()
