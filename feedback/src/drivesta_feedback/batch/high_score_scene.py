"""Search through the requested budget, then publish the best 256 eligible paths."""

from pathlib import Path
import time
import numpy as np
from .high_score_bank import COUNT, THRESHOLD, select_indices, save_bank, read_scene
from .high_score_sampler import next_request
from .scene_store import SceneTable, validate_rows
from .store import read_json, write_json, lock


def cleanup_staging(directory, retain_all_scores=False):
    # Only this workflow's own staging files, after a complete bank was read back.
    full = (directory / "ALL_SCORES.json").exists()
    # Keep a failed replay's diagnosis until its full bank has been published.
    names = () if retain_all_scores and not full else ("SHORTFALL.json", "ERROR.json")
    if not retain_all_scores and not full:
        names += ("candidates.npy", "state.json")
    for name in names:
        (directory / name).unlink(missing_ok=True)


def generate(record, job_tokens, output, protocol, config, contract_id):
    from .source import load_scene, construct, Evaluator

    directory = Path(output) / "scenes" / record["split"] / record["token"]
    budget = config.get("_query_budget", config["sampling"]["max_unique_goals"])
    retain = config["storage"]["retain_all_scores"]
    backfill = config.get("_backfill_full_scores", False)
    previous, previous_bank = None, None
    start = time.monotonic()
    with lock(directory):
        if backfill and not (directory / "COMPLETE.json").exists():
            raise ValueError("Backfill requires an existing verified top256 bank")
        if (directory / "COMPLETE.json").exists():
            if read_json(directory / "contract.json")["run_contract"] != contract_id:
                raise ValueError("Scene belongs to another run")
            _, previous_bank = read_scene(directory, verify=True)
            previous = read_json(directory / "COMPLETE.json")
            if not backfill or (directory / "ALL_SCORES.json").exists():
                if (directory / "ALL_SCORES.json").exists():
                    from .retained_bank import read_full_scene

                    read_full_scene(directory)
                cleanup_staging(directory, retain)
                return previous
            budget = previous["queries"]
        _, gt, cache, metadata = load_scene(record, job_tokens, protocol)
        contract = dict(identity=metadata, run_contract=contract_id, protocol=protocol)
        table = SceneTable(directory, contract, budget, allow_growth=True)
        reused = len(table.rows)
        evaluator = Evaluator(cache)
        try:
            if len(table.rows) < budget:
                (directory / "ALL_SCORES.json").unlink(missing_ok=True)
            if not (directory / "expert.json").exists():
                write_json(
                    directory / "expert.json",
                    dict(trajectory=gt.tolist(), labels=evaluator.score(gt).tolist()),
                )
            while len(table.rows) < budget or table.state["pending"] is not None:
                if table.state["pending"] is None:
                    table.begin(next_request(table.rows, config, record["token"], budget))
                pending = table.state["pending"]
                while pending["offset"] < len(pending["keys"]):
                    offset = pending["offset"]
                    keys = np.asarray(
                        pending["keys"][offset : offset + config["storage"]["chunk_candidates"]],
                        dtype=np.int32,
                    )
                    paths, _ = construct(gt, keys, protocol)
                    labels = np.stack([evaluator.score(t) for t in paths])
                    table.append(keys, paths, labels)
                table.end()
            validate_rows(table.rows, gt, protocol)
            ids = select_indices(table.rows["trajectories"], table.rows["labels"], gt)
            if previous is not None:
                from .retained_bank import check_selected_rows, publish_full_scene

                check_selected_rows(
                    table.rows, previous_bank, protocol["labeling"]["float64_parity_tolerance"]
                )
                if not np.array_equal(ids, previous_bank["query_indices"]):
                    raise ValueError("Replay selection differs from the published top256")
                publish_full_scene(directory, len(table.rows))
                return previous
            scores = table.rows["labels"][:, -1]
            result = dict(
                token=record["token"],
                split=record["split"],
                log_name=record["log_name"],
                passed=len(ids) == COUNT,
                status="complete" if len(ids) == COUNT else "shortfall",
                candidates=len(ids),
                missing_candidates=COUNT - len(ids),
                queries=len(table.rows),
                qualified_queries=int((scores >= THRESHOLD).sum()),
                format="high_score_256",
                best_pdms=float(scores.max() * 100),
                min_selected_pdms=float(scores[ids].min() * 100) if len(ids) else None,
                invocation=dict(
                    reused_queries=reused,
                    new_queries=len(table.rows) - reused,
                    elapsed_s=time.monotonic() - start,
                ),
                completed_unix=time.time(),
            )
            if len(ids) < COUNT:
                if retain:
                    from .retained_bank import publish_full_scene

                    publish_full_scene(directory, len(table.rows))
                write_json(directory / "SHORTFALL.json", result)
                return result
            # Every retained trajectory receives a second, official details evaluation.
            audit = Evaluator(cache)
            error = 0.0
            for i in ids:
                values, _ = audit.score(table.rows["trajectories"][i], details=True)
                error = max(error, float(abs(values - table.rows["labels"][i]).max()))
                if (
                    values[-1] < THRESHOLD
                    or error > protocol["labeling"]["float64_parity_tolerance"]
                ):
                    raise ValueError("Official verification failed for selected trajectory")
            _, metadata = save_bank(directory, table.rows, gt, protocol)
            if retain:
                from .retained_bank import publish_full_scene

                publish_full_scene(directory, len(table.rows))
            result.update(metadata, max_official_error=error, audited_candidates=len(ids))
            result["invocation"]["elapsed_s"] = time.monotonic() - start
            write_json(directory / "COMPLETE.json", result)
        finally:
            table.close()
        read_scene(directory, verify=True)
        cleanup_staging(directory, retain)
        return result
