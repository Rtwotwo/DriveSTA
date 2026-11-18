"""Complete one scene using shared TTO construction and official PDM calls."""

from pathlib import Path
import time
import numpy as np
from .budget_sampler import next_request, coverage
from .scene_store import SceneTable, validate_rows, read_scene
from .store import read_json, write_json, checksum, lock


def generate(record, job_tokens, output, protocol, config, contract_id):
    from .source import load_scene, construct, Evaluator

    directory = Path(output) / "scenes" / record["split"] / record["token"]
    start = time.monotonic()
    with lock(directory):
        if (directory / "COMPLETE.json").exists():
            if read_json(directory / "contract.json")["run_contract"] != contract_id:
                raise ValueError("Completed scene belongs to a different run contract")
            identity, rows = read_scene(directory, verify=True)
            del rows
            if identity["token"] != record["token"]:
                raise ValueError("Scene identity mismatch")
            return read_json(directory / "COMPLETE.json")
        load_start = time.monotonic()
        _, gt, cache, metadata = load_scene(record, job_tokens, protocol)
        load_s = time.monotonic() - load_start
        contract = dict(identity=metadata, run_contract=contract_id)
        table = SceneTable(directory, contract, config["sampling"]["max_unique_goals"])
        reused = len(table.rows)
        evaluator = Evaluator(cache)
        score_s, construct_s, io_s, sampler_s = 0.0, 0.0, 0.0, 0.0
        try:
            expert_path = directory / "expert.json"
            if not expert_path.exists():
                write_json(
                    expert_path, dict(trajectory=gt.tolist(), labels=evaluator.score(gt).tolist())
                )
            while len(table.rows) < len(table.data):
                if table.state["pending"] is None:
                    before = time.monotonic()
                    request = next_request(
                        table.rows, table.state["completed_stages"], config, record["token"]
                    )
                    sampler_s += time.monotonic() - before
                    if request is None:
                        break
                    table.begin(request)
                pending = table.state["pending"]
                while pending["offset"] < len(pending["keys"]):
                    offset = pending["offset"]
                    keys = np.asarray(
                        pending["keys"][offset : offset + config["storage"]["chunk_candidates"]],
                        dtype=np.int32,
                    )
                    before = time.monotonic()
                    trajectories, _ = construct(gt, keys, protocol)
                    construct_s += time.monotonic() - before
                    before = time.monotonic()
                    labels = np.stack([evaluator.score(t) for t in trajectories])
                    score_s += time.monotonic() - before
                    before = time.monotonic()
                    table.append(keys, trajectories, labels)
                    io_s += time.monotonic() - before
                table.end()
            # A crash after the final data commit can leave only the phase marker pending.
            if table.state["pending"] is not None:
                table.end()
            validate_rows(table.rows, gt, protocol)
            scores = table.rows["labels"][:, -1]
            zero = np.flatnonzero((table.rows["keys"] == 0).all(axis=1))
            if len(zero) != 1:
                raise ValueError("Expected one zero-offset candidate")
            audit_start = time.monotonic()
            probe_ids = sorted({int(zero[0]), int(np.argmax(scores)), int(np.argmin(scores))})
            audit_evaluator = Evaluator(cache)
            errors = []
            for index in probe_ids:
                values, _ = audit_evaluator.score(table.rows["trajectories"][index], details=True)
                errors.append(float(abs(values - table.rows["labels"][index]).max()))
            if max(errors) > protocol["labeling"]["float64_parity_tolerance"]:
                raise ValueError("Official details audit disagrees with stored labels")
            expert = read_json(expert_path)
            expert_values, _ = audit_evaluator.score(np.asarray(expert["trajectory"]), details=True)
            if np.max(abs(expert_values - expert["labels"])) > 1e-9:
                raise ValueError("Expert score audit failed")
            result = dict(
                token=record["token"],
                split=record["split"],
                log_name=record["log_name"],
                passed=True,
                candidates=len(table.rows),
                capacity=len(table.data),
                high_count=int((scores >= 0.95).sum()),
                zero_count=int((scores == 0).sum()),
                best_pdms=float(scores.max() * 100),
                expert_pdms=float(expert_values[-1] * 100),
                coverage=coverage(table.rows, config),
                phases=table.state["completed_stages"],
                official_audit_indices=probe_ids,
                max_official_error=max(errors),
                data_digest=checksum(table.path),
                data_bytes=table.path.stat().st_size,
                invocation=dict(
                    reused_candidates=reused,
                    new_candidates=len(table.rows) - reused,
                    elapsed_s=time.monotonic() - start,
                    load_s=load_s,
                    score_s=score_s,
                    construct_s=construct_s,
                    write_s=io_s,
                    sampling_s=sampler_s,
                    audit_s=time.monotonic() - audit_start,
                ),
                completed_unix=time.time(),
            )
            write_json(directory / "COMPLETE.json", result)
            return result
        finally:
            table.close()
