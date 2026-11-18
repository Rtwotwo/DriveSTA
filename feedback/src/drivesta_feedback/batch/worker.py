"""Block-level continuation of deterministic adaptive sampling."""

import os
from pathlib import Path
import resource
import time
import numpy as np

from . import sampler
from .store import digest, lock, read_json, write_json, write_npz, validate_block, checksum


class SceneWriter:
    def __init__(self, directory, protocol, identity):
        self.directory, self.protocol = Path(directory), protocol
        self.rows = {}
        self.endpoint_error = 0.0
        self.new_blocks = 0
        self.reused_blocks = 0
        self.cap_hit = False
        self.phases = []
        contract = dict(identity=identity, protocol=protocol)
        path = self.directory / "contract.json"
        if path.exists():
            if digest(read_json(path)) != digest(contract):
                raise ValueError("Existing scene belongs to a different protocol/source")
        else:
            write_json(path, contract)

    def phase(self, name, requests, gt, evaluator, constructor, stop_after_blocks=None):
        p = self.directory / "phases" / name
        p.mkdir(parents=True, exist_ok=True)
        keys, capped = sampler.missing_keys(
            requests, self.rows, self.protocol["sampling"]["max_unique_goals"] - len(self.rows)
        )
        chunk_size = self.protocol["storage"]["chunk_candidates"]
        plan = dict(
            name=name,
            count=len(keys),
            capped=capped,
            key_digest=digest(keys.tolist()),
            chunk_size=chunk_size,
        )
        if (p / "plan.json").exists():
            if read_json(p / "plan.json") != plan:
                raise ValueError("Phase recomputation differs from committed plan: " + name)
        else:
            write_json(p / "plan.json", plan)
        block_records = []
        for i in range(0, len(keys), chunk_size):
            target = p / f"block_{i//chunk_size:05d}.npz"
            expected = keys[i : i + chunk_size]
            if target.exists():
                with np.load(target, allow_pickle=False) as z:
                    validate_block(z, expected)
                    labels = z["labels"].copy()
                    endpoint_error = float(z["endpoint_error"])
                self.reused_blocks += 1
            else:
                trajectories, endpoint_error = constructor(gt, expected, self.protocol)
                labels = np.stack([evaluator.score(t) for t in trajectories])
                validate_block(
                    dict(keys=expected, trajectories=trajectories, labels=labels), expected
                )
                write_npz(
                    target,
                    keys=expected.astype(np.int32),
                    trajectories=trajectories.astype(np.float32),
                    labels=labels,
                    endpoint_error=np.array(endpoint_error),
                )
                self.new_blocks += 1
                # Explicit crash hook exercises a committed block without a progress update.
                if stop_after_blocks is not None and self.new_blocks >= stop_after_blocks:
                    os._exit(75)
            self.endpoint_error = max(self.endpoint_error, endpoint_error)
            for k, v in zip(expected, labels):
                key = tuple(int(x) for x in k)
                if key in self.rows:
                    raise ValueError("Duplicate goal across phases")
                self.rows[key] = v
            block_records.append(
                dict(file=target.name, count=len(expected), sha256=checksum(target))
            )
            write_json(
                self.directory / "progress.json",
                dict(
                    status="running",
                    phase=name,
                    candidates=len(self.rows),
                    new_blocks=self.new_blocks,
                    reused_blocks=self.reused_blocks,
                    updated_unix=time.time(),
                ),
            )
        if len(list(p.glob("block_*.npz"))) != len(block_records):
            raise ValueError("Unexpected extra committed block")
        write_json(p / "COMPLETE.json", dict(plan, blocks=block_records))
        self.cap_hit |= capped
        self.phases.append(dict(name=name, new_candidates=len(keys), capped=capped))


def generate_scene(record, tokens, directory, protocol, stop_after_blocks=None):
    from .source import load_scene, construct, Evaluator
    import torch

    torch.set_num_threads(1)
    started = time.perf_counter()
    directory = Path(directory)
    with lock(directory):
        _, gt, cache, meta = load_scene(record, tokens, protocol)
        evaluator = Evaluator(cache)
        writer = SceneWriter(directory, protocol, meta)
        if (directory / "COMPLETE.json").exists():
            # Audit performs block checks; verify contract here before skipping work.
            return dict(read_json(directory / "COMPLETE.json"), resumed_complete=True)
        if not (directory / "expert.json").exists():
            gt_labels, gt_sim = evaluator.score(gt, details=True)
            write_json(directory / "expert.json", dict(labels=gt_labels.tolist(), metadata=meta))
            write_npz(directory / "expert_rollout.npz", simulated_states=gt_sim)
        config = protocol["sampling"]
        unit = config["integer_lattice_m"]

        def phase(name, keys):
            writer.phase(name, keys, gt, evaluator, construct, stop_after_blocks)

        def explore(name, bounds):
            coarse, _, _ = sampler.grid(bounds, config["coarse_m"], unit)
            phase(name + "_coarse", coarse)
            probes = coarse + np.rint(np.asarray(config["offset_probe_m"]) / unit).astype(int)
            limits = np.rint(np.asarray(bounds) / unit).astype(int)
            keep = (probes[:, 0] <= limits[1]) & (probes[:, 1] <= limits[3])
            phase(name + "_probes", probes[keep])

        phase(
            "000_base",
            sampler.grid(
                config["base_dense_x_m"] + config["base_dense_y_m"], config["dense_m"], unit
            )[0],
        )
        bounds = config["initial_x_m"] + config["initial_y_m"]
        explore("001_initial", bounds)
        expansion = []
        for r in range(config["max_expansion_rounds"]):
            target, counts = sampler.expand_bounds(writer.rows, bounds, config)
            expansion.append(dict(bounds_m=list(bounds), high_edge_counts=counts))
            if target == bounds or writer.cap_hit:
                break
            bounds = target
            explore(f"01{r}_expansion", bounds)
        selected, dense_meta = sampler.dense_keys(writer.rows, bounds, config)
        phase("020_dense", selected)
        # Resolve open interior high boundaries with deterministic one-ring growth.
        # Finite domain or candidate cap terminates this loop, including uniform high fields.
        r = 0
        while not writer.cap_hit:
            open_keys = sampler.open_dense_keys(writer.rows, bounds, config)
            if not len(open_keys):
                break
            phase(f"030_revisit_{r:04d}", open_keys)
            r += 1
        phase("040_boundary", sampler.boundary_keys(writer.rows, bounds, config))
        keys = np.asarray(list(writer.rows), dtype=np.int32)
        labels = np.stack(list(writer.rows.values()))
        gt_score = read_json(directory / "expert.json")["labels"][-1]
        outer = sampler.edges(writer.rows, bounds, config["high_pdm_raw"], unit)
        complete = dict(
            record,
            candidates=len(keys),
            blocks=sum(len(list(p.glob("block_*.npz"))) for p in (directory / "phases").iterdir()),
            failed=0,
            phases=writer.phases,
            final_bounds_m=bounds,
            expansion=expansion,
            dense_seed_info=dense_meta,
            candidate_cap_reached=writer.cap_hit,
            outer_high_counts=outer,
            open_dense_neighbors=len(sampler.open_dense_keys(writer.rows, bounds, config)),
            coverage=(
                "candidate_cap"
                if writer.cap_hit
                else ("outer_high_boundary_open" if any(outer) else "requested_sampling_complete")
            ),
            original_gt_pdms=100 * gt_score,
            best_pdms=float(labels[:, -1].max() * 100),
            mean_pdms=float(labels[:, -1].mean() * 100),
            zero_count=int((labels[:, -1] == 0).sum()),
            high_count=int((labels[:, -1] >= 0.95).sum()),
            better_than_gt=int((labels[:, -1] > gt_score + 1e-9).sum()),
            max_endpoint_error_m=writer.endpoint_error,
            new_blocks=writer.new_blocks,
            reused_blocks=writer.reused_blocks,
            elapsed_s=time.perf_counter() - started,
            max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            bytes=sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()),
        )
        write_json(directory / "COMPLETE.json", complete)
        write_json(
            directory / "progress.json",
            dict(status="complete", candidates=len(keys), updated_unix=time.time()),
        )
        return complete
