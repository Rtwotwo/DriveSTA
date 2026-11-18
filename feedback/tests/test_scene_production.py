"""Production sampler, direct table, crash recovery and scheduler contracts."""

import copy
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import yaml
from drivesta_feedback.batch.budget_sampler import grid, next_request, coverage
from drivesta_feedback.batch.scene_store import (
    SceneTable,
    SceneDataset,
    ROW_DTYPE,
    read_scene,
    validate_rows,
)
from drivesta_feedback.batch.store import checksum, read_json, write_json
from drivesta_feedback.batch.production import validate_config, estimate, scan_completed, execute

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/navtrainval_030.yaml"


def configuration(budget=3072):
    c = yaml.safe_load(CONFIG_PATH.read_text())
    c["sampling"]["max_unique_goals"] = budget
    c["storage"]["chunk_candidates"] = 64
    return c


def synthetic(keys):
    gt = np.column_stack((np.arange(1, 9), np.zeros((8, 2)))).astype(float)
    u = np.linspace(0, 1, 8)
    beta = 10 * u**3 - 15 * u**4 + 6 * u**5
    paths = np.broadcast_to(gt, (len(keys), 8, 3)).copy()
    paths[:, :, :2] += beta[None, :, None] * (np.asarray(keys) * 0.05)[:, None, :]
    labels = np.ones((len(keys), 7), dtype=float)
    labels[:, 1] = np.asarray(keys)[:, 1] <= 0
    labels[:, 2] = np.clip(0.7 + np.asarray(keys)[:, 0] / 600, 0, 1)
    labels[:, 6] = labels[:, 1] * (5 * labels[:, 2] + 7) / 12
    return gt, paths.astype(np.float32), labels


def run_synthetic(directory, budget=1800, crash=False):
    c = configuration(budget)
    table = SceneTable(directory, dict(identity=dict(token="unit")), budget)
    try:
        while len(table.rows) < budget:
            if table.state["pending"] is None:
                table.begin(next_request(table.rows, table.state["completed_stages"], c, "unit"))
            p = table.state["pending"]
            while p["offset"] < len(p["keys"]):
                keys = np.array(p["keys"][p["offset"] : p["offset"] + 64], dtype=np.int32)
                _, paths, labels = synthetic(keys)
                hook = (lambda: os._exit(76)) if crash and len(table.rows) >= 1536 else None
                table.append(keys, paths, labels, after_flush=hook)
            table.end()
        if table.state["pending"] is not None:
            table.end()
        return table.rows.copy(), copy.deepcopy(table.state)
    finally:
        table.close()


class ProductionTests(unittest.TestCase):
    def test_global_grid_origin_and_domain(self):
        c = configuration()["sampling"]
        points = grid(c["hard_bounds_m"], 0.3)
        self.assertTrue((points % 6 == 0).all())
        self.assertIn([0, 0], points.tolist())
        self.assertEqual(len(points), 184 * 81)
        for stride, offset in ((1.2, 0.0), (1.2, 0.6)):
            coarse = grid(c["hard_bounds_m"], stride, offset=offset)
            self.assertTrue({tuple(k) for k in coarse} <= {tuple(k) for k in points})

    def test_same_scene_sampling_is_reproducible_and_retains_low_scores(self):
        with tempfile.TemporaryDirectory() as d:
            a, state = run_synthetic(Path(d) / "a")
            b, _ = run_synthetic(Path(d) / "b")
            np.testing.assert_array_equal(a, b)
            self.assertEqual(len(np.unique(a["keys"], axis=0)), len(a))
            self.assertTrue((a["labels"][:, -1] == 0).any())
            self.assertTrue((a["labels"][:, -1] >= 0.95).any())
            self.assertTrue(any(s.startswith("adaptive") for s in state["completed_stages"]))
            self.assertEqual(sum((a["keys"] == 0).all(axis=1)), 1)

    def test_actual_process_crash_after_flush_recovers_identical_adaptive_data(self):
        with tempfile.TemporaryDirectory() as d:
            recovered, continuous = Path(d) / "resumed", Path(d) / "continuous"
            child = mp.get_context("spawn").Process(
                target=run_synthetic, args=(recovered, 1800, True)
            )
            child.start()
            child.join(45)
            if child.is_alive():
                child.terminate()
                child.join()
                self.fail("Recovery test timed out")
            self.assertEqual(child.exitcode, 76)
            a, state_a = run_synthetic(recovered)
            b, state_b = run_synthetic(continuous)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(state_a, state_b)

    def test_coverage_reports_budget_and_unmeasured_domain(self):
        with tempfile.TemporaryDirectory() as d:
            rows, _ = run_synthetic(d, 32)
            info = coverage(rows, configuration(32))
            self.assertTrue(info["budget_reached"])
            self.assertFalse(info["complete_domain_grid"])
            self.assertGreater(info["unmeasured"], 14000)

    def test_all_zero_scene_keeps_exploring(self):
        c = configuration()
        rows = np.zeros(1, dtype=ROW_DTYPE)
        req = next_request(rows, ["base", "coarse", "offset"], c, "zero")
        self.assertEqual(len(req["keys"]), 256)
        self.assertEqual(len(np.unique(req["keys"], axis=0)), 256)
        keys = np.asarray(req["keys"])
        self.assertTrue((keys[:, 0] > 300).any())
        self.assertTrue((keys[:, 0] < -100).any())

    def test_corrupt_committed_data_refused(self):
        with tempfile.TemporaryDirectory() as d:
            run_synthetic(d, 32)
            mm = np.load(Path(d) / "candidates.npy", mmap_mode="r+")
            mm["labels"][0, 0] = 0.123
            mm.flush()
            del mm
            with self.assertRaisesRegex(ValueError, "corrupt"):
                SceneTable(d, dict(identity=dict(token="unit")), 32)

    def test_failed_commit_does_not_advance_durable_count(self):
        with tempfile.TemporaryDirectory() as d:
            table = SceneTable(d, {}, 8)
            keys = np.array([[0, 0], [6, 0]], dtype=np.int32)
            table.begin(dict(name="base", keys=keys.tolist()))
            _, paths, labels = synthetic(keys)
            with patch(
                "drivesta_feedback.batch.scene_store.write_json",
                side_effect=OSError(28, "disk full"),
            ):
                with self.assertRaises(OSError):
                    table.append(keys, paths, labels)
            table.close()
            resumed = SceneTable(d, {}, 8)
            self.assertEqual(len(resumed.rows), 0)
            resumed.append(keys, paths, labels)
            resumed.end()
            np.testing.assert_array_equal(resumed.rows["labels"], labels)
            resumed.close()

    def test_reader_uses_committed_prefix_and_separates_splits(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = []
            for split in ("train", "val"):
                directory = root / "scenes" / split / split
                record = dict(
                    split=split,
                    token=split,
                    log_name=split + "_log",
                    metric_cache_relative_path="cache",
                )
                table = SceneTable(directory, dict(identity=record), 16)
                keys = np.array([[0, 0], [6, 6]], dtype=np.int32)
                table.begin(dict(name="base", keys=keys.tolist()))
                gt, paths, labels = synthetic(keys)
                table.append(keys, paths, labels)
                table.end()
                validate_rows(
                    table.rows,
                    gt,
                    dict(
                        sampling=dict(integer_lattice_m=0.05),
                        constructor=dict(max_planned_endpoint_error_m=3e-5),
                    ),
                )
                table.close()
                write_json(
                    directory / "COMPLETE.json",
                    dict(
                        passed=True,
                        candidates=2,
                        data_digest=checksum(directory / "candidates.npy"),
                    ),
                )
                manifest.append(record)
                _, loaded = read_scene(directory, verify=True)
                self.assertEqual(len(loaded), 2)
                np.testing.assert_array_equal(loaded["labels"], labels)
                del loaded
            import json

            (root / "scenes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest))
            write_json(root / "COMPLETE.json", dict(passed=True))
            self.assertEqual(len(SceneDataset(root, "train")), 1)
            meta, rows = SceneDataset(root, "val")[0]
            self.assertEqual(meta["split"], "val")
            del rows
            self.assertEqual(len(scan_completed(root, manifest)), 2)

    def test_config_rejects_high_only_and_nonlattice_sampling(self):
        c = configuration()
        validate_config(c)
        c["storage"]["retain_all_scores"] = False
        with self.assertRaises(ValueError):
            validate_config(c)
        c = configuration()
        c["sampling"]["coarse_m"] = 1.0
        with self.assertRaises(ValueError):
            validate_config(c)

    def test_estimate_includes_audit_and_flags_unmeasured_workers(self):
        e = estimate(configuration(), 103288, 128)
        self.assertEqual(e["max_candidate_labels"], 317300736)
        self.assertAlmostEqual(e["scoring_hours"], 34.7443, places=2)
        self.assertGreater(e["trajectory_table_gb"], 50)
        self.assertIsNone(estimate(configuration(), 10, 64)["scoring_hours"])

    def test_scene_pipeline_synthetic_wiring_audit_and_completed_resume(self):
        import types
        from drivesta_feedback.batch.production_scene import generate

        c = configuration(1800)
        gt, _, _ = synthetic(np.array([[0, 0]]))
        record = dict(
            token="unit", split="train", log_name="unit_log", metric_cache_relative_path="unused"
        )
        calls = []

        def load(record, tokens, protocol):
            return None, gt, None, dict(record, expert_gt=gt.tolist())

        def construct(gt, keys, protocol):
            _, paths, _ = synthetic(keys)
            return paths, 0.0

        class Evaluator:
            def __init__(self, cache):
                pass

            def score(self, trajectory, details=False):
                keys = np.round((trajectory[-1, :2] - gt[-1, :2]) / 0.05).astype(int)[None]
                _, _, values = synthetic(keys)
                calls.append(details)
                return (values[0], np.zeros((41, 11))) if details else values[0]

        source = types.SimpleNamespace(load_scene=load, construct=construct, Evaluator=Evaluator)
        p = dict(
            sampling=c["sampling"],
            constructor=dict(max_planned_endpoint_error_m=3e-5),
            labeling=dict(float64_parity_tolerance=1e-9),
        )
        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict("sys.modules", {"drivesta_feedback.batch.source": source}),
        ):
            result = generate(record, ["unit"], d, p, c, "contract")
            self.assertTrue(result["passed"])
            self.assertEqual(result["candidates"], 1800)
            self.assertGreaterEqual(sum(calls), 3)
            count = len(calls)
            again = generate(record, ["unit"], d, p, c, "contract")
            self.assertEqual(again, result)
            self.assertEqual(len(calls), count)

    def test_completed_prefix_does_not_allow_duplicate_append(self):
        with tempfile.TemporaryDirectory() as d:
            table = SceneTable(d, {}, 8)
            keys = np.array([[0, 0]], dtype=np.int32)
            table.begin(dict(name="base", keys=keys.tolist()))
            _, paths, labels = synthetic(keys)
            table.append(keys, paths, labels)
            table.end()
            with self.assertRaisesRegex(ValueError, "repeats"):
                table.begin(dict(name="duplicate", keys=keys.tolist()))
            table.close()

    def test_scheduler_skips_completed_scenes_and_publishes_only_all_success(self):
        from concurrent.futures import ThreadPoolExecutor

        jobs = [
            dict(
                job_id=str(i),
                split="train",
                log_name="log_" + str(i),
                scenes=[
                    dict(token=str(i), scene_type="unknown", metric_cache_relative_path="unused")
                ],
            )
            for i in range(3)
        ]
        calls = []

        def make_pool(**kwargs):
            return ThreadPoolExecutor(max_workers=kwargs["max_workers"])

        def fake_job(job, *args):
            calls.append(job["scenes"][0]["token"])
            return [
                dict(token=r["token"], split="train", passed=True, candidates=32)
                for r in job["scenes"]
            ]

        with (
            tempfile.TemporaryDirectory() as d,
            patch("drivesta_feedback.batch.production.ProcessPoolExecutor", side_effect=make_pool),
            patch("drivesta_feedback.batch.production.run_job", side_effect=fake_job),
            patch("builtins.print"),
        ):
            completed = {"0": dict(token="0", split="train", passed=True, candidates=32)}
            execute(jobs, Path(d), {}, configuration(32), "test", completed, workers=2)
            self.assertEqual(sorted(calls), ["1", "2"])
            result = read_json(Path(d) / "COMPLETE.json")
            self.assertEqual(result["completed_scenes"], 3)
            self.assertEqual(result["complete_candidates"], 96)

    def test_scheduler_failure_leaves_dataset_incomplete(self):
        from concurrent.futures import ThreadPoolExecutor

        jobs = [dict(job_id="0", split="train", log_name="log", scenes=[dict(token="0")])]

        def make_pool(**kwargs):
            return ThreadPoolExecutor(max_workers=kwargs["max_workers"])

        with (
            tempfile.TemporaryDirectory() as d,
            patch("drivesta_feedback.batch.production.ProcessPoolExecutor", side_effect=make_pool),
            patch(
                "drivesta_feedback.batch.production.run_job",
                return_value=[dict(token="0", split="train", passed=False)],
            ),
            patch("builtins.print"),
        ):
            with self.assertRaisesRegex(RuntimeError, "scenes failed"):
                execute(jobs, Path(d), {}, configuration(32), "test", {}, workers=1)
            self.assertFalse((Path(d) / "COMPLETE.json").exists())


if __name__ == "__main__":
    unittest.main()
