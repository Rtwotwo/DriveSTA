"""Threshold selection, adaptive recovery and explicit incomplete-dataset behavior."""

import copy
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import yaml
from drivesta_feedback.batch.high_score_bank import (
    select_indices,
    save_bank,
    read_scene,
    HighScoreDataset,
)
from drivesta_feedback.batch.high_score_sampler import next_request
from drivesta_feedback.batch.high_score_scene import generate
from drivesta_feedback.batch.production import (
    validate_config,
    load_configuration,
    estimate,
    run_job,
    progress,
)
from drivesta_feedback.batch.scene_store import ROW_DTYPE, SceneTable
from drivesta_feedback.batch.store import read_json, write_json

ROOT = Path(__file__).resolve().parents[2]


def configuration():
    c = yaml.safe_load((ROOT / "feedback/configs/high_score_256.yaml").read_text())
    base = yaml.safe_load((ROOT / "feedback/configs/dense_dataset_v1.yaml").read_text())
    base["sampling"], base["storage"] = copy.deepcopy(c["sampling"]), copy.deepcopy(c["storage"])
    return c, base


GT = np.column_stack((np.arange(1, 9), np.zeros((8, 2)))).astype(float)


def construct(gt, keys, protocol):
    u = np.linspace(0, 1, 8)
    beta = 10 * u**3 - 15 * u**4 + 6 * u**5
    p = np.broadcast_to(gt, (len(keys), 8, 3)).copy()
    p[:, :, :2] += (
        beta[None, :, None]
        * (np.asarray(keys) * protocol["sampling"]["integer_lattice_m"])[:, None, :]
    )
    return p.astype(np.float32), 0.0


class FakeEvaluator:
    fail_once = False
    calls = 0

    def __init__(self, cache):
        self.value = cache

    def score(self, path, details=False):
        type(self).calls += 1
        if self.fail_once and self.calls == 80:
            raise RuntimeError("injected score failure")
        y = np.ones(7)
        y[5] = 0.0  # Official weight is zero, so a low DDC remains a valid stored label.
        y[2] = self.value
        y[6] = (5 * y[2] + 7) / 12
        return (y, None) if details else y


def run_mock(path, budget=512, value=1.0, token="mock", crash=False, retain=False, backfill=False):
    c, p = configuration()
    c["_query_budget"] = budget
    c["storage"]["retain_all_scores"] = retain
    p["storage"][
        "retain_all_scores"
    ] = False  # Preserve the legacy scene protocol in resume fixtures.
    c["_backfill_full_scores"] = backfill
    r = dict(token=token, split="train", log_name="log", metric_cache_relative_path="unused")
    meta = dict(r, expert_gt=GT.tolist())
    with (
        patch("drivesta_feedback.batch.source.load_scene", return_value=(None, GT, value, meta)),
        patch("drivesta_feedback.batch.source.construct", side_effect=construct),
        patch("drivesta_feedback.batch.source.Evaluator", FakeEvaluator),
    ):
        if crash:
            original = SceneTable.append

            def append(self, *args, **kwargs):
                return original(self, *args, **kwargs, after_flush=lambda: os._exit(77))

            with patch.object(SceneTable, "append", append):
                return generate(r, [token], path, p, c, "test")
        return generate(r, [token], path, p, c, "test")


class HighScoreTests(unittest.TestCase):
    def test_raw_threshold_sort_ties_duplicates_and_gt(self):
        p = np.repeat(GT[None], 7, axis=0).astype(np.float32)
        p[:, -1, 0] += np.arange(7)
        p[6] = p[2]
        y = np.ones((7, 7))
        y[:, -1] = [1.0, 0.949999999, 0.95, 0.98, 0.98, 0.97, 1.0]
        ids = select_indices(p, y, GT)
        self.assertEqual(ids.tolist(), [6, 3, 4, 5])
        y[6, -1] = 0.95
        self.assertEqual(select_indices(p, y, GT).tolist(), [3, 4, 5, 2])

    def test_no_padding_or_rounding(self):
        p = np.repeat(GT[None], 256, axis=0).astype(np.float32)
        p[:, -1, 0] += 1
        self.assertEqual(len(select_indices(p, np.ones((256, 7)), GT)), 1)
        y = np.ones((256, 7))
        y[:, -1] = 0.94999999
        self.assertEqual(len(select_indices(p, y, GT)), 0)

    def test_full_query_budget_before_selecting_and_direct_reader(self):
        with tempfile.TemporaryDirectory() as d:
            result = run_mock(d)
            self.assertEqual(result["queries"], 512)
            self.assertEqual(result["audited_candidates"], 256)
            directory = Path(d) / "scenes/train/mock"
            _, a = read_scene(directory, verify=True)
            self.assertEqual(a["labels"].shape, (256, 7))
            self.assertTrue((a["labels"][:, 5] == 0).all())
            self.assertFalse((directory / "candidates.npy").exists())
            self.assertEqual(run_mock(d, budget=1024)["queries"], 512)

    def test_shortfall_preserves_queries_and_higher_budget_continues(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_mock(d, budget=128)
            self.assertEqual((r["status"], r["missing_candidates"]), ("shortfall", 128))
            directory = Path(d) / "scenes/train/mock"
            self.assertFalse((directory / "COMPLETE.json").exists())
            self.assertTrue((directory / "candidates.npy").exists())
            r = run_mock(d, budget=512)
            self.assertEqual(r["invocation"]["reused_queries"], 128)
            self.assertEqual(r["queries"], 512)
            self.assertTrue(r["passed"])

    def test_low_score_scene_never_gets_complete(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_mock(d, value=0.0)
            self.assertEqual(r["missing_candidates"], 256)
            directory = Path(d) / "scenes/train/mock"
            self.assertFalse((directory / "high_score.npz").exists())
            with self.assertRaises(FileNotFoundError):
                read_scene(directory)

    def test_resume_real_process_death_after_flush(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a", Path(d) / "b"
            child = mp.get_context("spawn").Process(
                target=run_mock, args=(a, 512, 1.0, "mock", True)
            )
            child.start()
            child.join(45)
            if child.is_alive():
                child.terminate()
                child.join()
                self.fail("Crash test timed out")
            self.assertEqual(child.exitcode, 77)
            run_mock(a)
            run_mock(b)
            _, x = read_scene(a / "scenes/train/mock")
            _, y = read_scene(b / "scenes/train/mock")
            for k in x:
                np.testing.assert_array_equal(x[k], y[k])

    def test_score_failure_resumes_uncommitted_batch(self):
        with tempfile.TemporaryDirectory() as d:
            FakeEvaluator.calls = 0
            FakeEvaluator.fail_once = True
            try:
                with self.assertRaises(RuntimeError):
                    run_mock(d)
            finally:
                FakeEvaluator.fail_once = False
            r = run_mock(d)
            self.assertEqual(r["invocation"]["reused_queries"], 64)

    def test_deterministic_distinct_goals_and_all_zero_exploration(self):
        c, p = configuration()
        rows = np.zeros(0, dtype=ROW_DTYPE)
        a = next_request(rows, c, "scene", 3072)
        self.assertEqual(a, next_request(rows, c, "scene", 3072))
        self.assertEqual(len({tuple(k) for k in a["keys"]}), 64)
        rows = np.zeros(64, dtype=ROW_DTYPE)
        rows["keys"] = a["keys"]
        b = next_request(rows, c, "scene", 3072)
        self.assertFalse({tuple(k) for k in a["keys"]} & {tuple(k) for k in b["keys"]})

    def test_few_positive_centers_do_not_include_zero_score_fillers(self):
        c, _ = configuration()
        c["sampling"]["exploration_fraction"] = 0.01
        c["sampling"]["local_scales_m"] = [0.01]
        rows = np.zeros(64, dtype=ROW_DTYPE)
        rows["keys"][:, 0] = np.arange(64) * 100
        rows["keys"][0] = [10000, 0]
        rows["labels"][0, -1] = 0.8
        req = next_request(rows, c, "sparse", 3072)
        local = np.asarray(req["keys"][1:]) * 0.001
        self.assertTrue((np.linalg.norm(local - [10, 0], axis=1) < 0.1).all())

    def test_cli_scan_cleans_staging_after_complete_publication(self):
        from drivesta_feedback.batch.production import scan_completed

        with tempfile.TemporaryDirectory() as d:
            run_mock(d)
            directory = Path(d) / "scenes/train/mock"
            # Simulate interruption after bank completion and before staging cleanup.
            (directory / "candidates.npy").write_bytes(b"old staging")
            write_json(directory / "state.json", dict(count=512))
            r = dict(
                token="mock", split="train", log_name="log", metric_cache_relative_path="unused"
            )
            completed = scan_completed(Path(d), [r], contract_id="test")
            self.assertEqual(len(completed), 1)
            self.assertFalse((directory / "candidates.npy").exists())
            self.assertFalse((directory / "state.json").exists())
            read_scene(directory, verify=True)

    def test_explicit_partial_dataset_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            run_mock(d)
            root = Path(d)
            (root / "scenes.jsonl").write_text(
                '{"token":"mock","split":"train"}\n{"token":"absent","split":"train"}\n'
            )
            with self.assertRaises(FileNotFoundError):
                HighScoreDataset(root)
            ds = HighScoreDataset(root, allow_partial=True)
            self.assertEqual(ds.missing_tokens, ["absent"])
            self.assertEqual(ds.requested_scenes, 2)
            self.assertEqual(len(ds), 1)
            self.assertEqual(ds[0][1]["labels"].shape, (256, 7))

    def test_config_and_cost_calls_include_all_retained_audits(self):
        c, p = configuration()
        validate_config(c)
        e = estimate(c, 103288, 128)
        self.assertEqual(e["max_official_calls_including_audit"], 103288 * (3072 + 257))
        self.assertFalse(e["qualified_scene_coverage_guaranteed"])
        c["sampling"]["high_pdm_raw"] = 0.949
        with self.assertRaises(ValueError):
            validate_config(c)

    def test_shortfalls_are_not_runtime_errors(self):
        import time

        p = progress({}, {}, 0, 1, time.monotonic() - 2, 128, shortfalls={"s": {"candidates": 2}})
        self.assertEqual(p["shortfall_scenes"], 1)
        self.assertEqual(p["failed_scenes"], 0)
        self.assertEqual(p["eligible_scene_fraction"], 0)

    def test_queue_publishes_partial_then_resumes_only_shortfall(self):
        from concurrent.futures import ThreadPoolExecutor
        from drivesta_feedback.batch.production import execute

        c, _ = configuration()
        jobs = [
            dict(job_id=str(i), split="train", log_name="log", scenes=[dict(token=str(i))])
            for i in range(2)
        ]

        def make_pool(**kwargs):
            return ThreadPoolExecutor(max_workers=kwargs["max_workers"])

        def fake(job, *args):
            token = job["scenes"][0]["token"]
            return [
                dict(
                    token=token,
                    passed=token == "0",
                    status="complete" if token == "0" else "shortfall",
                    candidates=256 if token == "0" else 120,
                )
            ]

        with (
            tempfile.TemporaryDirectory() as d,
            patch("drivesta_feedback.batch.production.ProcessPoolExecutor", side_effect=make_pool),
            patch("drivesta_feedback.batch.production.run_job", side_effect=fake),
            patch("builtins.print"),
        ):
            complete = {}
            result = execute(jobs, Path(d), {}, c, "test", complete, 2)
            self.assertFalse(result["passed"])
            self.assertEqual(result["shortfall_scenes"], 1)
            self.assertTrue((Path(d) / "PARTIAL.json").exists())
            self.assertFalse((Path(d) / "COMPLETE.json").exists())

            def finish(job, *args):
                self.assertEqual(job["scenes"][0]["token"], "1")
                return [dict(token="1", passed=True, candidates=256)]

            with patch("drivesta_feedback.batch.production.run_job", side_effect=finish):
                result = execute(jobs, Path(d), {}, c, "test", complete, 2)
            self.assertTrue(result["passed"])
            self.assertFalse((Path(d) / "PARTIAL.json").exists())


if __name__ == "__main__":
    unittest.main()
