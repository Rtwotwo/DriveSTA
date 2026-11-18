import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from drivesta_feedback.batch.gt_recovery import refinement_keys, run_scene
from drivesta_feedback.batch.scene_store import ROW_DTYPE
from drivesta_feedback.batch.store import write_json
from drivesta_feedback.batch.retained_bank import read_full_scene
from drivesta_feedback.batch.high_score_bank import read_scene


CFG = dict(
    base_rings_m=[[0.02, 0.01]],
    micro_rings_m=[[0.01, 0.001]],
    seed_rings_m=[[0.008, 0.001]],
    refine_min_raw=0.94,
    max_offset_m=2.0,
    max_seeds=4,
    seed_separation_m=0.02,
    max_queries=600,
)


class FakeEvaluator:
    calls = 0
    fail_at = None
    low = False

    def __init__(self, cache):
        self.cache = cache

    def score(self, t, details=False):
        type(self).calls += 1
        if type(self).calls == type(self).fail_at:
            raise RuntimeError("injected mid-batch scoring interruption")
        label = np.ones(7, dtype=np.float64)
        if type(self).low:
            label[2] = 0
            label[-1] = 7 / 12
        return (label, None) if details else label


def construct(gt, keys, protocol):
    u = np.linspace(0, 1, 8)
    beta = 10 * u**3 - 15 * u**4 + 6 * u**5
    paths = np.broadcast_to(gt, (len(keys), 8, 3)).copy()
    paths[:, :, :2] += beta[None, :, None] * (np.asarray(keys) * 0.001)[:, None, :]
    return paths.astype(np.float32), 0.0


class RecoveryTests(unittest.TestCase):
    def test_refinement_is_bounded_unique_deterministic(self):
        rows = np.zeros(3, dtype=ROW_DTYPE)
        rows["keys"] = [[0, 0], [2000, 2000], [-2000, -2000]]
        rows["labels"][:, -1] = [0.96, 0.99, 0.97]
        config = dict(CFG, max_queries=100)
        keys = refinement_keys(rows, 0.8, config, 0.001)
        self.assertEqual(len(keys), 97)
        self.assertEqual(len(set(map(tuple, keys))), len(keys))
        self.assertFalse(set(map(tuple, rows["keys"])) & set(map(tuple, keys)))
        self.assertLessEqual(abs(keys).max(), 2000)
        np.testing.assert_array_equal(keys, refinement_keys(rows, 0.8, config, 0.001))

    def test_no_promising_region_finishes_base(self):
        rows = np.zeros(2, dtype=ROW_DTYPE)
        rows["labels"][:, -1] = 0.9
        self.assertEqual(refinement_keys(rows, 0.91, CFG, 0.001).shape, (0, 2))
        self.assertGreater(len(refinement_keys(rows, 0.95, CFG, 0.001)), 0)

    def fixture(self, root):
        record = dict(token="test123", split="train", log_name="test")
        gt = np.zeros((8, 3))
        gt[:, 0] = np.linspace(0.5, 4.0, 8)
        metadata = dict(record, expert_gt=gt.tolist())
        protocol = dict(
            constructor=dict(max_planned_endpoint_error_m=1e-3),
            sampling=dict(integer_lattice_m=0.001),
            storage={},
            labeling=dict(float64_parity_tolerance=1e-9),
        )
        config = dict(CFG, source=str(root / "source"), reuse_root=str(root / "no_pilot"))
        source = root / "source/scenes/train/test123"
        write_json(source / "contract.json", dict(identity=metadata, protocol=protocol))
        item = dict(record=record, token=record["token"])
        return item, config, gt, metadata

    def test_complete_bank_and_repeat_read_no_rescoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item, config, gt, meta = self.fixture(root)
            FakeEvaluator.calls, FakeEvaluator.fail_at, FakeEvaluator.low = 0, None, False
            with (
                patch(
                    "drivesta_feedback.batch.source.load_scene", return_value=(None, gt, None, meta)
                ),
                patch("drivesta_feedback.batch.source.construct", side_effect=construct),
                patch("drivesta_feedback.batch.source.Evaluator", FakeEvaluator),
            ):
                result = run_scene(item, config, root / "out", "run1")
                self.assertTrue(result["passed"])
                self.assertEqual(result["audited_candidates"], 256)
                self.assertLess(result["queries"], config["max_queries"])
                d = root / "out/scenes/train/test123"
                _, full = read_full_scene(d)
                _, high = read_scene(d, verify=True)
                np.testing.assert_array_equal(high["labels"], full["labels"][high["query_indices"]])
                before = FakeEvaluator.calls
                self.assertEqual(run_scene(item, config, root / "out", "run1"), result)
                self.assertEqual(FakeEvaluator.calls, before)

    def test_resume_interrupted_refinement_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item, config, gt, meta = self.fixture(root)
            FakeEvaluator.calls, FakeEvaluator.fail_at, FakeEvaluator.low = 0, 40, False
            with (
                patch(
                    "drivesta_feedback.batch.source.load_scene", return_value=(None, gt, None, meta)
                ),
                patch("drivesta_feedback.batch.source.construct", side_effect=construct),
                patch("drivesta_feedback.batch.source.Evaluator", FakeEvaluator),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    run_scene(item, config, root / "resumed", "run1")
                FakeEvaluator.fail_at = None
                result = run_scene(item, config, root / "resumed", "run1")
                expected = run_scene(item, config, root / "clean", "run1")
                self.assertEqual(result["queries"], expected["queries"])
                _, a = read_full_scene(root / "resumed/scenes/train/test123")
                _, b = read_full_scene(root / "clean/scenes/train/test123")
                for k in ("keys", "trajectories", "labels"):
                    np.testing.assert_array_equal(a[k], b[k])

    def test_shortfall_preserves_all_low_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item, config, gt, meta = self.fixture(root)
            FakeEvaluator.calls, FakeEvaluator.fail_at, FakeEvaluator.low = 0, None, True
            with (
                patch(
                    "drivesta_feedback.batch.source.load_scene", return_value=(None, gt, None, meta)
                ),
                patch("drivesta_feedback.batch.source.construct", side_effect=construct),
                patch("drivesta_feedback.batch.source.Evaluator", FakeEvaluator),
            ):
                result = run_scene(item, config, root / "out", "run1")
                self.assertFalse(result["passed"])
                self.assertEqual(result["queries"], 25)
                _, full = read_full_scene(root / "out/scenes/train/test123")
                self.assertEqual(len(full["labels"]), 25)
                self.assertFalse((root / "out/scenes/train/test123/high_score.npz").exists())
                np.testing.assert_array_equal(full["labels"][:, -1], np.full(25, 7 / 12))


if __name__ == "__main__":
    unittest.main()
