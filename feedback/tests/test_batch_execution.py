import multiprocessing as mp
from pathlib import Path
import tempfile
import unittest
import numpy as np

from drivesta_feedback.batch import sampler
from drivesta_feedback.batch.store import (
    lock,
    read_json,
    load_blocks,
    write_json,
    write_npz,
    atomic,
)
from drivesta_feedback.batch.worker import SceneWriter
from drivesta_feedback.batch.pack import pack_dataset, PackedDataset, ROW_DTYPE

CONFIG = dict(sampling=dict(max_unique_goals=20), storage=dict(chunk_candidates=3))


def fake_construct(gt, keys, protocol):
    t = np.zeros((len(keys), 8, 3), dtype=np.float32)
    t[:, -1, :2] = keys
    return t, 0.0


class FakeEvaluator:
    def score(self, t):
        return np.ones(7, dtype=np.float64) * (0.5 if t[-1, 0] % 2 else 1.0)


def crash(directory):
    with lock(directory):
        writer = SceneWriter(directory, CONFIG, dict(token="synthetic"))
        writer.phase(
            "000_base",
            np.array([[i, 0] for i in range(9)]),
            None,
            FakeEvaluator(),
            fake_construct,
            1,
        )


def adaptive(directory, fail=False):
    with lock(directory):
        protocol = dict(sampling=dict(max_unique_goals=11), storage=dict(chunk_candidates=3))
        writer = SceneWriter(directory, protocol, dict(token="adaptive"))
        writer.phase(
            "000_base", np.array([[i, 0] for i in range(9)]), None, FakeEvaluator(), fake_construct
        )
        proposals = np.array([[key[0], 2] for key, value in writer.rows.items() if value[-1] > 0.5])
        writer.phase(
            "001_refine", proposals, None, FakeEvaluator(), fake_construct, 4 if fail else None
        )
        return writer


class ExecutionTests(unittest.TestCase):
    def test_numpy_metadata_scalars_are_saved_without_changing_values(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "summary.json"
            write_json(p, dict(count=np.int64(12), fraction=np.float64(0.5), passed=np.bool_(True)))
            self.assertEqual(read_json(p), dict(count=12, fraction=0.5, passed=True))

    def test_unsupported_official_sampling_is_rejected(self):
        import json
        from drivesta_feedback.batch.runner import validate_protocol

        # The actual shipped protocol is checked through the saved pilot plan.
        import yaml

        p = Path(__file__).resolve().parents[1] / "configs/dense_dataset_v1.yaml"
        protocol = yaml.safe_load(p.read_text())
        validate_protocol(protocol)
        protocol["labeling"]["simulator_poses"] = 80
        with self.assertRaisesRegex(ValueError, "simulator_poses"):
            validate_protocol(protocol)

    def test_empty_distance_and_no_high_fallback(self):
        self.assertTrue(np.isinf(sampler.distance(np.zeros((5, 5), bool), 0.1)).all())
        c = dict(
            integer_lattice_m=0.05,
            dense_m=0.1,
            coarse_m=0.5,
            high_pdm_raw=0.95,
            high_buffer_m=0.1,
            transition_buffer_m=0.1,
            component_jump_raw=0.05,
        )
        rows = {tuple(k): np.zeros(7) for k in sampler.grid([-1, 1, -1, 1], 0.5)[0]}
        keys, info = sampler.dense_keys(rows, [-1, 1, -1, 1], c)
        self.assertEqual(len(keys), 0)
        self.assertFalse(info["fallback_used"])
        rows[(0, 0)] = np.full(7, 0.5)
        keys, info = sampler.dense_keys(rows, [-1, 1, -1, 1], c)
        self.assertTrue(info["fallback_used"])
        self.assertIn([0, 0], keys.tolist())

    def test_budget_order_is_independent_of_input(self):
        requests = np.array([[5, 1], [2, 2], [1, 1], [5, 1]])
        a, cap = sampler.missing_keys(requests, {}, 2)
        b, _ = sampler.missing_keys(requests[::-1], {}, 2)
        self.assertTrue(cap)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, [[1, 1], [2, 2]])

    def test_committed_block_recovers_after_actual_process_exit(self):
        with tempfile.TemporaryDirectory() as d:
            recovered = Path(d) / "recovered"
            continuous = Path(d) / "continuous"
            child = mp.get_context("spawn").Process(target=crash, args=(recovered,))
            child.start()
            child.join(30)
            self.assertEqual(child.exitcode, 75)
            self.assertFalse((recovered / "progress.json").exists())
            self.assertEqual(len(list(recovered.glob("phases/*/block*"))), 1)
            for directory in (recovered, continuous):
                with lock(directory):
                    writer = SceneWriter(directory, CONFIG, dict(token="synthetic"))
                    writer.phase(
                        "000_base",
                        np.array([[i, 0] for i in range(9)]),
                        None,
                        FakeEvaluator(),
                        fake_construct,
                    )
                    if directory == recovered:
                        self.assertEqual(writer.reused_blocks, 1)
            for (_, a), (_, b) in zip(load_blocks(recovered), load_blocks(continuous)):
                for key in ("keys", "labels", "trajectories"):
                    np.testing.assert_array_equal(a[key], b[key])
            with self.assertRaisesRegex(ValueError, "different protocol"):
                SceneWriter(recovered, dict(CONFIG, version=2), dict(token="synthetic"))

    def test_adaptive_decisions_and_budget_survive_restart(self):
        with tempfile.TemporaryDirectory() as d:
            recovered, continuous = Path(d) / "recovered", Path(d) / "continuous"
            child = mp.get_context("spawn").Process(target=adaptive, args=(recovered, True))
            child.start()
            child.join(30)
            self.assertEqual(child.exitcode, 75)
            a, b = adaptive(recovered), adaptive(continuous)
            self.assertEqual(len(a.rows), 11)
            self.assertTrue(a.cap_hit)
            self.assertEqual(list(a.rows), list(b.rows))
            np.testing.assert_array_equal(
                np.stack(list(a.rows.values())), np.stack(list(b.rows.values()))
            )

    def test_failed_atomic_publish_preserves_previous_file(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "state.json"
            write_json(path, dict(version=1))
            with patch(
                "drivesta_feedback.batch.store.os.replace", side_effect=OSError(28, "no space")
            ):
                with self.assertRaises(OSError):
                    write_json(path, dict(version=2))
            self.assertEqual(read_json(path), dict(version=1))
            self.assertEqual(list(Path(d).glob("*.tmp")), [])

    def test_corrupt_block_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            w = SceneWriter(p, CONFIG, {})
            k = np.array([[0, 0], [1, 0]])
            w.phase("000_base", k, None, FakeEvaluator(), fake_construct)
            target = p / "phases/000_base/block_00000.npz"
            target.write_bytes(b"broken")
            with self.assertRaises(Exception):
                SceneWriter(p, CONFIG, {}).phase(
                    "000_base", k, None, FakeEvaluator(), fake_construct
                )

    def test_pack_resume_and_split_isolation(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            output = Path(d) / "packed"
            reports = []
            for split in ("train", "val"):
                token = split + "_scene"
                p = run / "scenes" / split / token
                w = SceneWriter(p, CONFIG, dict(token=token))
                w.phase(
                    "000_base",
                    np.array([[i, 0] for i in range(5)]),
                    None,
                    FakeEvaluator(),
                    fake_construct,
                )
                meta = dict(
                    token=token,
                    split=split,
                    log_name=split + "_log",
                    candidates=5,
                    phases=[dict(name="000_base")],
                    coverage="test",
                )
                write_json(p / "COMPLETE.json", meta)
                write_json(p / "expert.json", dict(labels=[1] * 7))
                reports.append(meta)
            write_json(run / "run.json", dict(synthetic=True))
            write_json(run / "COMPLETE.json", dict(scenes=reports))
            write_json(run / "audit.json", dict(passed=True, scenes=[dict(official=True)]))
            a = pack_dataset(run, output, max_bytes=6 * ROW_DTYPE.itemsize)
            b = pack_dataset(run, output, max_bytes=6 * ROW_DTYPE.itemsize)
            self.assertEqual(a, b)
            self.assertEqual(a["shards"], 2)
            for split in ("train", "val"):
                dataset = PackedDataset(output, split)
                self.assertEqual(len(dataset), 1)
                info, rows = dataset[0]
                self.assertEqual(info["split"], split)
                np.testing.assert_array_equal(rows["keys"][:, 0], np.arange(5))
                self.assertTrue(all(s % 4 == 0 for s in rows["trajectories"].strides))
                self.assertTrue(all(s % 4 == 0 for s in rows["labels"].strides))


if __name__ == "__main__":
    unittest.main()
