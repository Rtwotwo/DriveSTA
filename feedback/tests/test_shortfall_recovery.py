"""Supplement isolation, pilot reuse, and unchanged original candidate labels."""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from feedback.tests.test_high_score_bank import run_mock
from drivesta_feedback.batch.shortfall_recovery import (
    distinct_output,
    cohort,
    prepare_scene,
    check_prefix,
    source_snapshot,
    pilot_selection,
    resume_budgets,
    clear_recovered_errors,
)
from drivesta_feedback.batch.store import read_json, write_json
from drivesta_feedback.batch.high_score_bank import read_scene


def record(t="mock"):
    return dict(token=t, split="train", log_name="log", metric_cache_relative_path="unused")


class ShortfallRecoveryTests(unittest.TestCase):
    def test_output_cannot_be_source_ancestor_or_child(self):
        for a, b in [("/tmp/a", "/tmp/a"), ("/tmp/a", "/tmp/a/b"), ("/tmp/a/b", "/tmp/a")]:
            with self.assertRaises(ValueError):
                distinct_output(a, b)
        self.assertEqual(distinct_output("/tmp/a", "/tmp/b")[1], Path("/tmp/b"))

    def test_copy_continue_and_training_read_preserve_source(self):
        with tempfile.TemporaryDirectory() as d:
            source, output = Path(d) / "source", Path(d) / "output"
            run_mock(source, budget=128, retain=True)
            original = source / "scenes/train/mock"
            before = source_snapshot(original)
            prepare_scene(source, output, record())
            run_mock(output, budget=512, retain=True)
            check_prefix(original, output / "scenes/train/mock")
            self.assertEqual(before, source_snapshot(original))
            _, bank = read_scene(output / "scenes/train/mock")
            self.assertEqual(len(bank["labels"]), 256)
            self.assertTrue(np.all(bank["labels"][:, -1] >= 0.95))
            prepare_scene(source, output, record())
            self.assertEqual(read_json(output / "scenes/train/mock/state.json")["count"], 512)

    def test_finished_pilot_rows_are_reused(self):
        with tempfile.TemporaryDirectory() as d:
            source, pilot, output = [Path(d) / n for n in ("source", "pilot", "output")]
            run_mock(source, budget=128, retain=True)
            prepare_scene(source, pilot, record())
            run_mock(pilot, budget=512, retain=True)
            prepare_scene(source, output, record(), seed=pilot)
            self.assertEqual(read_json(output / "scenes/train/mock/state.json")["count"], 512)
            self.assertEqual(
                (pilot / "scenes/train/mock/candidates.npy").read_bytes(),
                (output / "scenes/train/mock/candidates.npy").read_bytes(),
            )
            self.assertNotEqual(
                (pilot / "scenes/train/mock/candidates.npy").stat().st_ino,
                (output / "scenes/train/mock/candidates.npy").stat().st_ino,
            )

    def test_changed_prefix_and_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source, output = Path(d) / "source", Path(d) / "output"
            run_mock(source, budget=128, retain=True)
            prepare_scene(source, output, record())
            p = output / "scenes/train/mock/candidates.npy"
            arr = np.load(p, mmap_mode="r+")
            arr["keys"][0, 0] += 1
            arr.flush()
            arr._mmap.close()
            with self.assertRaisesRegex(ValueError, "prefix"):
                prepare_scene(source, output, record())
            p = source / "scenes/train/mock/expert.json"
            p.write_text(p.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "source changed"):
                prepare_scene(source, output, record())

    def test_cohort_requires_exact_finished_shortfalls(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            tokens = root / "tokens.txt"
            tokens.write_text("a\nb\n")
            write_json(
                root / "PARTIAL.json",
                dict(
                    failed_scenes=0,
                    processed_scenes=3,
                    total_scenes=3,
                    shortfalls=[dict(token=t) for t in ("a", "b")],
                ),
            )
            (root / "scenes.jsonl").write_text(
                "".join(json.dumps(record(t)) + "\n" for t in ("a", "b", "c"))
            )
            _, selected = cohort(root, tokens, 2)
            self.assertEqual([r["token"] for r in selected], ["a", "b"])
            tokens.write_text("a\na\n")
            with self.assertRaises(ValueError):
                cohort(root, tokens, 2)
            tokens.write_text("a\nc\n")
            with self.assertRaises(ValueError):
                cohort(root, tokens, 2)

    def test_pilot_includes_all_four_difficulty_groups(self):
        items = []
        for i, (n, gt, score) in enumerate(
            [(250, 0.9, 99), (0, 0.96, 94), (0, 0.8, 93), (0, 0.8, 0)]
        ):
            for k in range(6):
                items.append(
                    dict(
                        record=record(str(i * 6 + k)),
                        result=dict(candidates=n, best_pdms=score),
                        expert_pdm=gt,
                    )
                )
        selected = pilot_selection(items, 16)
        self.assertEqual(len(selected), 16)
        self.assertEqual(sum(i["result"]["candidates"] > 0 for i in selected), 4)

    def test_resume_never_returns_to_lower_stage(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            scene = output / "scenes/train/mock"
            write_json(scene / "state.json", dict(count=10048))
            self.assertEqual(resume_budgets(output, [8192, 32768], [record()]), [32768])
            with self.assertRaises(ValueError):
                resume_budgets(output, [8192], [record()])
            write_json(scene / "state.json", dict(count=8192))
            write_json(output / "continuation_phase.json", dict(budget=32768, state="running"))
            self.assertEqual(resume_budgets(output, [8192, 32768], [record()]), [32768])
            with self.assertRaises(ValueError):
                resume_budgets(output, [8192], [record()])

    def test_successfully_recovered_shortfall_clears_old_error(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)
            run_mock(output, budget=128, value=0.0, retain=True)
            marker = output / "scenes/train/mock/ERROR.json"
            write_json(marker, dict(error="old invocation"))
            clear_recovered_errors(output, [record()])
            self.assertFalse(marker.exists())
            self.assertTrue((marker.parent / "SHORTFALL.json").exists())


if __name__ == "__main__":
    unittest.main()
