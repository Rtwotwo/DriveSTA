"""Retention, replay equivalence, publication failures and old-run compatibility."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from feedback.tests.test_high_score_bank import run_mock, FakeEvaluator
from drivesta_feedback.batch.retained_bank import (
    read_full_scene,
    FullScoreDataset,
    resolve_run_contract,
)
from drivesta_feedback.batch.high_score_bank import read_scene, collate_high_score
from drivesta_feedback.batch.production import scan_completed, require_high_banks_complete
from drivesta_feedback.batch.store import read_json, write_json


class RetainedBankTests(unittest.TestCase):
    def test_retention_and_training_batch_preserve_all_rows_and_subset(self):
        with tempfile.TemporaryDirectory() as d:
            run_mock(d, retain=True)
            path = Path(d) / "scenes/train/mock"
            meta, full = read_full_scene(path)
            _, high = read_scene(path)
            self.assertEqual(full["labels"].shape, (512, 7))
            for k in ("keys", "labels", "trajectories"):
                np.testing.assert_array_equal(full[k][high["query_indices"]], high[k])
            batch = collate_high_score([(meta, full)])
            self.assertEqual(tuple(batch["scorer_trajectories"].shape), (1, 512, 8, 3))
            run_mock(d, retain=True)
            self.assertTrue((path / "candidates.npy").exists())

    def test_low_and_zero_scores_are_kept_without_high_bank(self):
        with tempfile.TemporaryDirectory() as d:
            run_mock(d, value=0.0, retain=True)
            path = Path(d) / "scenes/train/mock"
            _, full = read_full_scene(path)
            self.assertTrue((full["labels"][:, -1] < 0.95).all())
            self.assertFalse((path / "COMPLETE.json").exists())
            self.assertEqual(len(full["labels"]), 512)

        class ZeroEvaluator(FakeEvaluator):
            def score(self, path, details=False):
                y = np.ones(7)
                y[0] = y[-1] = 0.0
                return (y, None) if details else y

        with (
            tempfile.TemporaryDirectory() as d,
            patch("feedback.tests.test_high_score_bank.FakeEvaluator", ZeroEvaluator),
        ):
            run_mock(d, retain=True)
            _, full = read_full_scene(Path(d) / "scenes/train/mock")
            self.assertTrue((full["labels"][:, -1] == 0).all())

    def test_extension_republishes_valid_count(self):
        with tempfile.TemporaryDirectory() as d:
            run_mock(d, budget=128, retain=True)
            path = Path(d) / "scenes/train/mock"
            _, old = read_full_scene(path)
            result = run_mock(d, budget=512, retain=True)
            self.assertEqual(result["invocation"]["reused_queries"], 128)
            _, new = read_full_scene(path)
            np.testing.assert_array_equal(old["labels"], new["labels"][:128])
            self.assertEqual(read_json(path / "ALL_SCORES.json")["candidates"], 512)

    def test_failed_high_publication_is_recoverable(self):
        with tempfile.TemporaryDirectory() as d:

            def fail(path, value):
                if Path(path).name == "COMPLETE.json":
                    raise OSError("injected publication failure")
                write_json(path, value)

            with patch("drivesta_feedback.batch.high_score_scene.write_json", side_effect=fail):
                with self.assertRaises(OSError):
                    run_mock(d, retain=True)
            run_mock(d, retain=True)
            read_full_scene(Path(d) / "scenes/train/mock")
            read_scene(Path(d) / "scenes/train/mock")

    def test_backfill_preserves_old_bank_and_original_budget(self):
        with tempfile.TemporaryDirectory() as d:
            run_mock(d)
            path = Path(d) / "scenes/train/mock"
            before = {k: (path / k).read_bytes() for k in ("high_score.npz", "COMPLETE.json")}
            run_mock(d, budget=1024, retain=True, backfill=True)
            _, full = read_full_scene(path)
            self.assertEqual(len(full["labels"]), 512)
            for k, v in before.items():
                self.assertEqual((path / k).read_bytes(), v)

    def test_backfill_mismatch_does_not_publish_or_overwrite_high(self):
        with tempfile.TemporaryDirectory() as d:
            run_mock(d)
            path = Path(d) / "scenes/train/mock"
            before = (path / "high_score.npz").read_bytes()
            with self.assertRaisesRegex(ValueError, "Replay"):
                run_mock(d, value=0.9, retain=True, backfill=True)
            self.assertFalse((path / "ALL_SCORES.json").exists())
            self.assertEqual((path / "high_score.npz").read_bytes(), before)
            write_json(path / "ERROR.json", dict(error="replay label mismatch"))
            r = dict(
                token="mock", split="train", log_name="log", metric_cache_relative_path="unused"
            )
            scan_completed(Path(d), [r], contract_id="test", retain_all_scores=True, full_only=True)
            self.assertTrue((path / "ERROR.json").exists())
            self.assertTrue((path / "candidates.npy").exists())

    def test_parallel_scan_matches_serial_and_keeps_full_data(self):
        with tempfile.TemporaryDirectory() as d:
            rows = []
            for token in ("a", "b"):
                run_mock(d, retain=True, token=token)
                rows.append(
                    dict(
                        token=token,
                        split="train",
                        log_name="log",
                        metric_cache_relative_path="unused",
                    )
                )
            x = scan_completed(Path(d), rows, contract_id="test", retain_all_scores=True)
            y = scan_completed(
                Path(d), rows, contract_id="test", retain_all_scores=True, scan_workers=2
            )
            self.assertEqual(x, y)
            for r in rows:
                read_full_scene(Path(d) / "scenes/train" / r["token"])

    def test_dataset_explicit_coverage_and_corrupt_data_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run_mock(d, retain=True)
            (root / "scenes.jsonl").write_text(
                json.dumps(dict(token="mock", split="train"))
                + "\n"
                + json.dumps(dict(token="missing", split="train"))
                + "\n"
            )
            with self.assertRaises(ValueError):
                FullScoreDataset(root)
            ds = FullScoreDataset(root, allow_partial=True)
            self.assertEqual(ds.missing_tokens, ["missing"])
            self.assertEqual(len(ds), 1)
            p = root / "scenes/train/mock/candidates.npy"
            a = np.load(p, mmap_mode="r+")
            a["labels"][0, 0] = 0.1
            a.flush()
            a._mmap.close()
            with self.assertRaisesRegex(ValueError, "contents"):
                ds[0]

    def test_training_samples_equal_count_from_different_budgets(self):
        with tempfile.TemporaryDirectory() as d:
            samples = []
            for token, n in [("a", 512), ("b", 1024)]:
                run_mock(d, budget=n, token=token, retain=True)
                meta, full = read_full_scene(Path(d) / "scenes/train" / token)
                indices = np.arange(0, n, n // 64)
                samples.append((meta, {k: v[indices] for k, v in full.items()}))
            batch = collate_high_score(samples)
            self.assertEqual(tuple(batch["scorer_trajectories"].shape), (2, 64, 8, 3))
            self.assertEqual(tuple(batch["scorer_pdm"].shape), (2, 64))

    def test_retention_upgrade_rejects_numerical_drift_and_preserves_original(self):
        prefix = "feedback/src/drivesta_feedback/batch/"
        old = dict(
            storage=dict(retain_all_scores=False),
            protocol=dict(storage=dict(retain_all_scores=False)),
            sampling=dict(threshold=0.95),
            implementation={
                prefix + n + ".py": "old" for n in ("production", "high_score_scene", "source")
            },
        )
        new = copy.deepcopy(old)
        new["storage"]["retain_all_scores"] = new["protocol"]["storage"]["retain_all_scores"] = True
        for n in ("production", "high_score_scene", "retained_bank"):
            new["implementation"][prefix + n + ".py"] = "new"
        with tempfile.TemporaryDirectory() as d:
            write_json(Path(d) / "run.json", old)
            with self.assertRaises(ValueError):
                resolve_run_contract(d, new)
            bad = copy.deepcopy(new)
            bad["sampling"]["threshold"] = 0.94
            with self.assertRaises(ValueError):
                resolve_run_contract(d, bad, upgrade=True)
            bad = copy.deepcopy(new)
            bad["implementation"][prefix + "source.py"] = "changed"
            with self.assertRaises(ValueError):
                resolve_run_contract(d, bad, upgrade=True)
            self.assertEqual(resolve_run_contract(d, new, upgrade=True), old)
            self.assertEqual(resolve_run_contract(d, new), old)
            self.assertEqual(read_json(Path(d) / "run.json"), old)
            with self.assertRaises(ValueError):
                resolve_run_contract(d, bad)

    def test_backfill_waits_for_all_high_score_scenes(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                require_high_banks_complete(d, 10)
            write_json(Path(d) / "COMPLETE.json", dict(passed=True, completed_scenes=9))
            with self.assertRaises(ValueError):
                require_high_banks_complete(d, 10)
            write_json(
                Path(d) / "COMPLETE.json",
                dict(passed=True, completed_scenes=10, shortfall_scenes=0),
            )
            require_high_banks_complete(d, 10)

    def test_cli_followup_starts_only_after_success_and_persists_on_shortfall(self):
        from drivesta_feedback.batch.production import main

        jobs = [
            dict(
                job_id="0",
                split="train",
                log_name="log",
                scenes=[
                    dict(
                        token="a",
                        split="train",
                        log_name="log",
                        metric_cache_relative_path="unused",
                    )
                ],
            )
        ]
        for passed in (False, True):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                phases = []

                def execute(jobs, output, protocol, config, contract_id, completed, workers):
                    phases.append(config.get("_backfill_full_scores", False))
                    if len(phases) == 1:
                        if passed:
                            completed["a"] = dict(
                                token="a", passed=True, candidates=256, full_scores_available=False
                            )
                            write_json(
                                output / "COMPLETE.json",
                                dict(passed=True, completed_scenes=1, shortfall_scenes=0),
                            )
                        return dict(passed=passed, shortfall_scenes=0 if passed else 1)
                    self.assertEqual(completed, {})
                    self.assertTrue((output / "COMPLETE.json").exists())
                    return dict(passed=True, shortfall_scenes=0)

                argv = [
                    "prepare",
                    "run",
                    "--config",
                    "feedback/configs/high_score_256.yaml",
                    "--output",
                    d,
                    "--backfill-after-complete",
                ]
                with (
                    patch("sys.argv", argv),
                    patch("drivesta_feedback.batch.production.selected_jobs", return_value=jobs),
                    patch("drivesta_feedback.batch.production.execute", side_effect=execute),
                    patch("builtins.print"),
                ):
                    if passed:
                        main()
                    else:
                        with self.assertRaises(SystemExit) as error:
                            main()
                        self.assertEqual(error.exception.code, 2)
                self.assertEqual(phases, [False, True] if passed else [False])
                followup = read_json(root / "followup.json")
                self.assertTrue(followup["requested"])
                self.assertEqual(
                    followup["state"], "complete" if passed else "waiting_for_shortfalls"
                )
                if passed:
                    write_json(
                        root / "followup.json",
                        dict(requested=True, action="backfill", state="running"),
                    )
                    argv = [
                        "prepare",
                        "backfill",
                        "--config",
                        "feedback/configs/high_score_256.yaml",
                        "--output",
                        d,
                    ]
                    with (
                        patch("sys.argv", argv),
                        patch(
                            "drivesta_feedback.batch.production.selected_jobs", return_value=jobs
                        ),
                        patch("drivesta_feedback.batch.production.execute", side_effect=execute),
                        patch("builtins.print"),
                    ):
                        main()
                    self.assertEqual(read_json(root / "followup.json")["state"], "complete")


if __name__ == "__main__":
    unittest.main()
