"""Explicit recovery admits labelled shortfalls without changing the original high banks."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from feedback.tests.test_high_score_bank import run_mock, configuration
from drivesta_feedback.batch.full_score_backfill import (
    inventory,
    inspect_scene,
    require_finished_round,
    protected_files,
)
from drivesta_feedback.batch.store import write_json, read_json


def record(token):
    return dict(token=token, split="train", log_name="log", metric_cache_relative_path="unused")


class FullScoreBackfillTests(unittest.TestCase):
    def test_finished_partial_round_is_allowed_but_errors_and_unfinished_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            good = dict(
                state="partial",
                total_scenes=2,
                processed_scenes=2,
                completed_scenes=1,
                shortfall_scenes=1,
                failed_scenes=0,
            )
            write_json(root / "PARTIAL.json", good)
            self.assertEqual(require_finished_round(root, 2), good)
            for changes in (
                {"processed_scenes": 1},
                {"failed_scenes": 1},
                {"shortfall_scenes": 0},
                {"state": "running"},
            ):
                write_json(root / "PARTIAL.json", dict(good, **changes))
                with self.assertRaises(ValueError):
                    require_finished_round(root, 2)

    def test_inventory_skips_complete_low_score_tables_and_replays_only_missing_high_bank(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run_mock(d, token="legacy")
            run_mock(d, token="full", retain=True)
            run_mock(d, token="short", value=0.0, retain=True)
            rows = [record(t) for t in ("legacy", "full", "short")]
            before = (root / "scenes/train/short/SHORTFALL.json").read_bytes()
            completed, summary = inventory(root, rows, "test", workers=2)
            self.assertEqual(set(completed), {"full", "short"})
            self.assertEqual(completed["short"]["queries"], 512)
            self.assertEqual(completed["short"]["candidates"], 0)
            self.assertEqual(summary["missing"], [dict(token="legacy", split="train", queries=512)])
            run_mock(d, token="legacy", retain=True, backfill=True)
            resumed, summary = inventory(root, rows, "test", workers=2)
            self.assertEqual(len(resumed), 3)
            self.assertEqual(summary["missing_full_scenes"], 0)
            self.assertEqual((root / "scenes/train/short/SHORTFALL.json").read_bytes(), before)

    def test_missing_shortfall_table_fails_without_starting_new_search(self):
        with tempfile.TemporaryDirectory() as d:
            run_mock(d, token="short", value=0.0, retain=False)
            with self.assertRaisesRegex(ValueError, "shortfall scene has no full table"):
                inspect_scene(Path(d), record("short"), "test")

    def test_identity_contract_and_published_budget_are_checked(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run_mock(d, token="full", retain=True)
            with self.assertRaisesRegex(ValueError, "another numerical run"):
                inspect_scene(root, record("full"), "other")
            with self.assertRaisesRegex(ValueError, "identity"):
                inspect_scene(root, dict(record("full"), log_name="other"), "test")
            path = root / "scenes/train/full/COMPLETE.json"
            value = read_json(path)
            value["queries"] = 1024
            write_json(path, value)
            with self.assertRaisesRegex(ValueError, "recorded query budget"):
                inspect_scene(root, record("full"), "test")

    def test_backfill_markers_do_not_change_protected_first_round_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_json(root / "PARTIAL.json", dict(shortfall_scenes=1))
            before = protected_files(root)
            write_json(root / "BACKFILL_COMPLETE.json", dict(passed=True))
            self.assertEqual(protected_files(root), before)
            write_json(root / "PARTIAL.json", dict(shortfall_scenes=0))
            self.assertNotEqual(protected_files(root), before)

    def test_cli_completes_full_coverage_while_preserving_high_score_shortfalls(self):
        from drivesta_feedback.batch import full_score_backfill as module

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for token, retain, score in [
                ("legacy", False, 1.0),
                ("full", True, 1.0),
                ("short", True, 0.0),
            ]:
                run_mock(d, token=token, retain=retain, value=score)
            rows = [record(t) for t in ("legacy", "full", "short")]
            jobs = [dict(job_id="test", split="train", log_name="log", scenes=rows)]
            (root / "scenes.jsonl").write_text(
                "".join(__import__("json").dumps(r) + "\n" for r in rows)
            )
            write_json(root / "run.json", dict(protocol={}))
            write_json(
                root / "PARTIAL.json",
                dict(
                    state="partial",
                    total_scenes=3,
                    processed_scenes=3,
                    completed_scenes=2,
                    shortfall_scenes=1,
                    failed_scenes=0,
                ),
            )
            before = protected_files(root)
            config, _ = configuration()

            def replay(jobs, output, protocol, config, contract_id, completed, workers):
                self.assertTrue(config["_backfill_full_scores"])
                self.assertEqual(set(completed), {"full", "short"})
                run_mock(d, token="legacy", retain=True, backfill=True)
                result = dict(passed=True, full_score_scenes=3)
                write_json(root / "BACKFILL_COMPLETE.json", result)
                return result

            with (
                patch.object(module, "project_root", return_value=root),
                patch.object(module, "load_configuration", return_value=(config, {})),
                patch.object(module, "selected_jobs", return_value=jobs),
                patch.object(module, "source_contract", return_value={}),
                patch.object(module, "resolve_run_contract", return_value=dict(protocol={})),
                patch.object(module, "digest", return_value="test"),
                patch.object(module, "execute", side_effect=replay),
                patch(
                    "sys.argv",
                    [
                        "backfill_full_scores.py",
                        "run",
                        "--output",
                        d,
                        "--workers",
                        "2",
                        "--scan-workers",
                        "2",
                    ],
                ),
            ):
                module.main()
            self.assertEqual(protected_files(root), before)
            self.assertEqual(read_json(root / "followup.json")["state"], "complete")
            self.assertEqual(read_json(root / "followup.json")["high_score_shortfalls"], 1)
            self.assertTrue((root / "scenes/train/short/SHORTFALL.json").exists())


if __name__ == "__main__":
    unittest.main()
