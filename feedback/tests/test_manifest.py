"""Small contract tests for split isolation and durable manifest creation."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from drivesta_feedback.manifest import collect_records, commit_manifest


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.split = self.root / "split.yaml"
        self.split.write_text(
            "train_logs: [log_train]\nval_logs: [log_val]\ntest_logs: [log_test]\n"
        )
        self.metadata = self.root / "metadata.csv"
        self.rows = [
            "/old/cache/log_train/unknown/1111111111111111/metric_cache.pkl",
            "/old/cache/log_val/unknown/2222222222222222/metric_cache.pkl",
        ]

    def collect(self, rows=None, check_files=False):
        with self.metadata.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["file_name"])
            writer.writerows([[row] for row in (self.rows if rows is None else rows)])
        return collect_records(self.metadata, self.split, self.root / "cache", 1, 1, check_files)

    def test_rebase_split_and_atomic_commit(self):
        records, summary = self.collect()
        self.assertEqual(summary["splits"]["train"]["scenes"], 1)
        self.assertTrue(records[0]["metric_cache_path"].startswith(str(self.root / "cache")))
        self.assertEqual(summary["cache_contents"], "not_decoded")
        output = self.root / "manifest"
        commit_manifest(output, records, summary)
        self.assertEqual(len((output / "scenes.jsonl").read_text().splitlines()), 2)
        self.assertEqual(json.loads((output / "summary.json").read_text())["total_scenes"], 2)
        with self.assertRaises(FileExistsError):
            commit_manifest(output, records, summary)

    def test_duplicate_token_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.collect(self.rows + [self.rows[0]])

    def test_log_overlap_is_rejected(self):
        self.split.write_text("train_logs: [log_train]\nval_logs: [log_train]\n")
        with self.assertRaisesRegex(ValueError, "overlapping"):
            self.collect()

    def test_unknown_and_test_logs_are_rejected(self):
        for log, message in (("unknown_log", "Unknown"), ("log_test", "Test scene")):
            with self.subTest(log=log), self.assertRaisesRegex(ValueError, message):
                self.collect([self.rows[0].replace("log_train", log), self.rows[1]])

    def test_missing_cache_check_is_explicit(self):
        self.collect(check_files=False)
        with self.assertRaises(FileNotFoundError):
            self.collect(check_files=True)

    def test_incomplete_split_fails(self):
        with self.assertRaisesRegex(ValueError, "val count"):
            self.collect([self.rows[0]])


if __name__ == "__main__":
    unittest.main()
