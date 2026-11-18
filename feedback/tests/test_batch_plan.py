import json
from pathlib import Path
import tempfile
import unittest

from drivesta_feedback.batch.plan import build_jobs, commit_plan


def records():
    return [
        {
            "token": str(i),
            "log_name": f"log_{i // 3}",
            "split": "train" if i < 9 else "val",
            "scene_type": "unknown",
            "metric_cache_relative_path": f"log_{i // 3}/unknown/{i}/metric_cache.pkl",
        }
        for i in range(12)
    ]


class BatchPlanTests(unittest.TestCase):
    def test_full_coverage_and_job_bound(self):
        jobs = build_jobs(records(), {"splits": ["train", "val"], "max_scenes": None}, 2)
        self.assertEqual(sum(len(j["scenes"]) for j in jobs), 12)
        self.assertTrue(all(len(j["scenes"]) <= 2 for j in jobs))
        self.assertEqual(len({r["token"] for j in jobs for r in j["scenes"]}), 12)

    def test_seeded_subset_is_input_order_independent(self):
        profile = {"splits": ["train"], "max_scenes": 3, "selection_seed": 14}
        first = build_jobs(records(), profile)
        second = build_jobs(list(reversed(records())), profile)
        self.assertEqual(first, second)
        self.assertEqual(len({j["log_name"] for j in first}), 3)
        self.assertEqual({j["split"] for j in first}, {"train"})

    def test_duplicate_and_cross_split_logs_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_jobs(records() + [records()[0]], {"splits": ["train"]})
        bad = records()
        bad[0]["split"] = "val"
        with self.assertRaisesRegex(ValueError, "crosses"):
            build_jobs(bad, {"splits": ["train", "val"]})

    def test_plan_commit_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan"
            jobs = build_jobs(records(), {"splits": ["train", "val"]})
            commit_plan(output, jobs, {"selected_scenes": 12}, {}, {})
            self.assertEqual(len((output / "jobs.jsonl").read_text().splitlines()), len(jobs))
            with self.assertRaises(FileExistsError):
                commit_plan(output, [], {}, {}, {})


if __name__ == "__main__":
    unittest.main()
