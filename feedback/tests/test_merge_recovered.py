"""Replacement integrity and recovery, with explicitly synthetic scored fixtures."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from feedback.tests.test_high_score_bank import configuration, construct, GT
from drivesta_feedback.batch.merge_recovered import (
    prepare_plan,
    prepare_scene,
    install_scene,
    finalize,
    select_full,
    transaction,
)
from drivesta_feedback.batch.retained_bank import read_full_scene, publish_full_scene
from drivesta_feedback.batch.high_score_bank import read_scene, save_bank, collate_high_score
from drivesta_feedback.batch.scene_store import ROW_DTYPE, row_digest
from drivesta_feedback.batch.store import atomic, read_json, write_json


def fixture(directory):
    root = Path(directory)
    target, source = root / "target", root / "source"
    r = dict(token="mock", split="train", log_name="log", metric_cache_relative_path="unused")
    _, protocol = configuration()
    contract = dict(identity=dict(r, expert_gt=GT.tolist()), protocol=protocol, run_contract="test")
    for bank, n in ((target, 512), (source, 1024)):
        p = bank / "scenes/train/mock"
        p.mkdir(parents=True)
        (bank / "scenes.jsonl").write_text(json.dumps(r) + "\n")
        rows = np.zeros(n, dtype=ROW_DTYPE)
        rows["keys"][:, 0] = np.arange(1000, 1000 + n)
        rows["trajectories"] = construct(GT, rows["keys"], protocol)[0]
        rows["labels"] = 1.0
        rows["labels"][:512, 0] = rows["labels"][:512, -1] = 0.0
        write_json(p / "contract.json", contract)
        write_json(p / "expert.json", dict(trajectory=GT.tolist(), labels=[1.0] * 7))
        atomic(p / "candidates.npy", lambda f: np.save(f, rows, allow_pickle=False))
        write_json(
            p / "state.json",
            dict(
                version=1,
                count=n,
                pending=None,
                completed_stages=["fixture"],
                blocks=[dict(start=0, end=n, digest=row_digest(rows))],
            ),
        )
        publish_full_scene(p, n)
        if bank == source:
            _, info = save_bank(p, rows, GT, protocol)
            write_json(
                p / "COMPLETE.json",
                dict(
                    format="high_score_256",
                    passed=True,
                    candidates=256,
                    audited_candidates=256,
                    max_official_error=0.0,
                    **info,
                ),
            )
        else:
            write_json(
                p / "SHORTFALL.json", dict(token="mock", split="train", candidates=0, passed=False)
            )
    write_json(target / "BACKFILL_COMPLETE.json", dict(passed=True))
    write_json(
        target / "PARTIAL.json",
        dict(
            completed_scenes=0,
            total_scenes=1,
            shortfalls=[dict(token="mock", split="train", candidates=0)],
        ),
    )
    write_json(target / "status.json", dict(state="partial"))
    write_json(target / "followup.json", dict(state="complete", high_score_shortfalls=1))
    config = dict(
        target=str(target),
        sources=[str(source)],
        revision="test",
        expected_scenes=1,
        full_count=512,
    )
    return config, r


class MergeRecoveredTests(unittest.TestCase):
    def test_real_readers_subset_and_label_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, r = fixture(tmp)
            plan = prepare_plan(c)
            item = plan["scenes"][0]
            source = Path(item["source"])
            source_bytes = (source / "candidates.npy").read_bytes()
            _, old_high = read_scene(source)
            result = prepare_scene(c, item)
            self.assertEqual(
                (result["full_count"], result["high_count"], result["zero_scores"]), (512, 256, 256)
            )
            self.assertFalse((Path(c["target"]) / "scenes/train/mock/COMPLETE.json").exists())
            install_scene(c, item)
            dest = Path(c["target"]) / "scenes/train/mock"
            metadata, full = read_full_scene(dest)
            _, high = read_scene(dest, verify=True)
            for k in ("keys", "trajectories", "labels"):
                np.testing.assert_array_equal(high[k], old_high[k])
                np.testing.assert_array_equal(full[k][high["query_indices"]], high[k])
            self.assertEqual(
                tuple(collate_high_score([(metadata, full)])["scorer_trajectories"].shape),
                (1, 512, 8, 3),
            )
            self.assertEqual(
                tuple(collate_high_score([(metadata, high)])["scorer_trajectories"].shape),
                (1, 256, 8, 3),
            )
            self.assertEqual((source / "candidates.npy").read_bytes(), source_bytes)
            final = finalize(c, plan)
            self.assertEqual((final["completed_scenes"], final["shortfall_scenes"]), (1, 0))
            self.assertEqual((Path(c["target"]) / "remaining_shortfall_tokens.txt").read_text(), "")

    def test_interrupted_directory_switch_and_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, _ = fixture(tmp)
            plan = prepare_plan(c)
            item = plan["scenes"][0]
            prepare_scene(c, item)

            def stop():
                raise RuntimeError("injected interruption after backup")

            with self.assertRaisesRegex(RuntimeError, "injected"):
                install_scene(c, item, after_backup=stop)
            self.assertFalse((Path(c["target"]) / "scenes/train/mock").exists())
            install_scene(c, item)
            dest = Path(c["target"]) / "scenes/train/mock/candidates.npy"
            before = dest.read_bytes()
            install_scene(c, item)
            finalize(c, plan)
            finalize(c, plan)
            self.assertEqual(dest.read_bytes(), before)
            self.assertTrue((transaction(c) / "backup/scenes/train/mock/SHORTFALL.json").exists())

    def test_bad_source_indices_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, _ = fixture(tmp)
            _, old = read_full_scene(Path(c["target"]) / "scenes/train/mock")
            _, src = read_full_scene(Path(c["sources"][0]) / "scenes/train/mock")
            _, high = read_scene(Path(c["sources"][0]) / "scenes/train/mock")
            high["query_indices"][0] = 0
            with self.assertRaisesRegex(ValueError, "does not match"):
                select_full(old, src, high, count=512)

    def test_changed_source_blocks_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, _ = fixture(tmp)
            item = prepare_plan(c)["scenes"][0]
            prepare_scene(c, item)
            p = Path(item["source"]) / "COMPLETE.json"
            data = read_json(p)
            data["unexpected_change"] = True
            write_json(p, data)
            with self.assertRaisesRegex(ValueError, "source changed"):
                install_scene(c, item)
            self.assertTrue((Path(c["target"]) / "scenes/train/mock/SHORTFALL.json").exists())

    def test_identity_and_threshold_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, _ = fixture(tmp)
            item = prepare_plan(c)["scenes"][0]
            p = Path(item["source"]) / "contract.json"
            data = read_json(p)
            data["identity"]["log_name"] = "different"
            write_json(p, data)
            with self.assertRaisesRegex(ValueError, "identities differ"):
                prepare_scene(c, item)

    def test_remaining_shortfalls_and_full_counts_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, r = fixture(tmp)
            target = Path(c["target"])
            second = dict(r, token="unrecovered")
            with (target / "scenes.jsonl").open("a") as f:
                f.write(json.dumps(second) + "\n")
            old = read_json(target / "PARTIAL.json")
            old["shortfalls"].append(dict(token="unrecovered", candidates=0))
            write_json(target / "PARTIAL.json", old)
            plan = prepare_plan(c)
            item = plan["scenes"][0]
            prepare_scene(c, item)
            install_scene(c, item)
            result = finalize(c, plan)
            self.assertEqual(result["shortfall_scenes"], 1)
            self.assertEqual(result["full_score_candidates"], 1024)
            self.assertEqual(
                (target / "remaining_shortfall_tokens.txt").read_text(), "unrecovered\n"
            )


if __name__ == "__main__":
    unittest.main()
