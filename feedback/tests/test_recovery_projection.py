import unittest
import numpy as np
from drivesta_feedback.batch.filter_targets import TargetPolicy, preliminary_rules
from drivesta_feedback.batch.recovery_projection_search import (
    motion_slack,
    project_motion,
    displaced_target,
    local_displacement,
)


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        self.policy = TargetPolicy(min_signed_speed_mps=-0.02)
        self.straight = np.column_stack((np.arange(1, 9) * 2.0, np.zeros(8), np.zeros(8)))

    def test_zero_change_on_valid_motion(self):
        p, result = project_motion(self.straight, [4.0, 0.0], [0.0, 0.0], self.policy)
        np.testing.assert_allclose(p, self.straight, atol=1e-5)
        self.assertGreaterEqual(result["minimum_slack"], 0)

    def test_negative_slack_for_reversal(self):
        p = self.straight.copy()
        p[:, 0] *= -1
        self.assertLess(motion_slack(p, [4.0, 0.0], [0.0, 0.0], self.policy).min(), 0)

    def test_repair_removes_abrupt_position_jump(self):
        p = self.straight.copy()
        p[3, 0] += 1
        fixed, result = project_motion(p, [4.0, 0.0], [0.0, 0.0], self.policy, 200)
        masks = preliminary_rules(
            fixed[None], np.ones((1, 7)), self.straight, self.policy, [4.0, 0.0], [0.0, 0.0]
        )
        self.assertTrue(all(x[0] for x in masks.values()))
        self.assertLess(np.linalg.norm(fixed - p), 1.0)

    def test_displacement_zero_and_terminal(self):
        np.testing.assert_array_equal(displaced_target(self.straight, 0.0, 0.0), self.straight)
        p = displaced_target(self.straight, 2.0, 0.5)
        np.testing.assert_allclose(p[-1, :2] - self.straight[-1, :2], [2.0, 0.5])

    def test_conservative_slack_implies_original_checks(self):
        rng = np.random.default_rng(42)
        count = 0
        for _ in range(50):
            p = self.straight + rng.normal(0, 0.005, (8, 3))
            if motion_slack(p, [4.0, 0.0], [0.0, 0.0], self.policy).min() >= 0:
                masks = preliminary_rules(
                    p[None], np.ones((1, 7)), self.straight, self.policy, [4.0, 0.0], [0.0, 0.0]
                )
                self.assertTrue(all(x[0] for x in masks.values()))
                count += 1
        self.assertGreater(count, 0)

    def test_local_displacement_is_bounded_and_has_compact_support(self):
        p = local_displacement(self.straight, 3.0, 0.5, True)
        np.testing.assert_array_equal(p[:3], self.straight[:3])
        self.assertLessEqual(np.linalg.norm(p[:, :2] - self.straight[:, :2], axis=1).max(), 0.5)
        np.testing.assert_allclose(p[5, 1], 0.5)

    def test_new_seed_choice_does_not_depend_on_previous_scene_names(self):
        import tempfile, json
        from pathlib import Path
        from drivesta_feedback.batch.recovery_seed_pilot import choose_cohort

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "scenes"
            folder.mkdir()
            for token, scores in [("a", {"shared": 96}), ("b", {"shared": 95.1, "better": 99})]:
                data = dict(
                    record=dict(token=token),
                    arms={k: dict(pdms=v, generator_eligible=True) for k, v in scores.items()},
                )
                (folder / (token + ".json")).write_text(json.dumps(data))
            cohort, _ = choose_cohort(tmp, 10)
            selected = {r["record"]["token"]: r["seed"] for r in cohort}
            self.assertEqual(selected["b"], "better")

    def test_two_interior_shapes_are_distinct_and_keep_endpoint_poses(self):
        from drivesta_feedback.batch.recovery_seed_pilot import (
            paired_interior_paths,
            paired_interior_keys,
        )

        keys = paired_interior_keys()
        self.assertEqual(len(keys), len(np.unique(keys, axis=0)))
        self.assertLessEqual(abs(keys).max(), 200)
        paths = paired_interior_paths(
            self.straight, np.array([[0, 0, 0, 0], [0, 100, 0, 0], [0, 0, 0, 100]])
        )
        np.testing.assert_array_equal(paths[0], self.straight)
        for path in paths:
            np.testing.assert_array_equal(path[[0, -1]], self.straight[[0, -1]])
        self.assertFalse(np.array_equal(paths[1], paths[2]))


if __name__ == "__main__":
    unittest.main()
