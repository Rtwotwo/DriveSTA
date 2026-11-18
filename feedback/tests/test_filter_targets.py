import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from drivesta_feedback.batch.filter_targets import (
    TargetPolicy,
    preliminary_rules,
    read_target_view,
    select_targets,
    tracking_errors,
)


def bank(n=3):
    t = np.zeros((n, 8, 3), dtype=np.float32)
    for i in range(n):
        t[i, :, 0] = np.arange(1, 9) * (0.5 + 0.04 * i)
    y = np.ones((n, 7), dtype=float)
    return t, y, t[0].copy()


class TargetFilterTests(unittest.TestCase):
    def test_direction_is_separate_from_pdm(self):
        t, y, gt = bank()
        y[0, 5] = 0.5
        masks = preliminary_rules(t, y, gt, TargetPolicy())
        self.assertTrue(masks["pdm"][0])
        self.assertFalse(masks["direction_compliance"][0])
        np.testing.assert_array_equal(y[:, -1], 1.0)

    def test_backward_motion(self):
        t, y, gt = bank()
        t[1, :, 0] *= -1
        self.assertFalse(preliminary_rules(t, y, gt, TargetPolicy())["forward_motion"][1])

    def test_spatial_gate_required_and_rejects(self):
        t, y, gt = bank()
        policy = TargetPolicy(check_planned_space=True)
        with self.assertRaises(ValueError):
            select_targets(t, y, gt, policy, lambda i: (y[i], 0.0, 0.0, 0.0))
        result = select_targets(t, y, gt, policy, lambda i: (y[i], 0.0, 0.0, 0.0, False))
        self.assertEqual(result["target_count"], 0)

    def test_missing_route_geometry_is_explicit(self):
        from types import SimpleNamespace
        import shapely
        from drivesta_feedback.batch.filter_targets import (
            planned_space_checker,
            MissingPlannedGeometry,
        )

        class Map:
            tokens = ["road"]

            def get_indices_of_map_type(self, layers):
                return [0]

            def __getitem__(self, token):
                return shapely.box(-100, -100, 100, 100)

        with self.assertRaises(MissingPlannedGeometry):
            planned_space_checker(SimpleNamespace(drivable_area_map=Map(), route_lane_ids=[]))

    def test_slow_reverse_is_rejected(self):
        t, y, gt = bank()
        t[0, :, 0] = -np.arange(1, 9) * 0.015
        masks = preliminary_rules(t, y, gt, TargetPolicy(min_signed_speed_mps=-0.02))
        self.assertFalse(masks["forward_motion"][0])
        self.assertFalse(masks["backward_distance"][0])

    def test_duplicates_do_not_fill_slots(self):
        t, y, gt = bank()
        t[1:] = t[0]
        result = select_targets(t, y, gt, TargetPolicy(), lambda i: (y[i], 0.0, 0.0, 0.0))
        self.assertEqual(result["target_count"], 1)

    def test_tracking_failure_returns_expert_only(self):
        t, y, gt = bank()
        result = select_targets(t, y, gt, TargetPolicy(), lambda i: (y[i], 2.0, 3.0, 0.0))
        self.assertTrue(result["expert_only"])
        self.assertTrue(result["preserve_expert"])
        self.assertEqual(result["query_indices"], [])

    def test_label_mismatch_is_error(self):
        t, y, gt = bank()
        with self.assertRaises(ValueError):
            select_targets(t, y, gt, TargetPolicy(), lambda i: (np.zeros(7), 0.0, 0.0, 0.0))

    def test_empty_view_and_source_indices(self):
        t, y, gt = bank()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rows = np.zeros(len(t), dtype=[("trajectories", "f4", (8, 3)), ("labels", "f8", (7,))])
            rows["trajectories"] = t
            rows["labels"] = y
            np.save(folder / "candidates.npy", rows)
            (folder / "expert.json").write_text(json.dumps({"trajectory": gt.tolist()}))
            result = read_target_view(folder, {"query_indices": []})
            self.assertEqual(result["auxiliary_trajectories"].shape, (0, 8, 3))
            result = read_target_view(folder, {"query_indices": [2, 0]})
            np.testing.assert_array_equal(result["auxiliary_trajectories"], t[[2, 0]])
            np.testing.assert_array_equal(result["auxiliary_labels"], y[[2, 0]])
            with self.assertRaises(ValueError):
                read_target_view(
                    folder, {"query_indices": [0], "selected_labels": np.zeros((1, 7)).tolist()}
                )

    def test_budget_and_nonfinite(self):
        t, y, gt = bank()
        p = TargetPolicy(max_targets=1, max_track_queries=1)
        result = select_targets(t, y, gt, p, lambda i: (y[i], 2.0, 3.0, 0.0))
        self.assertTrue(result["verification_budget_exhausted"])
        t[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            preliminary_rules(t, y, gt, p)

    def test_initial_velocity_and_heading_tracking(self):
        t, y, gt = bank()
        masks = preliminary_rules(
            t, y, gt, TargetPolicy(), initial_velocity=[10.0, 0.0], initial_acceleration=[0.0, 0.0]
        )
        self.assertFalse(masks["initial_velocity"].any())
        result = select_targets(t, y, gt, TargetPolicy(), lambda i: (y[i], 0.0, 0.0, 0.5))
        self.assertTrue(result["expert_only"])

    def test_tracking_rotation_and_intermediate_times(self):
        trajectory = np.zeros((8, 3))
        trajectory[:, 0] = np.arange(1, 9)
        # Ego +x is world +y at pi/2; the 8 waypoints and initial point match.
        sim = np.zeros((41, 3))
        sim[:, 0] = 10.0
        sim[:, 1] = -3.0 + np.arange(41) * 0.2
        sim[:, 2] = np.pi / 2
        np.testing.assert_allclose(
            tracking_errors(trajectory, sim, [10.0, -3.0, np.pi / 2]), 0.0, atol=1e-12
        )
        sim[2, 0] += 3.0
        ade, maximum, heading = tracking_errors(trajectory, sim, [10.0, -3.0, np.pi / 2])
        self.assertAlmostEqual(ade, 3.0 / 40)
        self.assertAlmostEqual(maximum, 3.0)
        self.assertAlmostEqual(heading, 0.0)

    def test_heading_wrap_and_initial_error(self):
        trajectory = np.zeros((8, 3))
        trajectory[:, 0] = np.arange(1, 9)
        trajectory[:, 2] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, -3.0, -2.5]
        sim = np.zeros((41, 3))
        sim[:, 0] = np.arange(41) * 0.2
        sim[:, 2] = np.interp(
            np.arange(41) * 0.1, np.arange(9) * 0.5, np.unwrap(np.r_[0.0, trajectory[:, 2]])
        )
        sim[:, 2] = np.arctan2(np.sin(sim[:, 2]), np.cos(sim[:, 2]))
        np.testing.assert_allclose(
            tracking_errors(trajectory, sim, [0.0, 0.0, 0.0]), 0.0, atol=1e-12
        )
        sim[0, 2] = 0.6
        self.assertAlmostEqual(tracking_errors(trajectory, sim, [0.0, 0.0, 0.0])[2], 0.6)


if __name__ == "__main__":
    unittest.main()
