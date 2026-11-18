import unittest
import numpy as np
from drivesta_feedback.batch.recovery_gt_local import perturb_expert
from drivesta_feedback.audit.recovery_route_anchor import choose_anchor


class ExpertPerturbationTests(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(11) * 0.5
        self.poses = np.column_stack((self.times * 5, np.zeros(11), np.zeros(11)))

    def test_zero_retains_timed_expert(self):
        np.testing.assert_array_equal(
            perturb_expert(self.poses, self.times, np.zeros(6)), self.poses[1:9]
        )

    def test_middle_changes_keep_terminal_pose(self):
        c = np.array([0.3, 0, 0, 0.75, 0.35, 0.1])
        p = perturb_expert(self.poses, self.times, c)
        np.testing.assert_array_equal(p[-1], self.poses[8])
        self.assertLessEqual(abs(p[:, 1]).max(), 0.350001)
        self.assertLessEqual(abs(p[:, 2]).max(), 0.100001)

    def test_terminal_offset_is_explicit(self):
        p = perturb_expert(self.poses, self.times, [0, 1.5, 0.5, 0, 0, 0])
        np.testing.assert_allclose(p[-1], [21.5, 0.5, 0], atol=1e-6)

    def test_time_extrapolation_and_reversal_are_rejected(self):
        with self.assertRaises(ValueError):
            perturb_expert(self.poses[:8], self.times[:8], np.zeros(6))
        with self.assertRaises(ValueError):
            perturb_expert(self.poses, self.times, [10, 0, 0, 0, 0, 0])

    def test_heading_wrap_stays_finite(self):
        self.poses[:, 2] = np.linspace(3, 3.4, 11)
        self.poses[:, 2] = np.arctan2(np.sin(self.poses[:, 2]), np.cos(self.poses[:, 2]))
        p = perturb_expert(self.poses, self.times, [0, 0, 0, 0, 0, 0.1])
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue((abs(p[:, 2]) <= np.pi).all())

    def test_terminal_advance_uses_recorded_future(self):
        p = perturb_expert(self.poses, self.times, [0, 0, 0, 0, 0, 0, 0.75])
        self.assertAlmostEqual(float(p[-1, 0]), 23.75)
        with self.assertRaises(ValueError):
            perturb_expert(self.poses[:9], self.times[:9], [0, 0, 0, 0, 0, 0, 0.75])


class RouteAnchorTests(unittest.TestCase):
    def test_opposite_containing_lane_is_not_an_anchor(self):
        wrong = dict(
            lane_id="wrong",
            in_original_route=True,
            heading_error_deg=130,
            baseline_distance_m=0,
            covers_center=True,
        )
        self.assertIsNone(choose_anchor([wrong]))

    def test_directed_nearby_original_lane_is_selected(self):
        lane = dict(
            lane_id="a", in_original_route=True, heading_error_deg=2, baseline_distance_m=2.8
        )
        far = dict(
            lane_id="b", in_original_route=True, heading_error_deg=0, baseline_distance_m=3.1
        )
        self.assertEqual(choose_anchor([far, lane]), lane)
        self.assertIsNone(choose_anchor([dict(lane, in_original_route=False)]))


if __name__ == "__main__":
    unittest.main()
