import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from drivesta_feedback.batch.recovery_rollout_search import signed_clearance


class ClearanceTest(unittest.TestCase):
    def test_proposal_horizon_ignores_fifth_second(self):
        from drivesta_feedback.batch.recovery_proposal_seeds import scored_prefix

        states = np.zeros((51, 11))
        states[:, 0] = np.arange(51) * 0.1
        states[41:] = np.nan
        np.testing.assert_allclose(scored_prefix(states)[5::5, 0], np.arange(1, 9) * 0.5)

    def test_short_proposal_cannot_be_silently_padded(self):
        from drivesta_feedback.batch.recovery_proposal_seeds import scored_prefix

        with self.assertRaises(ValueError):
            scored_prefix(np.zeros((40, 11)))

    def test_direction_windows_match_official_inclusive_horizon(self):
        from drivesta_feedback.batch.recovery_rollout_search import direction_slack

        xy = np.column_stack((np.arange(41) * 0.2, np.zeros(41)))
        outside = np.ones(41, dtype=bool)
        slack = direction_slack(xy, outside, margin=1.0)
        self.assertAlmostEqual(slack[0], 1.0)
        self.assertAlmostEqual(slack[10], 0.0)
        self.assertAlmostEqual(slack[11], -0.1)
        outside[:] = False
        np.testing.assert_array_equal(direction_slack(xy, outside), np.ones(41))

    def test_spline_zero_preserves_all_waypoints(self):
        from drivesta_feedback.batch.recovery_rollout_search import spline_correction

        parent = np.arange(24, dtype=float).reshape(8, 3) / 10
        np.testing.assert_array_equal(spline_correction(parent, np.zeros(12)), parent)

    def test_spline_interpolates_control_knots(self):
        from drivesta_feedback.batch.recovery_rollout_search import spline_correction

        controls = np.arange(12, dtype=float).reshape(4, 3) / 10
        paths = spline_correction(np.zeros((8, 3)), controls)
        expected = controls.copy()
        expected[:, 2] *= 0.1
        np.testing.assert_allclose(paths[1::2], expected, atol=1e-12)

    def test_signed_distances(self):
        from shapely.geometry import box

        actual = signed_clearance(box(0, 0, 4, 4), [[2, 2], [0, 1], [-1, 2], [5, 5]])
        np.testing.assert_allclose(actual, [2, 0, -1, -np.sqrt(2)])

    def test_hole_is_outside(self):
        from shapely.geometry import box

        area = box(-3, -3, 3, 3).difference(box(-1, -1, 1, 1))
        actual = signed_clearance(area, [[0, 0], [2, 0], [3, 0]])
        np.testing.assert_allclose(actual, [-1, 1, 0])

    def test_feasible_reference_precedes_wrong_direction_full_score(self):
        from drivesta_feedback.batch.recovery_rollout_search import feasibility_parents

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "scenes").mkdir()
            names = ["wrong_direction", "valid_lower_score"]
            paths = np.zeros((2, 8, 3), np.float32)
            paths[1, :, 0] = 1
            labels = np.ones((2, 7))
            labels[1, -1] = 0.9
            np.savez(root / "scenes" / "token.npz", names=names, trajectories=paths, labels=labels)
            arms = {
                names[0]: dict(failed_rules=["direction_compliance"], tracking_errors=[0, 0, 0]),
                names[1]: dict(failed_rules=["pdm"], tracking_errors=[0, 0, 0]),
            }
            (root / "scenes" / "token.json").write_text(json.dumps(dict(arms=arms)))
            chosen = feasibility_parents("token", [root], 1)
            self.assertTrue(chosen[0][0].endswith("valid_lower_score"))

    def test_comfort_slack_matches_official_decision(self):
        from drivesta_feedback.batch.recovery_rollout_search import comfort_slack
        from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
            ego_is_comfortable,
        )
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex

        times = np.arange(41) * 0.1
        for acceleration in (0.0, 2.0, 2.5, -4.0, -4.2):
            state = np.zeros((41, StateIndex.size()))
            state[:, StateIndex.ACCELERATION_X] = acceleration
            actual = bool(ego_is_comfortable(state[None], times).all())
            self.assertEqual(bool((comfort_slack(state, margin=1.0) > 0).all()), actual)

    def test_fine_queries_are_distinct_integer_geometry_changes(self):
        from drivesta_feedback.batch.recovery_seed_pilot import (
            paired_interior_keys,
            paired_interior_paths,
        )

        keys = paired_interior_keys((10, 25, 50))
        self.assertEqual(len({tuple(k) for k in keys}), len(keys))
        self.assertLessEqual(abs(keys).max(), 50)
        seed = np.zeros((8, 3), np.float32)
        seed[:, 0] = np.arange(1, 9) * 2
        paths = paired_interior_paths(seed, keys)
        self.assertEqual(len({p.tobytes() for p in paths}), len(paths))
        np.testing.assert_array_equal(paths[:, 0], np.broadcast_to(seed[0], (len(paths), 3)))
        np.testing.assert_array_equal(paths[:, -1], np.broadcast_to(seed[-1], (len(paths), 3)))


if __name__ == "__main__":
    unittest.main()
