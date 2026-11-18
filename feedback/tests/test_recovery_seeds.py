import unittest
import numpy as np
from drivesta_feedback.batch.recovery_seeds import resample_recorded, to_local


class RecoverySeedsTest(unittest.TestCase):
    def test_irregular_time_interpolation(self):
        times = np.array([0.0, 0.5, 1.5, 3.0, 5.0])
        poses = np.column_stack((times * 2, times * 0, times * 0))
        result = resample_recorded(poses, times)
        np.testing.assert_allclose(result[:, 0], np.arange(1, 9))

    def test_no_extrapolation_or_duplicate_time(self):
        for times in (np.array([0.0, 3.0]), np.array([0.0, 0.0, 5.0])):
            with self.assertRaises(ValueError):
                resample_recorded(np.zeros((len(times), 3)), times)

    def test_periodic_heading(self):
        poses = np.array([[0.0, 0.0, 3.1], [0.0, 0.0, -3.1]])
        result = resample_recorded(poses, [0.0, 4.0], [2.0])
        self.assertAlmostEqual(abs(result[0, 2]), np.pi, places=6)

    def test_global_to_ego_rotation(self):
        origin = np.array([10.0, 20.0, np.pi / 2])
        result = to_local(np.array([[10.0, 22.0, np.pi / 2]]), origin)
        np.testing.assert_allclose(result, [[2.0, 0.0, 0.0]], atol=1e-6)

    def test_local_grid_unique_and_bounded(self):
        from drivesta_feedback.batch.recovery_seed_pilot import grid_keys

        keys = grid_keys()
        self.assertEqual(len(keys), len(np.unique(keys, axis=0)))
        np.testing.assert_array_equal(keys[0], [0, 0])
        self.assertLessEqual(abs(keys).max(), 800)
        self.assertGreater(len(keys), 1024)

    def test_time_profile_bounds_monotonic_and_initial_velocity(self):
        from drivesta_feedback.batch.recovery_motion_search import time_profile

        t = np.linspace(0, 4, 40001)
        for f in (0.84, 0.9, 1.0):
            for b in (-0.5, 0.0, 0.2):
                new = time_profile(t, f, b)
                self.assertTrue((np.diff(new) > 0).all())
                self.assertAlmostEqual(new[-1], 4 * f)
                self.assertAlmostEqual(new[1] / t[1], 1.0, places=3)

    def test_geometry_blend_wraps_heading(self):
        from drivesta_feedback.batch.recovery_motion_search import blend_paths

        a = np.array([[0.0, 0.0, 3.1]])
        b = np.array([[2.0, 4.0, -3.1]])
        result = blend_paths(a, b, 0.5)
        np.testing.assert_allclose(result[0, :2], [1, 2])
        self.assertAlmostEqual(abs(result[0, 2]), np.pi, places=6)

    def test_join_uses_measured_start_and_restores_reference(self):
        from drivesta_feedback.batch.recovery_motion_search import joined_reference

        t = np.linspace(0.0, 5.0, 50001)
        ref = np.column_stack((2 * t + 1, t * 0 + 1, t * 0))
        path = joined_reference(t, ref, [3.0, 0.0], [0.0, 0.0], 3.0)
        np.testing.assert_allclose(path[0], 0.0)
        np.testing.assert_allclose((path[1, :2] - path[0, :2]) / (t[1] - t[0]), [3, 0], atol=1e-6)
        np.testing.assert_allclose(path[t >= 3, :2], ref[t >= 3, :2], atol=1e-8)

    def test_interior_operator_keeps_endpoints_and_zero_exact(self):
        from drivesta_feedback.batch.recovery_seed_pilot import interior_paths

        seed = np.column_stack((np.arange(1, 9), np.zeros(8), np.zeros(8))).astype(np.float32)
        paths = interior_paths(seed, np.array([[0, 0], [0, 100]]))
        np.testing.assert_array_equal(paths[0], seed)
        np.testing.assert_array_equal(paths[1, [0, -1]], seed[[0, -1]])
        self.assertGreater(abs(paths[1, 1:-1, 1]).max(), 0.09)


if __name__ == "__main__":
    unittest.main()
