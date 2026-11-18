import unittest
import numpy as np
from drivesta_feedback.batch.profile import density_mask


class DensityProfileTests(unittest.TestCase):
    def test_keeps_probes_and_thins_only_defined_groups(self):
        keys = np.array([[0, 0], [2, 2], [4, 4], [5, 5], [1, 1], [2, 2]])
        phases = np.array([0, 0, 1, 2, 3, 3])
        names = ["000_base", "020_dense", "001_initial_probes", "040_boundary"]
        actual = density_mask(keys, phases, names, 0.2, 0.1)
        np.testing.assert_array_equal(actual, [True, False, True, True, False, True])

    def test_no_boundary_and_invalid_spacing(self):
        keys = np.array([[0, 0], [2, 2]])
        phases = np.array([0, 0])
        self.assertFalse(density_mask(keys, phases, ["040_boundary"], 0.1).any())
        with self.assertRaises(ValueError):
            density_mask(keys, phases, ["000_base"], 0.13)


if __name__ == "__main__":
    unittest.main()
