import unittest
from dataclasses import replace
import numpy as np
from drivesta_feedback.batch.physical_sampling import (
    perturb,
    physical_mask,
    proposal_parameters,
    score_mask,
    select_bank,
)
from drivesta_feedback.batch.filter_targets import TargetPolicy
from drivesta_feedback.batch.scene_store import ROW_DTYPE, validate_rows


class PhysicalSamplingTest(unittest.TestCase):
    def setUp(self):
        self.gt = np.column_stack([np.arange(1, 9) * 5.0, np.zeros(8), np.zeros(8)]).astype(
            np.float32
        )
        self.policy = replace(TargetPolicy(), min_signed_speed_mps=-0.02, check_planned_space=True)

    def mask(self, paths):
        return physical_mask(np.array(paths), self.gt, self.policy, [10.0, 0.0], [0.0, 0.0])[0]

    def test_forward_and_reversal(self):
        reverse = self.gt.copy()
        reverse[:, :2] *= -1
        self.assertEqual(self.mask([self.gt, reverse]).tolist(), [True, False])

    def test_position_jump_rejected(self):
        bad = self.gt.copy()
        bad[3, 1] = 4
        self.assertFalse(self.mask([bad])[0])

    def test_initial_velocity_rejected(self):
        bad = self.gt.copy()
        bad[:, 0] *= 2
        self.assertFalse(self.mask([bad])[0])

    def test_curvature_rejected(self):
        bad = self.gt.copy()
        bad[2:6, 2] = 1
        self.assertFalse(self.mask([bad])[0])

    def test_zero_perturb_and_determinism(self):
        np.testing.assert_array_equal(perturb(self.gt, np.zeros((1, 6)))[0], self.gt)
        np.testing.assert_array_equal(
            proposal_parameters(512, 42), proposal_parameters(1024, 42)[:512]
        )

    def test_score_requires_direction_and_comfort(self):
        y = np.ones((4, 7))
        y[1, 5] = 0.5
        y[2, 4] = 0
        y[3, 6] = np.nextafter(0.95, 0)
        self.assertEqual(score_mask(y, self.policy).tolist(), [True, False, False, False])

    def test_top256_are_subset_and_rest_kept(self):
        paths = np.zeros((3100, 8, 3))
        labels = np.ones((3100, 7))
        labels[:, -1] = np.linspace(0, 1, 3100)
        eligible = labels[:, -1] >= 0.9
        full, high = select_bank(paths, labels, eligible)
        self.assertEqual(len(full), 3072)
        self.assertEqual(len(set(full)), 3072)
        self.assertEqual(len(high), 256)
        self.assertTrue((np.diff(labels[full[high], -1]) <= 0).all())
        self.assertIn(0, full)

    def test_shortfall_is_not_padded(self):
        full, high = select_bank(
            np.zeros((300, 8, 3)), np.ones((300, 7)), np.r_[np.ones(7, bool), np.zeros(293, bool)]
        )
        self.assertEqual(len(full), 300)
        self.assertEqual(len(high), 7)

    def test_new_reader_checks_physics_and_formula(self):
        from dataclasses import asdict

        rows = np.zeros(1, ROW_DTYPE)
        rows["trajectories"][0] = self.gt
        rows["labels"] = 1
        p = {
            "constructor": dict(
                format="physical_simscale_v1",
                policy=asdict(self.policy),
                expert=self.gt.tolist(),
                initial_velocity=[10.0, 0.0],
                initial_acceleration=[0.0, 0.0],
            )
        }
        validate_rows(rows, self.gt, p)
        rows["labels"][0, -1] = 0.9
        with self.assertRaises(ValueError):
            validate_rows(rows, self.gt, p)
        rows["labels"] = 1
        rows["trajectories"][:, 3, 0] += 5
        with self.assertRaises(ValueError):
            validate_rows(rows, self.gt, p)

    def test_missing_ttc_tail_is_rejected(self):
        from types import SimpleNamespace as S
        from drivesta_feedback.batch.simscale_source import (
            require_observation_horizon,
            MissingObservationHorizon,
        )

        cache = S(
            observation=S(
                _sample_interval=0.1,
                _global_to_local_idcs=list(range(41)),
                _occupancy_maps=[None] * 41,
            )
        )
        with self.assertRaises(MissingObservationHorizon):
            require_observation_horizon(cache)
        cache.observation._global_to_local_idcs = list(range(51))
        cache.observation._occupancy_maps = [None] * 51
        require_observation_horizon(cache)


if __name__ == "__main__":
    unittest.main()
