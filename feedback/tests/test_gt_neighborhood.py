import unittest
import numpy as np
from drivesta_feedback.batch.gt_neighborhood import (
    local_keys,
    relative_heading,
    choose_cohort,
    experiment_protocol,
)


class GTNeighborhoodTests(unittest.TestCase):
    def test_deterministic_unique_nested_grid(self):
        rings = [[0.2, 0.01], [1.0, 0.05], [2.0, 0.1]]
        keys = local_keys(rings, 0.001)
        self.assertEqual(len(keys), 4521)
        self.assertEqual(len(np.unique(keys, axis=0)), len(keys))
        np.testing.assert_array_equal(keys[0], [0, 0])
        np.testing.assert_array_equal(keys, local_keys(rings, 0.001))
        self.assertEqual(np.abs(keys).max(), 2000)
        self.assertEqual(np.abs(keys[:1681]).max(), 200)

    def test_zero_offset_exact_reference_and_wrapped_delta(self):
        gt = np.zeros((8, 3), dtype=np.float64)
        gt[:, 2] = 0.1
        zero = gt.astype(np.float32)
        zero[:, 2] = np.pi - 0.05
        paths = np.stack([zero, zero])
        paths[1, :, 2] = -np.pi + 0.05
        paths[1, :, 0] = 0.2
        out = relative_heading(paths, zero, gt, np.array([[0, 0], [200, 0]]))
        np.testing.assert_array_equal(out[0], gt.astype(np.float32))
        np.testing.assert_allclose(out[1, :, 2], 0.2, atol=5e-7)
        np.testing.assert_array_equal(out[1, :, :2], paths[1, :, :2])

    def test_pilot_priorities(self):
        items = [
            dict(token=str(i), gt_pdms=(100 if i < 3 else i), eligible=(1 if 3 <= i < 12 else 0))
            for i in range(40)
        ]
        selected = choose_cohort(items, 16)
        self.assertEqual(len(selected), 16)
        self.assertEqual(len({x["token"] for x in selected}), 16)
        self.assertTrue(set(map(str, range(12))) <= {x["token"] for x in selected})
        self.assertEqual(choose_cohort(items, 0), items)

    def test_bad_spacing(self):
        with self.assertRaises(ValueError):
            local_keys([[0.2, 0.3]], 0.001)

    def test_protocol_describes_actual_output(self):
        source = dict(
            constructor={},
            sampling=dict(integer_lattice_m=0.001, max_unique_goals=3072),
            storage=dict(retain_all_scores=False),
        )
        p = experiment_protocol(source, "gt_relative_heading", np.zeros((4521, 2), np.int32))
        self.assertEqual(p["sampling"]["max_unique_goals"], 4521)
        self.assertEqual(p["sampling"]["query_budgets"], [4521])
        self.assertTrue(p["storage"]["retain_all_scores"])
        self.assertEqual(source["sampling"]["max_unique_goals"], 3072)
        self.assertFalse(source["storage"]["retain_all_scores"])


import pytest
from drivesta_feedback.batch.gt_neighborhood import gt_local_proposals, neighborhood_parameters


def straight():
    return np.column_stack([np.arange(1, 9) * 2.0, np.zeros((8, 2))]).astype(np.float32)


def test_zero_preserves_straight_curved_and_stopped():
    curve = straight()
    curve[:, 1] = np.linspace(0, 2, 8)
    curve[:, 2] = np.linspace(0, 0.3, 8)
    for ref in (straight(), curve, np.zeros((8, 3), np.float32)):
        assert np.array_equal(gt_local_proposals(ref, np.zeros((1, 6)))[0], ref)


def test_tangent_tracks_offset_and_returns_heading_at_endpoint():
    c = np.zeros((1, 6))
    c[0, 2] = 0.5
    result = gt_local_proposals(straight(), c)[0]
    assert np.all(result[:-1, 2] > 0)
    assert abs(result[-1, 2]) < 1e-6
    assert abs(result[-1, 1] - 0.5) < 1e-6
    assert np.allclose(result[:, 0], straight()[:, 0])


def test_timing_remains_in_observed_window_and_endpoint_fixed():
    c = np.zeros((2, 6))
    c[:, 0] = [-0.2, 0.2]
    result = gt_local_proposals(straight(), c)
    assert np.all(np.diff(result[:, :, 0], axis=1) > 0)
    assert np.all(result[:, :, 0] >= 0) and np.all(result[:, :, 0] <= 16)
    assert np.array_equal(result[:, -1], np.repeat(straight()[None, -1], 2, axis=0))
    c[0, 0] = 10
    with pytest.raises(ValueError):
        gt_local_proposals(straight(), c)


def test_batch_equals_single_and_small_rows_pass_motion_checks():
    from drivesta_feedback.batch.filter_targets import TargetPolicy
    from drivesta_feedback.batch.physical_sampling import physical_mask

    coeff = neighborhood_parameters(600, 123) * np.array([0.1, 0.2, 0.1, 0.1, 0.05, 0.03])
    batch = gt_local_proposals(straight(), coeff)
    assert np.array_equal(
        batch, np.stack([gt_local_proposals(straight(), c[None])[0] for c in coeff])
    )
    ok, _ = physical_mask(batch, straight(), TargetPolicy(), [4.0, 0.0], [0.0, 0.0])
    assert ok.all()


def test_structured_lines_and_seeded_combinations():
    a = neighborhood_parameters(1000, 1)
    b = neighborhood_parameters(1000, 1)
    assert np.array_equal(a, b)
    assert np.all(np.count_nonzero(a[:384], axis=1) == 1)
    assert np.all(np.abs(a) <= 1)
    assert not np.array_equal(a[384:], neighborhood_parameters(1000, 2)[384:])


if __name__ == "__main__":
    unittest.main()
