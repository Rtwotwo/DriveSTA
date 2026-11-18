import numpy as np
import pytest
from drivesta_feedback.epdms.sampling import family_parameters, construct, unique_ranked


def expert():
    t = np.arange(1, 9) * 0.5
    return np.stack([5 * t, 0.05 * t * t, np.arctan(0.02 * t)], -1)


def test_families_have_persistent_identity():
    a = family_parameters()
    assert a.shape == (3072, 6)
    np.testing.assert_array_equal(a, family_parameters())
    assert len(np.unique(a, axis=0)) == 3072


def test_exact_expert_zero_and_finite():
    gt = expert()
    p = construct(gt, family_parameters())
    np.testing.assert_array_equal(p[0], gt.astype(np.float32))
    assert p.shape == (3072, 8, 3) and np.isfinite(p).all()
    assert len(np.unique(p.reshape(3072, -1), axis=0)) == 3072


def test_strict_cutoff_and_deterministic_ties():
    p = construct(expert(), family_parameters()[:5])
    scores = np.array([0.95, np.nextafter(0.95, 1), 1.0, 1.0, 0.5])
    np.testing.assert_array_equal(unique_ranked(p, scores, np.ones(5, bool)), [2, 3, 1])


def test_no_duplicate_padding_or_unsafe_targets():
    p = np.repeat(expert()[None], 4, axis=0)
    p[3, :, 0] += 0.2
    scores = np.ones(4)
    np.testing.assert_array_equal(unique_ranked(p, scores, [True, True, True, False]), [0])
    assert unique_ranked(p, scores, np.zeros(4, bool)).size == 0


def test_stationary_heading_preserved():
    p = construct(np.zeros((8, 3)), family_parameters()[:100])
    assert np.isfinite(p).all()
    np.testing.assert_array_equal(p[0], np.zeros((8, 3)))


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        construct(np.zeros((10, 3)), family_parameters())
    with pytest.raises(ValueError):
        unique_ranked(np.zeros((1, 8, 3)), np.array([np.nan]), [True])


def test_strict_reversing_rule():
    from drivesta_feedback.epdms.dataset import POLICY
    from drivesta_feedback.batch.physical_sampling import physical_mask

    paths = np.zeros((2, 8, 3))
    paths[0, :, 0] = -np.arange(1, 9) * 0.001
    paths[1, :, 0] = np.arange(1, 9) * 0.001
    _, masks = physical_mask(paths, np.zeros((8, 3)), POLICY, [0, 0], [0, 0])
    assert not masks["forward_motion"][0] and not masks["backward_distance"][0]
    assert masks["forward_motion"][1] and masks["backward_distance"][1]


def test_ec_fail_caps_score_below_cutoff():
    from drivesta_feedback.epdms.scoring import score_from_components

    assert float(score_from_components(np.ones(8), 0)) == 0.875
    assert float(score_from_components(np.ones(8), 1)) == 1.0
    assert float(score_from_components(np.ones(8))) == 1.0


def test_ec_time_shift_matches_official():
    from drivesta_feedback.epdms.dataset import ec_pair
    from navsim.v2.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
        ego_is_two_frame_extended_comfort,
    )

    rng = np.random.default_rng(2)
    a = rng.normal(size=(2, 41, 11))
    b = rng.normal(size=(2, 41, 11))
    np.testing.assert_array_equal(
        ec_pair(a, b, 0.5),
        ego_is_two_frame_extended_comfort(a[:, :-5], b[:, 5:], np.arange(36) * 0.1),
    )
    with pytest.raises(ValueError):
        ec_pair(a, b, 0.55)


def test_reference_world_to_local():
    from drivesta_feedback.epdms.sampling import local_reference

    p = np.array([[10.0, 21.0, np.pi / 2], [8.0, 22.0, np.pi]])
    np.testing.assert_allclose(
        local_reference(p, [10.0, 20.0, np.pi / 2]), [[1, 0, 0], [2, 2, np.pi / 2]], atol=1e-12
    )


def test_mixed_family_splice_keeps_ids():
    from drivesta_feedback.epdms.sampling import construct_bank

    gt = expert()
    pdm = gt.copy()
    pdm[:, 0] *= 1.05
    params = family_parameters()
    mixed = construct_bank(gt, pdm, params)
    np.testing.assert_array_equal(mixed[:2304], construct(gt, params[:2304]))
    np.testing.assert_array_equal(mixed[2304:], construct(pdm, params[2304:]))


def test_regular_endpoint_grids_and_exact_completion():
    params = family_parameters()
    gt = expert()
    for section, nx, ny in [(slice(0, 2304), 48, 48), (slice(2304, None), 32, 24)]:
        offsets = params[section, :2]
        assert len(np.unique(offsets, axis=0)) == nx * ny
        np.testing.assert_allclose(np.diff(np.unique(offsets[:, 0])), 0.05)
        np.testing.assert_allclose(np.diff(np.unique(offsets[:, 1])), 0.05)
        paths = construct(gt, params[section])
        c, s = np.cos(gt[-1, 2]), np.sin(gt[-1, 2])
        achieved = (paths[:, -1, :2] - gt[-1, :2]) @ np.array([[c, -s], [s, c]])
        np.testing.assert_allclose(achieved, offsets, atol=2e-6)
