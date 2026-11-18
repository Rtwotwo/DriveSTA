import numpy as np
from drivesta_feedback.batch.adaptive_sampling import (
    perturb_preserving_heading,
    forward_seeds,
    shuffled_parameters,
    quality,
)


def reference():
    t = np.arange(1, 9) * 0.5
    return np.column_stack([t * 3, np.sin(t) * 0.1, np.cos(t) * 0.03]).astype(np.float32)


def test_zero_preserves_stored_reference():
    ref = reference()
    np.testing.assert_array_equal(perturb_preserving_heading(ref, np.zeros((1, 6)))[0], ref)


def test_backward_reference_not_disguised_as_pi_heading():
    ref = reference()
    ref[:, :2] *= -1
    ref[:, 2] = 0
    result = perturb_preserving_heading(ref, np.zeros((1, 6)))[0]
    np.testing.assert_array_equal(result, ref)
    from drivesta_feedback.batch.physical_sampling import physical_mask
    from drivesta_feedback.batch.filter_targets import TargetPolicy

    assert not physical_mask(result[None], ref, TargetPolicy(), [0, 0], [0, 0])[0][0]


def test_forward_integrated_paths_have_nonnegative_body_progress():
    paths, meta = forward_seeds(reference(), [-0.5, 0], [-0.1, 0])
    full = np.concatenate([np.zeros((len(paths), 1, 3)), paths], 1)
    v = np.diff(full[:, :, :2], axis=1)
    h = (full[:, 1:, 2] + full[:, :-1, 2]) / 2
    assert (v[:, :, 0] * np.cos(h) + v[:, :, 1] * np.sin(h)).min() >= -1e-7
    assert len(paths) == len(meta)


def test_scale_subsets_cover_both_sides():
    a = shuffled_parameters(8192, 41)
    np.testing.assert_array_equal(a, shuffled_parameters(8192, 41))
    for i in range(4):
        assert abs(a[i::4].mean(0)).max() < 0.04


def test_quality_is_priority_only():
    labels = np.ones((2, 7))
    alternative = labels.copy()
    tracking = np.array([[0, 0, 0], [1.1, 2.1, 0.36]])
    q = quality(labels, alternative, tracking, [1, 2, 0.35], [True, True])
    assert q[0] > q[1]
    np.testing.assert_array_equal(labels, np.ones((2, 7)))
