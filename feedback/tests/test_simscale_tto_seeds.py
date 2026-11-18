import numpy as np
from drivesta_feedback.batch.simscale_tto_seeds import control_residual, compensated
from drivesta_feedback.batch.filter_targets import TargetPolicy
from drivesta_feedback.batch.physical_sampling import physical_mask, score_mask


def path():
    t = np.arange(1, 9) * 0.5
    return np.column_stack([5 * t, 0.03 * t * t, 0.012 * t]).astype(np.float32)


def test_control_zero_exact_even_when_reference_not_kinematic_rollout():
    assert np.array_equal(control_residual(path(), np.zeros((1, 6)), [5, 0])[0], path())


def test_zero_preserves_stationary_heading():
    p = np.zeros((8, 3), np.float32)
    p[:, 2] = 0.02
    assert np.array_equal(control_residual(p, np.zeros((1, 6)), [0, 0])[0], p)


def test_batch_and_singleton_control_match():
    c = np.random.default_rng(3).uniform(-0.03, 0.03, (5, 6))
    batch = control_residual(path(), c, [5, 0])
    assert np.allclose(
        batch, np.stack([control_residual(path(), x[None], [5, 0])[0] for x in c]), atol=1e-6
    )
    assert batch.shape == (5, 8, 3) and np.isfinite(batch).all()


def test_compensation_zero_and_wrapped_heading():
    p = path()
    assert np.array_equal(compensated(p, p, 0), p)
    tracked = p.copy()
    tracked[:, 2] += 2 * np.pi
    assert np.allclose(compensated(p, tracked, 0.5), p, atol=1e-6)


def test_residual_proposals_do_not_bypass_driving_policy():
    c = np.zeros((1, 6))
    c[0, 0] = -20
    proposals = control_residual(path(), c, [5, 0])
    accepted, _ = physical_mask(proposals, path(), TargetPolicy(max_jerk_mps3=0.01), [5, 0], [0, 0])
    assert not accepted[0]
    labels = np.ones((1, 7))
    labels[:, 5] = 0.5
    assert not score_mask(labels, TargetPolicy())[0]
