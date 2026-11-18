import numpy as np
import pytest
from drivesta_feedback.audit.simscale_state import (
    quaternion_matrix,
    quaternion_yaw,
    local_xy,
    pose_errors,
    finite,
)
from drivesta_feedback.audit.simscale_state import can_dynamic_errors, numeric_maxima


def test_identity_and_wxyz_yaw():
    assert np.allclose(quaternion_matrix([1, 0, 0, 0]), np.eye(3))
    assert np.allclose(
        quaternion_matrix([2**-0.5, 0, 0, 2**-0.5]), [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    )
    assert np.allclose(quaternion_matrix([-1, 0, 0, 0]), np.eye(3))


def test_local_xy_large_global_origin_and_nonzero_yaw():
    origin = np.array([600000.0, 4000000.0, np.pi / 2])
    assert np.allclose(
        local_xy([[600000.0, 4000002.0], [599997.0, 4000000.0]], origin), [[2.0, 0.0], [0.0, 3.0]]
    )


def test_invalid_quaternion_is_not_normalized_into_valid_pose():
    for q in [[0, 0, 0, 0], [1, 2, 3], [np.nan, 0, 0, 1]]:
        with pytest.raises(ValueError):
            quaternion_matrix(q)
        with pytest.raises(ValueError):
            quaternion_yaw(q)


def test_nonzero_pitch_roll_convention_is_explicit():
    q = np.array([0.8, 0.2, 0.4, 0.4])
    assert np.isclose(quaternion_yaw(q), np.arctan2(0.48, 0.36))
    matrix = quaternion_matrix(q)
    assert not np.isclose(quaternion_yaw(q), np.arctan2(matrix[1, 0], matrix[0, 0]))
    assert np.allclose(quaternion_matrix(q * 2), matrix)


def test_periodic_heading_and_xy_are_separate():
    e = pose_errors([1, 2, np.pi], [1.01, 2, -np.pi])
    assert np.isclose(e["xy_m"], 0.01)
    assert e["heading_rad"] < 1e-12


def test_nonfinite_comparisons_fail():
    with pytest.raises(ValueError):
        finite([1, np.nan])
    with pytest.raises(ValueError):
        pose_errors([0, 0, 0], [0, np.inf, 0])


def test_summary_missing_first_actor_comparison_does_not_hide_later_errors():
    rows = [
        dict(actor_center_max_error_m=None, matched_actors=0),
        dict(actor_center_max_error_m=3.25, matched_actors=1),
    ]
    expected = {"actor_center_max_error_m": 3.25}
    assert numeric_maxima(rows) == expected
    assert numeric_maxima(rows[::-1]) == expected
    assert numeric_maxima([rows[0]]) == {"actor_center_max_error_m": None}


def test_can_velocity_and_acceleration_keep_separate_units():
    can = np.zeros(18)
    can[10:12] = [2.0, -3.0]
    can[7:9] = [0.5, -0.2]
    assert can_dynamic_errors(can, [2.0, -3.0, 0.5, -0.2]) == (0.0, 0.0)
    v, a = can_dynamic_errors(can, [2.25, -3.0, 0.5, 0.3])
    assert v == 0.25
    assert a == 0.5
