import numpy as np

from drivesta_feedback.batch.simscale_recovery import local_parent_indices


def paths_at(offsets):
    paths = np.zeros((len(offsets), 8, 3), np.float32)
    paths[:, :, 0] = np.arange(1, 9)
    paths[:, :, 1] = np.asarray(offsets)[:, None]
    return paths


def test_better_far_parents_cannot_starve_near_gt_feedback():
    paths = paths_at([5.0, 6.0, 0.2, 0.8])
    scores = np.array([1.0, 1.0, 0.97, 0.96])
    ids = local_parent_indices(paths, scores, paths_at([0.0])[0], 3.0, 2, 0.1)
    assert ids.tolist() == [2, 3]


def test_qualified_subset_preserves_full_bank_ids():
    paths = paths_at([0.1, 5.0, 0.4, 0.8])
    scores = np.array([1.0, 1.0, 0.98, 0.97])
    ids = local_parent_indices(paths, scores, paths_at([0.0])[0], 3.0, 8, 0.1, ids=[1, 3])
    assert ids.tolist() == [3]


def test_no_feasible_parent_does_not_relax_bound_or_repeat():
    paths = paths_at([3.01, 4.0])
    reference = paths_at([0.0])[0]
    assert len(local_parent_indices(paths, np.ones(2), reference, 3.0, 12, 0.01)) == 0
    assert len(local_parent_indices(paths, np.ones(2), reference, 3.0, 12, 0.01, ids=[])) == 0


def test_bound_applies_to_every_waypoint_not_only_endpoint():
    paths = paths_at([0.0, 0.0])
    paths[0, 3, 1] = 3.01
    paths[1, :, 1] = 3.0
    ids = local_parent_indices(paths, np.ones(2), paths_at([0.0])[0], 3.0, 12, 0.01)
    assert ids.tolist() == [1]
