import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from shapely.geometry import Polygon, Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drivesta_feedback.audit.simscale_geometry import (
    polygon_record,
    precise_map,
    check_reuse_intent,
)
from drivesta_feedback.batch.simscale_recovery import repaired_cache_path, relabel_existing
from drivesta_feedback.batch.store import write_json


def test_float32_can_change_initial_containment():
    source = Polygon(
        [
            (589014.94, 4474151.39),
            (589024.94, 4474151.39),
            (589024.94, 4474161.39),
            (589014.94, 4474161.39),
        ]
    )
    cached = Polygon(np.asarray(source.exterior.coords).astype(np.float32))
    point = Point(589015, 4474151.4)
    assert source.covers(point) and not cached.covers(point)
    row = polygon_record(cached, source)
    assert row["float32_source_match"] and row["changed"]
    assert row["max_coordinate_error_m"] > 0.1


def test_exact_source_geometry_retains_identity_and_invalidity():
    from drivesta_feedback.paths import bootstrap_navsim

    bootstrap_navsim()
    from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
        PDMDrivableMap,
    )
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L

    invalid = Polygon([(0, 0), (1, 1), (0, 1), (1, 0), (0, 0)])
    valid = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    old = PDMDrivableMap(["one", "two"], [L.LANE, L.LANE_CONNECTOR], [valid, valid])
    objects = {("one", L.LANE): valid, ("two", L.LANE_CONNECTOR): invalid}
    api = SimpleNamespace(get_map_object=lambda t, k: SimpleNamespace(polygon=objects[t, k]))
    restored, rows = precise_map(old, api)
    assert restored.tokens == old.tokens and restored.map_types == old.map_types
    assert restored["one"].equals_exact(valid, 0)
    assert not restored["two"].is_valid and not rows[1]["source_valid"]
    assert old["two"].equals_exact(valid, 0)


def test_cache_repair_rejects_wrong_source(tmp_path):
    cfg = {"cache_repairs": str(tmp_path)}
    write_json(
        tmp_path / "scenes/t/repair.json",
        {"token": "t", "cache_path": "new", "provenance": {"source_cache_path": "other"}},
    )
    with pytest.raises(ValueError):
        repaired_cache_path("t", cfg, {"identity": {"official_cache_path": "old"}})


def test_route_intent_cannot_silently_reuse_geometry_only_cache():
    row = {"token": "t", "geometry_only": {"valid": False}, "provenance": {"route_repair": None}}
    with pytest.raises(ValueError):
        check_reuse_intent(row, {"token": "t"})
    check_reuse_intent(row, None)
    row["provenance"]["route_repair"] = {"connected": True}
    check_reuse_intent(row, {"token": "t"})
    with pytest.raises(ValueError):
        check_reuse_intent(row, None)


def test_relabel_recomputes_old_eligibility_and_resumes(tmp_path):
    old = dict(
        trajectories=np.zeros((3, 8, 3), np.float32),
        labels=np.ones((3, 7)),
        alternative_labels=np.ones((3, 7)),
        tracking=np.zeros((3, 3)),
        eligible=np.ones(3, bool),
    )
    old["trajectories"][:, 0, 0] = np.arange(3)
    untouched = copy.deepcopy(old)

    class Evaluator:
        def score(self, paths):
            labels = np.ones((len(paths), 7))
            labels[paths[:, 0, 0] == 1, 0] = 0
            labels[paths[:, 0, 0] == 1, -1] = 0
            self.alternative_labels = labels.copy()
            return labels, paths

    policy = SimpleNamespace(
        max_tracking_ade_m=1,
        max_tracking_error_m=2,
        max_tracking_heading_error_rad=0.35,
        min_pdm=0.95,
    )
    cfg = {"score_batch": 2, "digest": "test"}
    space = SimpleNamespace(batch=lambda t: t[:, 0, 0] != 2)
    with (
        patch(
            "drivesta_feedback.batch.simscale_recovery.physical_mask",
            side_effect=lambda t, *a: (np.ones(len(t), bool), {}),
        ),
        patch("drivesta_feedback.batch.filter_targets.tracking_errors", return_value=[0, 0, 0]),
    ):
        first = relabel_existing(
            old, tmp_path, cfg, Evaluator(), space, None, policy, None, None, None
        )
        assert first["eligible"].tolist() == [True, False, False]
        assert first["labels"][1, -1] == 0
        second = relabel_existing(old, tmp_path, cfg, None, None, None, None, None, None, None)
    for k in old:
        assert np.array_equal(first[k], second[k])
        assert np.array_equal(old[k], untouched[k])
    changed = copy.deepcopy(old)
    changed["trajectories"][0, 0, 0] = 99
    with pytest.raises(ValueError):
        relabel_existing(changed, tmp_path, cfg, None, None, None, None, None, None, None)
