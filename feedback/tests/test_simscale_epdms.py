"""Protocol checks with artificial arrays; no scene generation or production launch."""

import ast
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import textwrap

import numpy as np
import pytest
from shapely import Point
from drivesta_feedback.epdms.simscale import (
    DTYPE,
    SCHEMA,
    PROTOCOL,
    atomic_npy,
    atomic_npz,
    read_simscale_scene,
    score_from_components,
    export_views,
    training_targets,
    representative_indices,
)
from drivesta_feedback.epdms.tail_gate import FullHorizonTTC
from drivesta_feedback.epdms.scoring import PDMScorer, PDMScorerConfig, SAMPLING
from navsim.v2.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex


def test_tail_preserves_official_collision_logic():
    official = ast.parse(textwrap.dedent(inspect.getsource(PDMScorer._calculate_ttc)))
    extra = ast.parse(textwrap.dedent(inspect.getsource(FullHorizonTTC._calculate_ttc)))

    def clean(tree):
        nodes = []
        for node in tree.body[0].body:
            if isinstance(node, ast.Assign):
                names = [getattr(x, "id", None) for x in node.targets]
                if "required" in names:
                    continue
                if "n_proposal_steps_to_evaluate" in names:
                    node.value = ast.Constant(40)
            if isinstance(node, ast.If) and "required" in ast.dump(node.test):
                continue
            nodes.append(node)
        tree.body[0].body = nodes
        return ast.dump(tree, include_attributes=False)

    assert clean(official) == clean(extra)


def mock_ttc(hit_at=None, length=51, speed=5.0):
    class Occupancy:
        tokens = ["actor"]

        def __init__(self, i):
            self.i = i

        def query(self, polygons, predicate):
            return (
                np.array([[0], [0]], dtype=int) if self.i == hit_at else np.empty((2, 0), dtype=int)
            )

        def __getitem__(self, token):
            return SimpleNamespace(centroid=Point(100.0, 0.0))

    class Observation:
        _global_to_local_idcs = list(range(length))
        collided_track_ids = []
        red_light_token = "red_light"
        unique_objects = {
            "actor": SimpleNamespace(box=SimpleNamespace(center=SimpleNamespace(heading=0.0)))
        }

        def __getitem__(self, i):
            assert 0 <= i < length
            return Occupancy(i)

    states = np.zeros((1, 41, 11))
    states[:, :, 3] = speed
    coords = np.tile(np.array([[1, 1], [1, -1], [-1, -1], [-1, 1], [0, 0]], float), (1, 41, 1, 1))
    return SimpleNamespace(
        _num_proposals=1,
        _states=states,
        _ego_coords=coords,
        _ego_areas=np.zeros((1, 41, 8), bool),
        _ttc_time_idcs=np.full(1, np.inf),
        _weighted_metrics=np.zeros((8, 1)),
        _observation=Observation(),
        _drivable_area_map=SimpleNamespace(is_in_layer=lambda *args, **kwargs: False),
        _config=PDMScorerConfig(),
        proposal_sampling=SAMPLING,
    )


@pytest.mark.parametrize(
    "hit,official_expected,tail_expected", [(None, 1, 1), (30, 0, 0), (49, 1, 0)]
)
def test_inferred_tail_is_separate_from_official_metric(hit, official_expected, tail_expected):
    official = mock_ttc(hit)
    tail = mock_ttc(hit)
    PDMScorer._calculate_ttc(official)
    FullHorizonTTC._calculate_ttc(tail)
    assert official._weighted_metrics[WeightedMetricIndex.TTC, 0] == official_expected
    assert tail._weighted_metrics[WeightedMetricIndex.TTC, 0] == tail_expected


def test_short_environment_rejected():
    with pytest.raises(ValueError, match="4.9"):
        FullHorizonTTC._calculate_ttc(mock_ttc(length=41))


def test_stopped_ego_keeps_official_rule():
    scorer = mock_ttc(49, speed=0)
    FullHorizonTTC._calculate_ttc(scorer)
    assert scorer._weighted_metrics[WeightedMetricIndex.TTC, 0] == 1.0


def fake_bank(path, ec=np.nan, tail_value=1.0, progress=1.0):
    full = np.zeros(3072, dtype=DTYPE)
    full["family_id"] = np.arange(3072)
    full["components"] = 1.0
    full["components"][:, 4] = progress
    full["components_raw"] = full["components"]
    full["epdms"] = score_from_components(full["components"])
    full["ec"] = ec
    full["admissible"][0] = True
    atomic_npy(path / "candidates.npy", full)
    ids = np.array([0])
    pair = np.full((3072, 2), np.nan)
    pair[0] = [1.0, tail_value]
    atomic_npz(
        path / "tail_checks.npz",
        ttc_nominal=pair[:, 0],
        ttc_alternative=pair[:, 1],
        evaluated=np.isfinite(pair).all(1),
    )
    atomic_npz(
        path / "high_score.npz",
        query_indices=ids,
        **{
            k: full[k][ids]
            for k in ("trajectories", "epdms", "components", "components_raw", "ec", "family_id")
        },
        tail_ttc_nominal=pair[ids, 0],
        tail_ttc_alternative=pair[ids, 1],
    )
    (path / "summary.json").write_text(
        json.dumps(
            dict(
                schema=SCHEMA,
                protocol=PROTOCOL,
                selected_count=1,
                ec_defined=False,
                previous_token=None,
            )
        )
    )


def test_reader_retains_missing_ec(tmp_path):
    fake_bank(tmp_path)
    _, full, high = read_simscale_scene(tmp_path, verify=False)
    assert np.isnan(full["ec"]).all() and np.isnan(high["ec"]).all()


@pytest.mark.parametrize("kwargs", [{"ec": 1.0}, {"tail_value": 0.0}, {"progress": 0.8}])
def test_reader_rejects_invalid_targets(tmp_path, kwargs):
    fake_bank(tmp_path, **kwargs)
    with pytest.raises(ValueError):
        read_simscale_scene(tmp_path, verify=False)


def test_eight_target_manifest_is_distinct_from_full256(tmp_path):
    results = [
        dict(
            token=str(i),
            split="train",
            source_group=0,
            selected_count=n,
            representative_count=min(n, 32),
            best_epdms=100.0,
            funnel={},
        )
        for i, n in enumerate([0, 1, 7, 8, 32, 255, 256])
    ]
    export_views(tmp_path, results)
    assert (tmp_path / "train_at_least8_tokens.txt").read_text().splitlines() == [
        "3",
        "4",
        "5",
        "6",
    ]
    assert (tmp_path / "train_full256_tokens.txt").read_text().splitlines() == ["6"]
    assert (tmp_path / "under8_tokens.txt").read_text().splitlines() == ["0", "1", "2"]
    assert len((tmp_path / "shortfall_tokens.txt").read_text().splitlines()) == 6
    assert len((tmp_path / "training_manifest.jsonl").read_text().splitlines()) == 4


@pytest.mark.parametrize("count", [0, 1, 7, 8, 32])
def test_target_view_masks_short_banks_without_repeating(monkeypatch, tmp_path, count):
    from drivesta_feedback.epdms import simscale

    paths = np.arange(count * 8 * 3, dtype=np.float32).reshape(count, 8, 3)
    monkeypatch.setattr(
        simscale,
        "read_simscale_scene",
        lambda _: (
            {"token": "test"},
            {"trajectories": paths},
            {"query_indices": np.arange(count), "trajectories": paths},
        ),
    )
    np.savez(tmp_path / "generation_targets.npz", query_indices=np.arange(count))
    view = training_targets(tmp_path, epoch=2)
    n = min(8, count)
    assert view["mask"].sum() == n
    assert len(np.unique(view["query_indices"][view["mask"]])) == n
    assert (view["query_indices"][~view["mask"]] == -1).all()
    np.testing.assert_array_equal(
        view["trajectories"][view["mask"]], paths[view["query_indices"][view["mask"]]]
    )


def test_representatives_reject_near_copies_and_unsafe_candidates():
    paths = np.zeros((7, 8, 3))
    paths[:, :, 0] = np.arange(7)[:, None] * 0.1
    scores = np.ones(7)
    valid = np.array([1, 1, 1, 1, 1, 1, 0], dtype=bool)
    ids = representative_indices(paths, scores, valid)
    assert ids.tolist() == [0, 5]
    assert 6 not in ids
