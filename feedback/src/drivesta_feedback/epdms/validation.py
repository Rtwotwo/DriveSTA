"""Independent official per-plan simulation and official one-stage aggregation."""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .dataset import read_scene, cache_for, BASE, SAMPLING, OUTPUT_SAMPLING, planned_space_many
from .scoring import ArchiveScorer, pdm_score, Trajectory, component_rows
from ..batch.filter_targets import planned_space_checker
from ..batch.store import write_json
from navsim.evaluate.epdms.one_stage import (
    infer_start_adjacent_mapping,
    create_scene_aggregators,
    compute_final_scores,
)


def validate(out, full_high=False, max_logs=0):
    out = Path(out)
    summary = json.loads((out / "summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("Pilot not complete")
    groups = {}
    for row in summary["scenes"]:
        groups.setdefault(row["log_name"], []).append(row)
    errors = []
    checked = 0
    high_replayed = 0
    space_checked = 0
    component_max = 0.0
    score_max = 0.0
    selected_groups = sorted(groups.items())[: max_logs or len(groups)]
    checked_scenes = 0
    for log, rows in selected_groups:
        checked_scenes += len(rows)
        rows.sort(key=lambda r: r["start_time"])
        ids = set([0, 1, 767, 768, 1535, 2303, 3071])
        loaded = {}
        for row in rows:
            info, bank, high = read_scene(out / "scenes" / row["split"] / row["token"])
            ids.update(
                high["query_indices"].tolist() if full_high else high["query_indices"][:2].tolist()
            )
            loaded[row["token"]] = (bank, high)
        policies = {i: [] for i in sorted(ids)}
        for row in rows:
            cache, _ = cache_for(row)
            engine = ArchiveScorer(cache)
            bank, high = loaded[row["token"]]
            # Check batch geometry against the prior scalar implementation.
            check_ids = sorted(set(list(sorted(ids))[:3] + high["query_indices"][:3].tolist()))
            scalar = planned_space_checker(cache)
            batch = planned_space_many(cache, bank["trajectories"][check_ids])
            assert np.array_equal(
                batch, np.array([scalar(bank["trajectories"][i]) for i in check_ids])
            )
            space_checked += len(check_ids)
            for i in sorted(ids):
                result, states = pdm_score(
                    cache,
                    Trajectory(bank["trajectories"][i].astype(float), OUTPUT_SAMPLING),
                    SAMPLING,
                    engine.sim,
                    engine.official,
                    engine.traffic,
                )
                err = float(abs(component_rows([result])[0] - bank["components"][i]).max())
                component_max = max(component_max, err)
                assert err <= 1e-10, (row["token"], i, err)
                for key, value in [
                    ("token", row["token"]),
                    ("log_name", cache.log_name),
                    ("frame_type", cache.scene_type),
                    ("start_time", cache.timepoint.time_s),
                ]:
                    result[key] = value
                result["ego_simulated_states"] = [states]
                policies[i].append(result)
                checked += 1
                high_replayed += int(i in high["query_indices"])
        for i, rs in policies.items():
            df = pd.concat(rs, ignore_index=True)
            pairs = infer_start_adjacent_mapping(df)
            if pairs:
                df = create_scene_aggregators(pairs, df, SAMPLING)
            else:
                df["two_frame_extended_comfort"] = np.nan
            df = compute_final_scores(df)
            for _, row in df.iterrows():
                saved = loaded[row.token][0][i]
                err = abs(float(row.score) - float(saved["epdms"]))
                score_max = max(score_max, err)
                assert err <= 1e-10, (row.token, i, err)
                ec = float(row.two_frame_extended_comfort)
                assert (np.isnan(ec) and np.isnan(saved["ec"])) or ec == saved["ec"]
    result = dict(
        scenes=checked_scenes,
        official_per_plan_calls=checked,
        high_targets_independently_replayed=high_replayed,
        all_high_targets=full_high,
        geometry_scalar_comparisons=space_checked,
        max_component_error=component_max,
        max_final_epdms_error=score_max,
        official_ec_exact=True,
        readback_all_checked_scenes=True,
    )
    write_json(out / "official_validation.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--all-high", action="store_true")
    p.add_argument("--max-logs", type=int, default=0)
    a = p.parse_args()
    validate(a.out, a.all_high, a.max_logs)
