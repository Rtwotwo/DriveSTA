"""Audit unique GT-witnessed prefixes that rejoin the original directed route."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import traceback

from .simscale_directed_repair import complete_source_suffix
from .simscale_geometry import load_scene
from drivesta_feedback.batch.store import read_json, write_json


def ordered_membership(route, groups):
    """Require a nondecreasing graph position for every observed GT waypoint."""
    possible = {-1}
    for group in groups:
        possible = {
            i
            for i, node in enumerate(route)
            if node in group and any(previous <= i for previous in possible)
        }
        if not possible:
            return False
    return True


def witnessed_choice(audit, successors, connector_supports, max_edges=6):
    raw = audit["raw_roadblocks"]
    groups = [{x["roadblock_id"] for x in g} for g in audit["gt_directed_membership"]]
    if not raw or not all(groups):
        return None, "missing_directed_gt_witness", []
    witnessed = set.union(*groups)
    routes = []
    queue = [[a] for a in sorted(groups[0])]
    while queue:
        prefix = queue.pop()
        node = prefix[-1]
        if node in raw:
            if len(prefix) == 1:
                # This mode requires a new source-GT-supported initial prefix.
                # Restoring only a distant suffix connector is a separate mode.
                continue
            suffix = complete_source_suffix(
                raw[raw.index(node) :], successors, connector_supports, max_edges
            )
            if suffix is None:
                continue
            route = prefix + suffix[1:]
            if len(set(route)) != len(route) or not ordered_membership(route, groups):
                continue
            if route not in routes:
                routes.append(route)
            if len(routes) > 1:
                return None, "multiple_gt_supported_routes", routes
        elif len(prefix) - 1 < max_edges:
            for nxt in sorted(set(successors(node))):
                if nxt not in prefix and (nxt in witnessed or nxt in raw):
                    queue.append(prefix + [nxt])
    if len(routes) != 1:
        return None, "no_gt_supported_rejoining_route", []
    return routes[0], "unique_gt_supported_rejoining_route", routes


def inspect(audit, root):
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L

    token = audit["token"]
    identity = read_json(Path(root) / "scenes/train" / token / "contract.json")["identity"]
    scene = load_scene(identity)
    blocks = {}

    def block(k):
        if k not in blocks:
            blocks[k] = scene.map_api.get_map_object(
                k, L.ROADBLOCK
            ) or scene.map_api.get_map_object(k, L.ROADBLOCK_CONNECTOR)
        if blocks[k] is None:
            raise ValueError("Missing source roadblock " + k)
        return blocks[k]

    def successors(k):
        return [x.id for x in block(k).outgoing_edges]

    witnessed = {x["roadblock_id"] for g in audit["gt_directed_membership"] for x in g}
    cached = set(audit["cached_route_lane_ids"])

    def supports(k):
        return k in witnessed or bool({x.id for x in block(k).interior_edges} & cached)

    route, reason, choices = witnessed_choice(audit, successors, supports)
    added = [] if route is None else [x for x in route if x not in audit["raw_roadblocks"]]
    return dict(
        token=token,
        reason=reason,
        route=route,
        added_roadblocks=added,
        changed=bool(added),
        initial_valid=audit["initial"]["valid"],
        choices=choices,
        selected_without_scores=True,
    )


def worker(args):
    try:
        return inspect(*args)
    except Exception:
        return dict(token=args[0]["token"], error=traceback.format_exc())


def main():
    from drivesta_feedback.paths import bootstrap_navsim

    bootstrap_navsim()
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "audit", "tokens", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    tokens = set(a.tokens.read_text().split())
    audits = {
        x["token"]: x
        for x in map(json.loads, a.audit.read_text().splitlines())
        if x["token"] in tokens
    }
    if set(audits) != tokens:
        raise ValueError("Missing source graph audits")
    a.output.mkdir(parents=True, exist_ok=True)
    journal = a.output / "audit.jsonl"
    done = (
        {x["token"]: x for x in map(json.loads, journal.read_text().splitlines())}
        if journal.exists()
        else {}
    )
    with (
        ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool,
        journal.open("a") as f,
    ):
        futures = [
            pool.submit(worker, (audits[t], str(a.root))) for t in sorted(tokens - set(done))
        ]
        for future in as_completed(futures):
            row = future.result()
            done[row["token"]] = row
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(
                json.dumps(
                    dict(
                        processed=len(done),
                        total=len(tokens),
                        changed=sum(x.get("changed", False) for x in done.values()),
                    )
                ),
                flush=True,
            )
    from collections import Counter

    summary = dict(
        scenes=len(done),
        changed=sum(x.get("changed", False) for x in done.values()),
        reasons=dict(Counter(x.get("reason", "error") for x in done.values())),
        errors=[x for x in done.values() if "error" in x],
    )
    write_json(a.output / "summary.json", summary)
    if summary["errors"]:
        raise RuntimeError("Source graph audit failures")


if __name__ == "__main__":
    main()
