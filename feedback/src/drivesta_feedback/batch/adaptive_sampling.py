"""Reference-preserving perturbations and feedback-guided proposal allocation."""

import numpy as np


def perturb_preserving_heading(reference, coefficients):
    """Keep stored body headings at zero offset; never turn reverse motion into pi.

    Heading changes describe the added lateral motion relative to the stored
    body frame. The unchanged physical and tracking checks judge every result.
    """
    from scipy.interpolate import CubicSpline

    ref = np.asarray(reference, float)
    coeff = np.asarray(coefficients, float).reshape(-1, 6)
    t = np.arange(9) * 0.5
    u = t / 4
    base = np.vstack([np.zeros(3), ref])
    basis = np.stack(
        [
            10 * u**3 - 15 * u**4 + 6 * u**5,
            64 * u**3 * (1 - u) ** 3,
            64 * u**3 * (1 - u) ** 3 * (2 * u - 1),
        ],
        -1,
    )
    tangent = np.array([np.cos(ref[-1, 2]), np.sin(ref[-1, 2])])
    normal = np.array([-tangent[1], tangent[0]])
    offsets = np.einsum("tk,nkd->ntd", basis, coeff.reshape(-1, 3, 2))
    delta = offsets[..., :1] * tangent + offsets[..., 1:] * normal
    xy = base[None, :, :2] + delta
    dv = CubicSpline(t, delta, axis=1)(t, 1)
    bv = CubicSpline(t, base[:, :2], axis=0)(t, 1)
    body = np.stack([np.cos(base[:, 2]), np.sin(base[:, 2])], -1)
    left = np.stack([-body[:, 1], body[:, 0]], -1)
    longitudinal = np.maximum((bv * body).sum(-1), 0.2)[None] + (dv * body).sum(-1)
    angle = np.arctan2((dv * left).sum(-1), np.maximum(longitudinal, 0.2))
    heading = base[None, :, 2] + angle
    heading = np.arctan2(np.sin(heading), np.cos(heading))
    return np.concatenate([xy[:, 1:], heading[:, 1:, None]], -1).astype(np.float32)


def forward_seeds(expert, velocity, acceleration):
    """Integrate nonnegative speed with bounded acceleration/jerk and turn rate.

    This only proposes paths. Measured initial velocity is retained in the
    independent admission check and official tracking simulator.
    """
    ref = np.vstack([np.zeros(3), np.asarray(expert, float)])
    times = np.arange(41) * 0.1
    yaw_ref = np.interp(times, np.arange(9) * 0.5, np.unwrap(ref[:, 2]))
    paths = []
    metadata = []
    for target_accel in (-5.0, -3.0, -1.5, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0):
        for turn_gain in (0.0, 0.5, 0.75, 1.0, 1.25):
            v = max(0.0, float(velocity[0]))
            a = float(np.clip(acceleration[0], -3.0, 3.0))
            x = y = h = 0.0
            states = [[x, y, h]]
            for k in range(1, 41):
                a += np.clip(target_accel - a, -0.5, 0.5)
                new_v = max(0.0, min(25.0, v + a * 0.1))
                desired = turn_gain * yaw_ref[k]
                err = np.arctan2(np.sin(desired - h), np.cos(desired - h))
                rate = float(
                    np.clip(err / 0.5, -min(0.5, 3.5 / max(v, 1.0)), min(0.5, 3.5 / max(v, 1.0)))
                )
                speed = (v + new_v) / 2
                rate *= min(speed, 1.0)
                mid = h + rate * 0.05
                x += speed * np.cos(mid) * 0.1
                y += speed * np.sin(mid) * 0.1
                h += rate * 0.1
                states.append([x, y, np.arctan2(np.sin(h), np.cos(h))])
                v = new_v
            path = np.asarray(states, dtype=np.float32)[5::5]
            if not any(np.array_equal(path, p) for p in paths):
                paths.append(path)
                metadata.append(
                    dict(
                        parent="forward_integrated",
                        target_accel=target_accel,
                        turn_gain=turn_gain,
                        time_fraction=1.0,
                    )
                )
    return np.asarray(paths), metadata


def shuffled_parameters(count, seed):
    from .physical_sampling import proposal_parameters

    values = proposal_parameters(count, seed)
    # Residue classes of a Sobol index are not themselves balanced samples.
    # Shuffle before assigning scales/parents so each arm spans both signs.
    np.random.default_rng(seed ^ 0x51A73).shuffle(values)
    return values


def diverse_indices(paths, scores, count=12, separation=0.05):
    ordered = np.argsort(-np.asarray(scores), kind="stable")
    selected = []
    for i in ordered:
        if (
            not selected
            or np.linalg.norm(paths[selected, :, :2] - paths[i, :, :2], axis=-1).mean(-1).min()
            >= separation
        ):
            selected.append(int(i))
        if len(selected) == count:
            break
    return selected


def quality(labels, alternative, tracking, limits, spatial):
    """Search priority only; the unchanged boolean admission decides acceptance."""
    score = np.minimum(labels[:, -1], alternative[:, -1])
    failures = (labels[:, [0, 1, 3, 4, 5]] < 1).sum(1) + (alternative[:, [0, 1, 3, 4, 5]] < 1).sum(
        1
    )
    return (
        score
        - 0.15 * failures
        - 0.2 * np.maximum(tracking / np.asarray(limits) - 1, 0).sum(1)
        - 0.4 * (~np.asarray(spatial, bool))
    )


class PreparedSpace:
    """Same official plan interpolation and polygon predicates, prepared once."""

    def __init__(self, cache):
        import shapely
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer
        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        from .filter_targets import MissingPlannedGeometry

        self.cache = cache
        self.vehicle = get_pacifica_parameters()
        self.sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        m = cache.drivable_area_map
        road_ids = m.get_indices_of_map_type(
            [
                SemanticMapLayer.ROADBLOCK,
                SemanticMapLayer.LANE,
                SemanticMapLayer.LANE_CONNECTOR,
                SemanticMapLayer.INTERSECTION,
                SemanticMapLayer.CARPARK_AREA,
            ]
        )
        self.omitted = [t for t in m.tokens if not m[t].is_valid or m[t].is_empty]
        self.road = shapely.union_all(
            [m[m.tokens[i]] for i in road_ids if m.tokens[i] not in self.omitted]
        )
        self.route = shapely.union_all(
            [m[t] for t in m.tokens if t in cache.route_lane_ids and t not in self.omitted]
        )
        if self.road.is_empty or self.route.is_empty:
            raise MissingPlannedGeometry("Missing drivable/route geometry")
        shapely.prepare(self.road)
        shapely.prepare(self.route)

    def batch(self, paths):
        import shapely
        from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
        from navsim.common.dataclasses import Trajectory
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
            state_array_to_coords_array,
        )
        from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex

        if not len(paths):
            return np.zeros(0, bool)
        states = np.asarray(
            [
                get_trajectory_as_array(
                    transform_trajectory(Trajectory(t), self.cache.ego_state),
                    self.sampling,
                    self.cache.ego_state.time_point,
                )
                for t in paths
            ]
        )
        coords = state_array_to_coords_array(states, self.vehicle)
        road = shapely.covers(self.road, shapely.points(coords[:, :, :4, :])).all((1, 2))
        route = shapely.covers(
            self.route, shapely.points(coords[:, :, BBCoordsIndex.CENTER, :])
        ).all(1)
        return road & route

    def __call__(self, path):
        return bool(self.batch(np.asarray(path)[None])[0])
