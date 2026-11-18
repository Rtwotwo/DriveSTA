"""Exact array implementation of expensive official scorer operations.

The original official simulator, collision/TTC/history kernels and thresholds
are unchanged. This module is validated against the immutable v1 runner.
"""

from .scoring import *
import shapely
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from navsim.v2.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    BBCoordsIndex,
    EgoAreaIndex,
)


def trajectory_array(trajectory, sampling, start_time):
    # Identical absolute float-time -> integer-us rounding and clipping as
    # get_trajectory_as_array. Interpolate directly instead of constructing
    # 41 EgoState/CarFootprint/DynamicCarState objects per proposal.
    times = (
        np.arange(0, sampling.time_horizon + sampling.interval_length, sampling.interval_length)
        + start_time.time_s
    )
    times = np.array([int(t * 1e6) for t in times])
    times = np.clip(times, trajectory.start_time.time_us, trajectory.end_time.time_us)
    linear = trajectory._function_interp_linear(times)
    angular = trajectory._angular_interpolator.interpolate(times)
    result = np.zeros((len(times), 11), dtype=np.float64)
    # Reproduce CarFootprint rear axle -> center -> rear axle arithmetic,
    # including its float64 rounding, instead of treating it as an identity.
    h = angular[:, 0]
    d = trajectory._fixed_state[0].rear_axle_to_center
    cosine = np.fromiter((np.cos(v) for v in h), dtype=np.float64, count=len(h))
    sine = np.fromiter((np.sin(v) for v in h), dtype=np.float64, count=len(h))
    result[:, 0] = (linear[:, 1] + d * cosine) + (-float(d) * cosine)
    result[:, 1] = (linear[:, 2] + d * sine) + (-float(d) * sine)
    result[:, 2] = angular[:, 0]
    result[:, 3:8] = linear[:, 3:8]
    # EgoState.from_split_state defaults steering rate/angular v/a to zero.
    return result


class ArrayPairScorer(IndependentPairScorer):
    def _calculate_ego_area(self):
        # Same strict contains test as the official implementation. Skip only
        # points outside each polygon's axis-aligned bounding box (no sampling).
        m = self._drivable_area_map
        points = self._ego_coords.reshape(-1, 2)
        boxes = shapely.bounds(m._geometries)
        membership = np.zeros((len(boxes), len(points)), dtype=bool)
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        possible = np.flatnonzero(
            (boxes[:, 0] <= xmax)
            & (boxes[:, 2] >= xmin)
            & (boxes[:, 1] <= ymax)
            & (boxes[:, 3] >= ymin)
        )
        for i in possible:
            x0, y0, x1, y1 = boxes[i]
            inside_box = (
                (points[:, 0] >= x0)
                & (points[:, 0] <= x1)
                & (points[:, 1] >= y0)
                & (points[:, 1] <= y1)
            )
            ids = np.flatnonzero(inside_box)
            membership[i, ids] = shapely.contains_xy(
                m._geometries[i], points[ids, 0], points[ids, 1]
            )
        inside = membership.reshape((len(boxes),) + self._ego_coords.shape[:-1]).transpose(
            1, 2, 0, 3
        )
        road_ids = m.get_indices_of_map_type(
            [
                SemanticMapLayer.ROADBLOCK,
                SemanticMapLayer.INTERSECTION,
                SemanticMapLayer.DRIVABLE_AREA,
                SemanticMapLayer.CARPARK_AREA,
            ]
        )
        lane_ids = m.get_indices_of_map_type(
            [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        )
        route_ids = [i for i in lane_ids if m.tokens[i] in self._route_lane_ids]
        corners = inside[..., :-1]
        centers = inside[..., -1]
        corner_count = corners[:, :, lane_ids].sum(axis=-1)
        multiple = ((corner_count > 0).sum(axis=-1) > 1) & np.all(corner_count != 4, axis=-1)
        non_drivable = (corners[:, :, road_ids].sum(axis=-2) > 0).sum(axis=-1) < 4
        oncoming = centers[..., route_ids].sum(axis=-1) == 0
        self._ego_areas[multiple, EgoAreaIndex.MULTIPLE_LANES] = True
        self._ego_areas[non_drivable, EgoAreaIndex.NON_DRIVABLE_AREA] = True
        self._ego_areas[oncoming, EgoAreaIndex.ONCOMING_TRAFFIC] = True
        intersection_ids = m.get_indices_of_map_type([SemanticMapLayer.INTERSECTION])
        self._intersection_cache = centers[..., intersection_ids].any(axis=-1)
        self._center_points = shapely.points(
            self._ego_coords[:, :, BBCoordsIndex.CENTER].reshape(-1, 2)
        )

    def _intersection_mask(self):
        if self._intersection_cache is None:
            centers = self._ego_coords[:, :, BBCoordsIndex.CENTER]
            self._center_points = shapely.points(centers.reshape(-1, 2))
            hits = self._drivable_area_map._str_tree.query(self._center_points, predicate="within")
            mask = np.zeros(len(self._center_points), dtype=bool)
            if hits.shape[1]:
                kinds = np.asarray(self._drivable_area_map._map_types)
                valid = kinds[hits[1]] == SemanticMapLayer.INTERSECTION
                mask[hits[0, valid]] = True
            self._intersection_cache = mask.reshape(centers.shape[:2])
        return self._intersection_cache

    def _calculate_driving_direction_compliance(self):
        centers = self._ego_coords[:, :, BBCoordsIndex.CENTER]
        progress = np.zeros(centers.shape[:2], dtype=np.float64)
        progress[:, 1:] = np.linalg.norm(centers[:, 1:] - centers[:, :-1], axis=-1)
        oncoming = self._ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC]
        progress[(~oncoming) | self._intersection_mask()] = 0.0
        horizon = int(
            self._config.driving_direction_horizon / self.proposal_sampling.interval_length
        )
        # Preserve the official summation order near threshold boundaries.
        windows = np.concatenate(
            [
                progress[:, max(0, i - horizon) : i + 1].sum(axis=-1)[..., None]
                for i in range(progress.shape[-1])
            ],
            dtype=np.float64,
            axis=-1,
        )
        maxima = windows.max(axis=-1)
        self._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION] = np.where(
            maxima < self._config.driving_direction_compliance_threshold,
            1.0,
            np.where(maxima < self._config.driving_direction_violation_threshold, 0.5, 0.0),
        )

    def _calculate_lane_keeping(self):
        intersections = self._intersection_mask()
        # Distance to a polyline is the minimum exact distance to its segments.
        # STRtree nearest prunes distant segments; it does not approximate them.
        ids, d = self._centerline_segments.query_nearest(
            self._center_points, return_distance=True, all_matches=False
        )
        flat = np.empty(len(self._center_points), dtype=np.float64)
        flat[ids[0]] = d
        distances = flat.reshape(intersections.shape)
        counter = np.zeros(self._num_proposals, dtype=np.int64)
        failed = np.zeros(self._num_proposals, dtype=bool)
        required = int(
            np.ceil(
                self._config.lane_keeping_horizon_window / self.proposal_sampling.interval_length
            )
        )
        exceeds = distances > self._config.lane_keeping_deviation_limit
        for i in range(distances.shape[1]):
            # Official intersection handling skips the point WITHOUT resetting
            # the existing consecutive-deviation counter.
            active = ~intersections[:, i]
            counter[active] = np.where(exceeds[active, i], counter[active] + 1, 0)
            failed |= active & (counter >= required)
        self._weighted_metrics[WeightedMetricIndex.LANE_KEEPING] = (~failed).astype(np.float64)

    def score_array(self, states, cache, environment):
        cache.observation.update_detections_tracks(detection_tracks=environment)
        self._reset(
            states,
            cache.observation,
            cache.centerline,
            cache.route_lane_ids,
            cache.drivable_area_map,
            cache.past_human_trajectory,
        )
        self._intersection_cache = None
        self._calculate_ego_area()
        self._calculate_no_at_fault_collision()
        self._calculate_drivable_area_compliance()
        self._calculate_traffic_light_compliance()
        self._calculate_driving_direction_compliance()
        self._calculate_progress()
        self._calculate_ttc()
        self._calculate_lane_keeping()
        self._calculate_history_comfort()
        self._aggregate_pdm_scores()
        mi = [
            MultiMetricIndex.NO_COLLISION,
            MultiMetricIndex.DRIVABLE_AREA,
            MultiMetricIndex.DRIVING_DIRECTION,
            MultiMetricIndex.TRAFFIC_LIGHT_COMPLIANCE,
        ]
        wi = [
            WeightedMetricIndex.PROGRESS,
            WeightedMetricIndex.TTC,
            WeightedMetricIndex.LANE_KEEPING,
            WeightedMetricIndex.HISTORY_COMFORT,
        ]
        return np.concatenate([self._multi_metrics[mi].T, self._weighted_metrics[wi].T], axis=1)


class FastArchiveScorer(ArchiveScorer):
    def __init__(self, cache):
        shapely.prepare(cache.drivable_area_map._geometries)
        super().__init__(cache)
        self.scorer = ArrayPairScorer(SAMPLING, PDMScorerConfig(human_penalty_filter=True))
        coordinates = np.asarray(cache.centerline.linestring.coords)
        segments = shapely.linestrings(np.stack([coordinates[:-1], coordinates[1:]], axis=1))
        self.scorer._centerline_segments = shapely.STRtree(segments)

    def simulate(self, trajectories, chunk=128):
        c = self.cache
        result = []
        for start in range(0, len(trajectories), chunk):
            arrays = [self.ref]
            for t in trajectories[start : start + chunk]:
                trajectory = transform_trajectory(
                    Trajectory(np.asarray(t, dtype=np.float64), OUTPUT_SAMPLING), c.ego_state
                )
                arrays.append(trajectory_array(trajectory, SAMPLING, c.ego_state.time_point))
            result.append(self.sim.simulate_proposals(np.stack(arrays), c.ego_state)[1:])
        return np.concatenate(result)

    def score(self, trajectories, chunk=128, return_states=False):
        c = self.cache
        raw = []
        states_all = []
        for start in range(0, len(trajectories), chunk):
            arrays = [self.ref]
            for t in trajectories[start : start + chunk]:
                trajectory = transform_trajectory(
                    Trajectory(np.asarray(t, dtype=np.float64), OUTPUT_SAMPLING), c.ego_state
                )
                arrays.append(trajectory_array(trajectory, SAMPLING, c.ego_state.time_point))
            states = self.sim.simulate_proposals(np.stack(arrays), c.ego_state)
            raw.append(self.scorer.score_array(states, c, self.environment)[1:])
            if return_states:
                states_all.append(states[1:])
        raw = np.concatenate(raw)
        filtered = raw.copy()
        filtered[:, self.filter_mask] = 1.0
        assert np.isfinite(filtered).all() and filtered.min() >= 0 and filtered.max() <= 1
        return raw, filtered, np.concatenate(states_all) if return_states else None
