import unittest
from types import SimpleNamespace
import numpy as np
from drivesta_feedback.batch.actor_tail import (
    TailPolicy,
    fit_motion,
    predict_motion,
    choose_motion,
    extend_observation,
)


class TailTests(unittest.TestCase):
    def setUp(self):
        self.p = TailPolicy()

    def test_constant_velocity(self):
        t = np.arange(16) * 0.1
        xy = np.c_[t * 8, t * 2]
        m = fit_motion(t, xy, np.zeros(16), "cv", self.p)
        x, w = predict_motion(m, [0, 0.5, 1], self.p)
        np.testing.assert_allclose(x, xy[-1] + np.array([[0, 0], [4, 1], [8, 2]]), atol=1e-10)
        np.testing.assert_array_equal(w, 0)

    def test_stopped(self):
        t = np.arange(16) * 0.1
        xy = np.tile([5.0, 6.0], (16, 1))
        for mode in ("cv", "turn_accel"):
            m = fit_motion(t, xy, np.zeros(16), mode, self.p)
            x, _ = predict_motion(m, [1], self.p)
            np.testing.assert_array_equal(x, [[5.0, 6.0]])

    def test_braking_does_not_reverse(self):
        m = dict(center=[0, 0], speed=0.2, direction=0, accel=-6, yaw_rate=0, max_speed=40)
        x, _ = predict_motion(m, np.arange(11) * 0.1, self.p)
        self.assertTrue((np.diff(x[:, 0]) >= 0).all())
        self.assertLess(x[-1, 0], 0.03)

    def test_turn_and_acceleration_are_bounded(self):
        t = np.arange(16) * 0.1
        xy = np.c_[2 * t + 30 * t * t, np.zeros(16)]
        m = fit_motion(t, xy, 5 * t, "turn_accel", self.p)
        self.assertLessEqual(m["speed"], 40)
        self.assertLessEqual(m["accel"], 3)
        self.assertLessEqual(abs(m["yaw_rate"]) * m["speed"], 4 + 1e-10)

    def test_stationary_type_ignores_spurious_motion(self):
        m = fit_motion([0, 1], [[0, 0], [5, 0]], [0, 0], "cv", self.p, "TRAFFIC_CONE")
        self.assertEqual(m["speed"], 0)

    def test_new_actor_fallback(self):
        m = fit_motion([0], [[1, 2]], [0], "cv", self.p, "VEHICLE", [2, 0])
        x, _ = predict_motion(m, [1], self.p)
        np.testing.assert_allclose(x, [[3, 2]])

    def test_no_time_reversal(self):
        with self.assertRaises(ValueError):
            fit_motion([1, 0], [[0, 0], [1, 0]], [0, 0], "cv", self.p)

    def test_invalid_newborn_velocity_not_clipped_to_high_speed(self):
        m = fit_motion([0], [[1, 2]], [0], "cv", self.p, "VEHICLE", [40000, -60000])
        self.assertTrue(m["fallback_rejected"])
        self.assertEqual(m["speed"], 0)

    def test_cv_preferred_without_turn(self):
        t = np.arange(16) * 0.1
        m, _, meta = choose_motion(t, np.c_[t * 5, t * 0], t * 0, self.p, "VEHICLE", [0, 0])
        self.assertEqual(m["mode"], "cv")
        self.assertIsNotNone(meta["backtest_errors"][0])

    def observation(self, after4=False):
        from shapely.geometry import box
        from shapely.affinity import translate
        from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
            PDMOccupancyMap,
        )

        class Observation:
            def __getitem__(self, i):
                return self._occupancy_maps[self._global_to_local_idcs[i]]

            @property
            def unique_objects(self):
                return self._unique_objects

            @property
            def red_light_token(self):
                return "red_light"

        o = Observation()
        o._sample_interval = 0.1
        o._occupancy_maps = []
        obj = SimpleNamespace(
            tracked_object_type=SimpleNamespace(name="VEHICLE"), velocity=SimpleNamespace(x=2, y=0)
        )
        o._unique_objects = {t: obj for t in ("car", "lost", "gone")}
        for i in range(51 if after4 else 41):
            tokens = ["car"]
            polys = [translate(box(-2, -1, 2, 1), xoff=(i * 0.2 if i <= 40 else 1000))]
            if i <= 38:
                tokens.append("lost")
                polys.append(translate(box(-2, 9, 2, 11), xoff=i * 0.2))
            if i <= 30:
                tokens.append("gone")
                polys.append(box(-2, 19, 2, 21))
            o._occupancy_maps.append(PDMOccupancyMap(tokens, polys))
        o._global_to_local_idcs = list(range(len(o._occupancy_maps)))
        return o

    def test_prefix_exact_and_not_mutated(self):
        o = self.observation()
        a, b, m = extend_observation(o)
        self.assertEqual(len(o._occupancy_maps), 41)
        self.assertEqual(len(a._occupancy_maps), 51)
        for i in range(41):
            self.assertIs(a[i], o[i])
            self.assertIs(b[i], o[i])
        self.assertTrue(m["actor_motion_is_inferred"])
        self.assertAlmostEqual(a[50]["car"].centroid.x, 10)

    def test_recent_missing_actor_retained(self):
        a, _, _ = extend_observation(self.observation())
        self.assertIn("lost", a[50].tokens)
        self.assertNotIn("gone", a[50].tokens)

    def test_future_not_used(self):
        a, _, _ = extend_observation(self.observation(False))
        b, _, _ = extend_observation(self.observation(True))
        for i in range(41, 51):
            for token in a[i].tokens:
                self.assertEqual(a[i][token].wkb, b[i][token].wkb)

    def test_no_identity_is_error(self):
        o = self.observation()
        o._unique_objects.pop("car")
        with self.assertRaises(ValueError):
            extend_observation(o)

    def test_newborn_uncertainty_envelope_preserves_nominal_and_prefix(self):
        from shapely.geometry import box
        from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
            PDMOccupancyMap,
        )

        o = self.observation()
        o._unique_objects["new"] = SimpleNamespace(
            tracked_object_type=SimpleNamespace(name="VEHICLE"),
            velocity=SimpleNamespace(x=50000, y=10000),
        )
        before = o[40]
        o._occupancy_maps[40] = PDMOccupancyMap(
            list(before.tokens) + ["new"],
            [before[t] for t in before.tokens] + [box(20, 20, 24, 22)],
        )
        a, b, meta = extend_observation(o)
        self.assertIs(a[40], o[40])
        self.assertIs(b[40], o[40])
        self.assertTrue(a[50]["new"].equals(o[40]["new"]))
        self.assertGreater(b[50]["new"].area, a[50]["new"].area)
        self.assertTrue(
            next(x for x in meta["actors"] if x["token"] == "new")[
                "alternative_is_speed_bound_envelope"
            ]
        )

    def test_checkpoint_digest_roundtrip(self):
        import tempfile
        from pathlib import Path
        from drivesta_feedback.batch.store import write_npz

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "checkpoint.npz"
            s = "123abc"
            write_npz(p, config_digest=np.frombuffer(s.encode("ascii"), dtype=np.uint8))
            with np.load(p) as z:
                self.assertEqual(z["config_digest"].tobytes().decode("ascii"), s)


if __name__ == "__main__":
    unittest.main()
