import unittest
from drivesta_feedback.audit.simscale_route_graph import shortest_connection
from drivesta_feedback.audit.simscale_directed_repair import route_choice, bounded_connections


class DirectedRouteTests(unittest.TestCase):
    def audit(self, anchors=("A",), raw=("T", "U")):
        return dict(
            initial=dict(valid=False),
            current_directed_blocks=list(anchors),
            raw_roadblocks=list(raw),
            gt_directed_membership=[[], [dict(roadblock_id="T")]],
        )

    def test_unique_source_supported_prefix(self):
        graph = {"A": ["B"], "B": ["T"], "T": ["U"], "U": []}
        route, _ = route_choice(self.audit(), lambda k: graph[k], lambda k: True)
        self.assertEqual(route, ["A", "B", "T", "U"])

    def test_ambiguity_is_not_resolved_using_scores(self):
        graph = {"A": ["B", "C"], "B": ["T"], "C": ["T"], "T": ["U"], "U": []}
        self.assertEqual(len(shortest_connection("A", "T", lambda k: graph[k])), 2)
        self.assertIsNone(route_choice(self.audit(), lambda k: graph[k], lambda k: True)[0])

    def test_no_wrong_direction_or_multiple_anchor(self):
        for anchors in ((), ("A", "B")):
            self.assertIsNone(route_choice(self.audit(anchors), lambda k: ["T"], lambda k: True)[0])

    def test_distant_loop_and_unsupported_prefix_rejected(self):
        graph = {"A": ["B"], "B": ["T"], "T": ["U"], "U": []}
        self.assertEqual(shortest_connection("A", "T", lambda k: graph[k], 1), [])
        self.assertIsNone(route_choice(self.audit(), lambda k: graph[k], lambda k: k != "B")[0])

    def test_disconnected_suffix_rejected(self):
        graph = {"A": ["T"], "T": [], "U": []}
        self.assertIsNone(route_choice(self.audit(), lambda k: graph[k], lambda k: True)[0])

    def test_existing_valid_route_untouched(self):
        z = self.audit()
        z["initial"]["valid"] = True
        self.assertIsNone(route_choice(z, lambda k: ["T"], lambda k: True)[0])

    def test_bfs_cycle_terminates(self):
        graph = {"A": ["B"], "B": ["A"]}
        self.assertEqual(shortest_connection("A", "T", lambda k: graph[k]), [])

    def test_longer_alternative_is_also_ambiguous(self):
        graph = {"A": ["T", "B"], "B": ["T"], "T": ["U"], "U": []}
        self.assertEqual(len(bounded_connections("A", "T", lambda k: graph[k], 6)), 2)
        self.assertIsNone(route_choice(self.audit(), lambda k: graph[k], lambda k: True)[0])

    def test_optimizer_keeps_gt_labels_but_rejects_gt_targets(self):
        import numpy as np
        from drivesta_feedback.batch.simscale_constrained_seeds import exclude_expert_probes

        expert = np.zeros((8, 3), np.float32)
        raw = expert + 1
        new = expert + 2
        labels = np.ones((3, 7))
        arrays = dict(
            trajectories=np.array([expert, raw, new]), labels=labels, eligible=np.ones(3, bool)
        )
        out = exclude_expert_probes(arrays, expert, raw)
        np.testing.assert_array_equal(out["eligible"], [False, False, True])
        np.testing.assert_array_equal(out["labels"], labels)
        np.testing.assert_array_equal(out["trajectories"], [expert, raw, new])

    def test_completed_probe_archive_resumes_without_scoring(self):
        import numpy as np
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from drivesta_feedback.batch.simscale_constrained_seeds import optimize_seeds

        expert = np.zeros((8, 3), np.float32)
        raw = expert + 1
        new = expert + 2
        arrays = dict(
            trajectories=np.array([expert, raw, new]),
            labels=np.ones((3, 7)),
            alternative_labels=np.ones((3, 7)),
            tracking=np.zeros((3, 3)),
            eligible=np.ones(3, bool),
        )
        space = SimpleNamespace(cache=None, road=None, route=None, vehicle=None, sampling=None)
        policy = SimpleNamespace(
            max_tracking_ade_m=1, max_tracking_error_m=2, max_tracking_heading_error_rad=0.35
        )

        class NoScoring:
            def score(self, *args):
                raise AssertionError("Completed probes must not be rescored")

        with tempfile.TemporaryDirectory() as directory:
            np.savez(
                Path(directory) / "optimized_seeds.npz",
                **arrays,
                slacks=np.ones((3, 2)),
                objectives=np.ones(3),
                finished=np.array(True),
                digest=np.frombuffer(b"test", np.uint8),
            )
            result = optimize_seeds(
                directory,
                dict(constrained_seeds={}, digest="test"),
                NoScoring(),
                space,
                expert,
                policy,
                None,
                None,
                None,
                None,
                None,
                raw,
            )
        np.testing.assert_array_equal(result["labels"], arrays["labels"])
        np.testing.assert_array_equal(result["eligible"], [False, False, True])


if __name__ == "__main__":
    unittest.main()
