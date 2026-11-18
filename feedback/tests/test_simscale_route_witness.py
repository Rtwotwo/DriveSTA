from drivesta_feedback.audit.simscale_route_witness import ordered_membership, witnessed_choice


def fixture(groups, raw=("r", "z")):
    return dict(
        raw_roadblocks=list(raw),
        gt_directed_membership=[[dict(roadblock_id=x) for x in g] for g in groups],
    )


def test_rejoin_after_gt_window_with_unique_directed_witness():
    a = fixture([["a"], ["a"], ["b"], ["b"]])
    edges = {"a": ["b"], "b": ["r"], "r": ["z"], "z": []}
    route, reason, _ = witnessed_choice(a, edges.get, lambda x: False)
    assert route == ["a", "b", "r", "z"]


def test_ambiguous_anchor_resolved_only_by_ordered_gt():
    a = fixture([["a", "d"], ["b"], ["r"]])
    edges = {"a": ["b"], "d": ["r"], "b": ["r"], "r": ["z"], "z": []}
    assert witnessed_choice(a, edges.get, lambda x: False)[0] == ["a", "b", "r", "z"]


def test_two_supported_routes_are_rejected():
    a = fixture([["a"], ["b", "c"], ["r"]])
    edges = {"a": ["b", "c"], "b": ["r"], "c": ["r"], "r": ["z"], "z": []}
    assert witnessed_choice(a, edges.get, lambda x: False)[1] == "multiple_gt_supported_routes"


def test_no_unwitnessed_prefix_or_backwards_or_missing_waypoints():
    edges = {"a": ["b"], "b": ["r"], "r": ["z"], "z": []}
    assert witnessed_choice(fixture([["a"], ["a"]]), edges.get, lambda x: False)[0] is None
    assert witnessed_choice(fixture([["a"], [], ["r"]]), edges.get, lambda x: False)[0] is None
    assert not ordered_membership(["a", "b", "r"], [{"a"}, {"r"}, {"b"}])


def test_disconnected_suffix_is_not_truncated():
    edges = {"a": ["r"], "r": [], "z": []}
    assert witnessed_choice(fixture([["a"], ["r"]]), edges.get, lambda x: False)[0] is None


def test_added_suffix_connector_does_not_count_as_initial_prefix():
    edges = {"r": ["c"], "c": ["z"], "z": []}
    assert witnessed_choice(fixture([["r"], ["z"]]), edges.get, lambda x: True)[0] is None


def test_cycles_are_rejected_and_depth_is_bounded():
    edges = {"a": ["b"], "b": ["a", "r"], "r": ["z"], "z": []}
    a = fixture([["a"], ["b"], ["r"]])
    assert witnessed_choice(a, edges.get, lambda x: False, max_edges=1)[0] is None
    assert witnessed_choice(a, edges.get, lambda x: False)[0] == ["a", "b", "r", "z"]
