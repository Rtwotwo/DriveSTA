from drivesta_feedback.audit.simscale_directed_repair import complete_source_suffix


def test_restore_unique_source_connector():
    edges = {"a": ["c"], "c": ["b"], "b": []}
    assert complete_source_suffix(["a", "b"], edges.get, lambda x: x == "c") == ["a", "c", "b"]


def test_reject_ambiguous_or_unwitnessed_connector():
    edges = {"a": ["c", "d"], "c": ["b"], "d": ["b"], "b": []}
    assert complete_source_suffix(["a", "b"], edges.get, lambda x: True) is None
    edges["a"] = ["c"]
    assert complete_source_suffix(["a", "b"], edges.get, lambda x: False) is None


def test_ordered_adjacent_source_blocks_unchanged():
    edges = {"a": ["b"], "b": ["c"], "c": []}
    assert complete_source_suffix(["a", "b", "c"], edges.get, lambda x: False) == ["a", "b", "c"]
