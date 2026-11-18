"""Task partition preserves all candidate labels and original scene order."""

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from navsim.evaluate.drivesta.refinement import online


@pytest.mark.parametrize("chunk", [0, 1, 32, 64, 128])
@pytest.mark.parametrize("batch", [1, 6])
def test_partition_preserves_scene_and_candidate_order(monkeypatch, chunk, batch):
    count = 65  # exercise the final partial chunk
    trajectories = np.zeros((batch, count, 8, 3))
    trajectories[:, :, 0, 0] = np.arange(batch * count).reshape(batch, count)
    records = [dict(token=str(i), log_name="log" + str(i)) for i in range(batch)]
    seen = []

    def scorer(job):
        path, trajs = job
        seen.append((path, len(trajs)))
        ids = trajs[:, 0, 0]
        return np.repeat(ids[:, None], 7, axis=1)

    monkeypatch.setattr(online, "score_scene", scorer)
    api = online.OnlinePDM("/cache", records, workers=2, chunk_size=chunk)
    api.pool = ThreadPoolExecutor(2)
    try:
        result = api.score([r["token"] for r in records], trajectories)
    finally:
        api.close()
    np.testing.assert_array_equal(result[:, :, 0], trajectories[:, :, 0, 0])
    assert result.shape == (batch, count, 7)
    assert sum(n for _, n in seen) == batch * count
    for r in records:
        path = f"/cache/{r['log_name']}/unknown/{r['token']}/metric_cache.pkl"
        assert sum(n for p, n in seen if p == path) == count


def test_chunked_scoring_serial_mode_and_error_propagation(monkeypatch):
    api = online.OnlinePDM(
        "/cache", [dict(token="t", log_name="log")], workers=0, chunk_size=32
    )
    trajectories = np.zeros((1, 64, 8, 3))
    calls = []

    def scorer(job):
        calls.append(len(job[1]))
        if len(calls) == 2:
            raise RuntimeError("failed scoring chunk")
        return np.zeros((len(job[1]), 7))

    monkeypatch.setattr(online, "score_scene", scorer)
    with pytest.raises(RuntimeError, match="failed scoring chunk"):
        api.score(["t"], trajectories)
    with pytest.raises(ValueError, match="mismatch"):
        api.score([], trajectories)
    assert api.pool is None
