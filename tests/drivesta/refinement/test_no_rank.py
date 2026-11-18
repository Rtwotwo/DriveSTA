from unittest.mock import patch

import torch
from torch.nn import functional as F
from navsim.agents.drivesta.losses import refinement as loss


def test_no_rank_matches_official_calibration_gradients_without_computing_pairs():
    torch.manual_seed(2)
    logits = torch.randn(2, 64, 6, requires_grad=True)
    labels = torch.rand(2, 64, 7)
    mask = torch.ones(2, 64, dtype=torch.bool)
    mask[1, 32:] = False
    heads = dict(zip(loss.COMPONENTS, logits.unbind(-1)))
    with patch.object(
        loss, "frontier_rank_loss", side_effect=AssertionError("rank was called")
    ):
        actual, metrics = loss.score_loss(heads, labels, mask, rank_weight=0.0)
    pdm = loss.compose_pdm(logits.sigmoid())
    expected = F.binary_cross_entropy_with_logits(
        logits, labels[..., :6], reduction="none"
    ).sum(-1)
    expected = (
        ((expected + 0.5 * (pdm - labels[..., 6]).square()) * mask).sum(1) / mask.sum(1)
    ).mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits)[0], torch.autograd.grad(expected, logits)[0]
    )
    assert metrics["rank"] == metrics["rank_pairs"] == 0
    positive, stats = loss.score_loss(heads, labels, mask, rank_weight=0.05)
    assert stats["rank_pairs"] > 0 and positive > actual
