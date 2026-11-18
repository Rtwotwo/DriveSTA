import unittest
import torch
from navsim.agents.drivesta.losses.proposal import DriveSTAProposalLoss, COMPONENTS
from functools import partial

MaskedDriveSTAProposalLoss = partial(DriveSTAProposalLoss, mask_unavailable_long_gt=True)


def example():
    torch.manual_seed(37)
    ar = torch.randn(3, 64, 8, 3, requires_grad=True)
    labels = torch.rand(3, 64, 7)
    targets = dict(
        trajectory=torch.randn(3, 8, 3),
        trajectory_long=torch.randn(3, 8, 3),
        auxiliary_trajectories=torch.randn(3, 8, 8, 3),
        auxiliary_mask=torch.ones(3, 8, dtype=torch.bool),
        native_labels=labels,
        replay_labels=labels,
        replay_mask=torch.ones(3, 64, dtype=torch.bool),
    )
    pred = dict(
        proposal_list=[ar],
        proposals=ar,
        pred_logit={k: torch.randn(3, 64, requires_grad=True) for k in COMPONENTS},
        replay_pred_logit={
            k: torch.randn(3, 64, requires_grad=True) for k in COMPONENTS
        },
    )
    return ar, targets, pred


class ScalingLossTest(unittest.TestCase):
    def test_real_loss_and_gradients_exact(self):
        ar, t, p = example()
        t["trajectory_long_valid"] = torch.ones(3, dtype=torch.bool)
        original = DriveSTAProposalLoss()
        scaled = MaskedDriveSTAProposalLoss()
        original.epoch_progress = scaled.epoch_progress = 3
        a = original(t, p)
        b = scaled(t, p)
        for k in a:
            torch.testing.assert_close(a[k], b[k], atol=0, rtol=0)
        ga = torch.autograd.grad(a["loss"], ar, retain_graph=True)[0]
        gb = torch.autograd.grad(b["loss"], ar)[0]
        torch.testing.assert_close(ga, gb, atol=0, rtol=0)

    def test_synthetic_long_placeholder_has_no_effect(self):
        ar, t, p = example()
        t["trajectory_long_valid"] = torch.tensor([True, False, False])
        loss = MaskedDriveSTAProposalLoss()
        loss.epoch_progress = 3
        a = loss(t, p)["loss"]
        ga = torch.autograd.grad(a, ar, retain_graph=True)[0]
        t["trajectory_long"][1:] += 100
        b = loss(t, p)["loss"]
        gb = torch.autograd.grad(b, ar)[0]
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-5)
        torch.testing.assert_close(ga, gb, atol=1e-6, rtol=1e-6)

    def test_same_v2_hyperparameters(self):
        a = DriveSTAProposalLoss().protocol()
        b = MaskedDriveSTAProposalLoss().protocol()
        b.pop("long_gt")
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
