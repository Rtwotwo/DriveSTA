"""Checkpoint migration and configurable supervision for one DriveSTA family."""

from collections import OrderedDict

import pytest
import torch

from navsim.common.compatibility.checkpoints import (
    normalize_checkpoint,
    remap_state_dict,
)
from navsim.agents.drivesta.losses.refinement import DriveSTALoss
from navsim.agents.drivesta.losses.proposal import DriveSTAProposalLoss


@pytest.mark.parametrize(
    "old,current",
    [
        ("FeedbackV2Ablation03TrajectoryLoss", "DriveSTAProposalLoss"),
        ("ScaledFeedbackV2Loss", "DriveSTAProposalLoss"),
        ("Stage2V4Ablation00TrajectoryLoss", "DriveSTALoss"),
        ("FeedbackStage2V4TrajectoryLoss", "DriveSTALoss"),
    ],
)
def test_checkpoint_translation_preserves_tensor_identity(old, current):
    weight = torch.arange(6).reshape(2, 3)
    checkpoint = {
        "state_dict": OrderedDict([("agent._chainflow_vla_model.weight", weight)]),
        "feedback_training": {"loss": old, "dataset_size": 103263},
    }
    normalize_checkpoint(checkpoint)
    assert checkpoint["state_dict"]["agent._drivesta_model.weight"] is weight
    assert checkpoint["feedback_training"]["loss"] == current
    assert checkpoint["feedback_training"]["dataset_size"] == 103263
    normalize_checkpoint(checkpoint)
    assert checkpoint["state_dict"]["agent._drivesta_model.weight"] is weight


def test_mixed_state_names_fail_instead_of_overwriting():
    with pytest.raises(ValueError, match="Conflicting"):
        remap_state_dict(
            {
                "_chainflow_vla_model.weight": torch.ones(1),
                "_drivesta_model.weight": torch.zeros(1),
            }
        )


@pytest.mark.parametrize("loss", [DriveSTALoss, DriveSTAProposalLoss])
def test_rank_and_mask_are_explicit_configuration(loss):
    off = loss(rank_weight=0)
    on = loss(rank_weight=0.05, mask_unavailable_long_gt=True)
    assert off.protocol()["rank_weight"] == 0
    assert on.protocol()["rank_weight"] == 0.05
    assert "long_gt" not in off.protocol()
    assert on.protocol()["long_gt"] == "real_only_batch_mean_mask"
    with pytest.raises(ValueError, match="nonnegative"):
        loss(rank_weight=-0.05)
