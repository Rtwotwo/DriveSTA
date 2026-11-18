"""Compatibility for released checkpoints and immutable cache filenames.

Historical names are restricted to this adapter. Saved tensor bytes are never
modified; keys are translated in memory before strict loading or full resume.
"""

from collections import OrderedDict
from pathlib import Path

LOSS_NAMES = {
    "DriveTrajStage1Loss": "DriveSTAProposalLoss",
    "DriveTrajLoss": "DriveSTALoss",
    "FeedbackV2TrajectoryLoss": "DriveSTAProposalLoss",
    "FeedbackV2Ablation03TrajectoryLoss": "DriveSTAProposalLoss",
    "ScaledFeedbackV2Loss": "DriveSTAProposalLoss",
    "FeedbackStage2V4TrajectoryLoss": "DriveSTALoss",
    "Stage2V4Ablation00TrajectoryLoss": "DriveSTALoss",
}
OLD_MODEL_PREFIX = "_chainflow_vla_model"
NEW_MODEL_PREFIX = "_drivesta_model"
OLD_MODEL_PREFIXES = {OLD_MODEL_PREFIX, "_drivetraj_model"}


def remap_state_dict(state):
    def rename(key):
        parts = key.split(".")
        return ".".join(
            NEW_MODEL_PREFIX if part in OLD_MODEL_PREFIXES else part for part in parts
        )

    result = OrderedDict()
    for key, value in state.items():
        mapped = rename(key)
        if mapped in result:
            raise ValueError(f"Conflicting legacy and DriveSTA state keys: {mapped}")
        result[mapped] = value
    if hasattr(state, "_metadata"):
        result._metadata = {
            rename(key): value for key, value in state._metadata.items()
        }
    return result


def normalize_checkpoint(checkpoint):
    """Translate names without weakening strict tensor or supervision checks."""
    if "state_dict" in checkpoint:
        checkpoint["state_dict"] = remap_state_dict(checkpoint["state_dict"])
    metadata = checkpoint.get("feedback_training")
    if metadata is not None:
        from navsim.common.release_paths import relocate

        metadata = normalize_metadata(relocate(dict(metadata)))
        metadata["loss"] = LOSS_NAMES.get(metadata.get("loss"), metadata.get("loss"))
        checkpoint["feedback_training"] = metadata
    return checkpoint


def normalize_metadata(value):
    """Migrate serialized protocol keys without rewriting published weights."""
    names = {"stage2": "refinement", "stage2_phase": "refinement_phase",
             "stage2_dit_trained": "refinement_dit_trained"}
    if isinstance(value, dict):
        return {names.get(k, k): normalize_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_metadata(v) for v in value]
    if isinstance(value, str):
        return LOSS_NAMES.get(value, value)
    return value

def _cached_path(folder, names):
    folder = Path(folder)
    return next((folder / name for name in names if (folder / name).is_file()), folder / names[0])

def cached_feature_path(folder):
    return _cached_path(folder, ("drivesta_proposal_feature.gz", "drivetraj_stage1_feature.gz", "chainflow_vla_feature.gz"))

def cached_target_path(folder):
    return _cached_path(folder, ("drivesta_target.gz", "drivetraj_target.gz", "chainflow_vla_target.gz"))
