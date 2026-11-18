"""Train DriveSTA with a named data and supervision recipe."""
from importlib import import_module
import hydra

@hydra.main(config_path="config/training", config_name="drivesta_base", version_base=None)
def main(cfg):
    from navsim.common.compatibility.indexes import index_path
    for key in ("index_path", "history_index", "future_index", "synthetic_video_index"):
        if key in cfg.feedback and cfg.feedback[key]:
            cfg.feedback[key] = str(index_path(cfg.feedback[key]))
    mode, profile = str(cfg.training_mode), str(cfg.data_profile)
    if mode not in ("proposal", "refinement") or profile not in ("real", "mixed"):
        raise ValueError("Use training_mode=proposal|refinement and data_profile=real|mixed")
    if cfg.agent._target_ != "navsim.agents.drivesta.drivesta_agent.DriveSTAAgent":
        raise ValueError("DriveSTA recipes require DriveSTAAgent")
    if bool(cfg.agent.config.use_diffusion_decoder) != (mode == "refinement"):
        raise ValueError("Training mode does not match the model refinement configuration")
    if bool(cfg.agent.loss.mask_unavailable_long_gt) != (profile == "mixed"):
        raise ValueError("Mixed data requires missing long-GT supervision masking")
    if float(cfg.agent.loss.rank_weight) < 0:
        raise ValueError("rank_weight must be nonnegative")
    import_module(f"navsim.planning.training.drivesta.runners.{mode}_{profile}").run(cfg)

if __name__ == "__main__":
    main()
