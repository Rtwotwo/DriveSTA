from contextlib import nullcontext
from time import sleep

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from typing import Dict, Tuple, Any, List
from navsim.common.dataclasses import Trajectory
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Trajectory


def _rowwise_isin(tensor_1: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    matches = tensor_1[:, None] == target_tensor

    return torch.sum(matches, dim=1, dtype=torch.bool)


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(
        self,
        agent: AbstractAgent,
        for_viz=False,
        trajectory_optimizer: Any = None,
    ):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self.checkpoint_file = None
        self.for_viz = for_viz
        self.trajectory_optimizer = trajectory_optimizer

    def _step_autocast_context(self, logging_prefix: str):
        """Disable the CUDA autocast weight cache for Direct-DGPO training.

        V4 first runs the current DiT under ``no_grad`` for its rollout and then
        reuses the same weights in several non-reentrant checkpointed forwards.
        With PyTorch 2.5 the process-wide autocast weight cache can make the
        checkpoint recomputation observe a different tensor sequence.  Keeping
        autocast enabled but disabling only its weight cache preserves mixed
        precision and activation checkpointing while making recomputation exact.
        """
        config = getattr(self.agent, "_config", None)
        stage3_dgpo = bool(
            config is not None
            and getattr(config, "stage3_dgpo_enabled", False)
            and logging_prefix == "train"
        )
        if not stage3_dgpo:
            return nullcontext()

        try:
            autocast_enabled = torch.is_autocast_enabled("cuda")
        except TypeError:
            autocast_enabled = torch.is_autocast_enabled()
        if not autocast_enabled:
            return nullcontext()

        try:
            autocast_dtype = torch.get_autocast_dtype("cuda")
        except AttributeError:
            autocast_dtype = torch.get_autocast_gpu_dtype()
        return torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=True,
            cache_enabled=False,
        )

    def _step(
        self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str
    ) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch

        # Keep rollout and the M re-noised Direct-DGPO forwards in one autocast
        # context so non-reentrant checkpoint recomputation uses identical AMP
        # settings.  This context is a no-op for every non-V4 training path.
        with self._step_autocast_context(logging_prefix):
            prediction = self.agent.forward(features, targets)
            loss_dict = self.agent.compute_loss(features, targets, prediction)

        if type(loss_dict) is dict:
            config = getattr(self.agent, "_config", None)
            scorer_cache_epoch_logging = bool(
                config is not None
                and getattr(config, "stage3_scorer_calibration", False)
                and getattr(config, "scorer_cache_only", False)
            )
            scorer_online_validation = bool(
                config is not None
                and getattr(config, "stage3_scorer_calibration", False)
                and getattr(config, "stage3_scorer_online_validation", False)
            )
            for key, value in loss_dict.items():
                # Scorer-only Stage3 uses the same deterministic online
                # candidate/PDM objective for training and held-out validation.
                # Its held-out metrics must be reduced over the complete
                # recording-disjoint selection split.
                # Logging ~40 diagnostics on_step with sync_dist=True caused
                # ~40 extra 8-GPU collectives per tiny adapter update.  Keep
                # legacy cache-only scorer diagnostics as epoch aggregates.
                scorer_core_loss = scorer_cache_epoch_logging and key == "loss"
                if (
                    scorer_online_validation
                    or bool(getattr(config, "scorer_distillation_enabled", False))
                    or bool(getattr(config, "future_teacher_validation", False))
                ):
                    log_on_step = logging_prefix == "train"
                    log_on_epoch = logging_prefix == "val"
                else:
                    log_on_step = not scorer_cache_epoch_logging or scorer_core_loss
                    log_on_epoch = scorer_cache_epoch_logging
                self.log(
                    f"{logging_prefix}/" + key,
                    value,
                    # Retain one step-level loss curve for observability while
                    # moving the remaining diagnostics off the hot path.
                    on_step=log_on_step,
                    on_epoch=log_on_epoch,
                    prog_bar=True,
                    sync_dist=True,
                )
            return loss_dict["loss"]
        else:
            return loss_dict

    def training_step(
        self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int
    ) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        config = getattr(self.agent, "_config", None)
        scorer_objective_validation = bool(
            config is not None
            and getattr(config, "stage3_scorer_calibration", False)
            and (
                getattr(config, "scorer_cache_only", False)
                or getattr(config, "stage3_scorer_online_validation", False)
            )
        )
        if (
            scorer_objective_validation
            or bool(getattr(config, "scorer_distillation_enabled", False))
            or bool(getattr(config, "future_teacher_validation", False))
        ):
            # Scorer Stage3 uses its own true-component/listwise objective for
            # held-out selection metrics. Online mode recomputes the frozen
            # Epoch06 candidate pool and true PDM labels on the disjoint split.
            return self._step(batch, "val")
        if "chainflow_vla" in self.agent.name() or "ChainFlowVLA" in self.agent.name():
            features, targets = batch
            # score,best_score=self.agent.inference(features, targets)
            predictions = self.agent.forward(features, targets)
            all_chosen_trajectories = predictions["trajectory"][:, None]
            all_proposed_trajectories = predictions["proposals"]
            final_score, fake_best_score, proposal_scores, l2, trajectoy_scores = (
                self.agent.compute_score(targets, all_chosen_trajectories)
            )
            _, best_score, all_proposal_scores, _, _ = self.agent.compute_score(
                targets, all_proposed_trajectories
            )
            mean_score = proposal_scores.mean()

            logging_prefix = "val"
            if "pdm_score" in predictions:
                pdm_score = predictions["pdm_score"]
                best_pred_score_values = pdm_score[
                    torch.arange(len(pdm_score)), torch.argmax(pdm_score, dim=1)
                ]
                score_error = torch.abs(best_pred_score_values - proposal_scores).mean()
                self.log(
                    f"{logging_prefix}/score_error",
                    score_error,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )

                best_pred_score_index = torch.argmax(pdm_score, dim=1)
                best_real_score_index = torch.argmax(all_proposal_scores, dim=1)
                score_hit_rate = torch.mean(
                    best_pred_score_index == best_real_score_index, dtype=torch.float32
                )

                best_possible_scores = all_proposal_scores[
                    torch.arange(len(all_proposal_scores)), best_real_score_index
                ]
                best_actual_scores = all_proposal_scores[
                    torch.arange(len(all_proposal_scores)), best_pred_score_index
                ]
                lost_score = torch.mean(best_possible_scores - best_actual_scores)
                self.log(
                    f"{logging_prefix}/score_hit_rate",
                    score_hit_rate,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
                self.log(
                    f"{logging_prefix}/lost_score",
                    lost_score,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )

                top_5_indices_real = torch.topk(all_proposal_scores, k=5, dim=1).indices
                top_5_score_hit_rate = _rowwise_isin(
                    best_pred_score_index, top_5_indices_real
                ).mean(dtype=torch.float32)
                self.log(
                    f"{logging_prefix}/top_5_score_hit_rate",
                    top_5_score_hit_rate,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )

            self.log(
                f"{logging_prefix}/score",
                final_score,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                f"{logging_prefix}/best_score",
                best_score,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                f"{logging_prefix}/mean_score",
                mean_score,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                f"{logging_prefix}/l2",
                l2,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            collision = trajectoy_scores[:, 0].mean()
            self.log(
                f"{logging_prefix}/collision",
                collision,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            drivable_area_compliance = trajectoy_scores[:, 1].mean()
            self.log(
                f"{logging_prefix}/dac",
                drivable_area_compliance,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            ego_progress = trajectoy_scores[:, 2].mean()
            self.log(
                f"{logging_prefix}/progress",
                ego_progress,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            time_to_collision_within_bound = trajectoy_scores[:, 3].mean()
            self.log(
                f"{logging_prefix}/ttc",
                time_to_collision_within_bound,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            comfort = trajectoy_scores[:, 4].mean()
            self.log(
                f"{logging_prefix}/comfort",
                comfort,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

            return final_score
        else:
            return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        configure_schedule = getattr(self.agent, "configure_optimizer_schedule", None)
        if callable(configure_schedule):
            configure_schedule(self.trainer)
        return self.agent.get_optimizers()

    def predict_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Used during the multi-gpu proccessing to parallelize the prediction of trajectories.
        NOTE: requires append_token_to_batch=True in the dataset used to instantiate the trainer.
        """
        return self.predict_step_drivesta(batch, batch_idx)

    def predict_step_drivesta(
        self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor], List[str]], batch_idx: int
    ):
        features, targets, tokens = batch
        self.agent.eval()
        with torch.no_grad():
            optimizer = self.trajectory_optimizer
            if optimizer is not None and optimizer.enabled:
                with optimizer.capture_scorer(self.agent):
                    predictions = self.agent.forward(features, targets)
                predictions = optimizer.refine_predictions(
                    predictions,
                    features,
                    sample_ids=tokens,
                )
            else:
                predictions = self.agent.forward(features, targets)
            poses = predictions["trajectory"]
            if self.for_viz:
                all_proposed_trajectories = predictions["proposal_list"]
                final_trajectories = predictions["proposals"]
                _, _, final_scores, _, _ = self.agent.compute_score(targets, final_trajectories)
                ego_status = features["ego_status"]
        result = {}
        for index, (pose, token) in enumerate(zip(poses.cpu().numpy(), tokens)):
            proposal = Trajectory(pose)
            if self.for_viz:
                proposal_list = [
                    proposal_list[index].cpu().numpy()
                    for proposal_list in all_proposed_trajectories
                ]
                result[token] = {
                    "trajectory": proposal,
                    "all_proposals": proposal_list,
                    "all_proposal_scores": final_scores[index],
                    "high_level_command": ego_status[index],
                }
            else:
                result[token] = {"trajectory": proposal}
        return result
