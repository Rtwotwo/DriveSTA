"""Separate V4 data-scaling launch; the original agent and trainer are reused."""

import json, math, os
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytorch_lightning as pl
from navsim.planning.training.drivesta.data.refinement.mixed_dataset import (
    DriveSTAMixedDataset,
    DriveSTAMixedOnlinePDM,
    digest,
)
from navsim.planning.training.drivesta.refinement.module import DriveSTALightningModule
from navsim.planning.training.drivesta.data.refinement.datamodule import DriveSTADataModule
from navsim.planning.training.collate import drivesta_collate_fn_for_agent


def run(cfg):
    if str(cfg.trainer.params.precision) not in ("bf16-mixed", "32-true", "32"):
        raise ValueError("V4 EMA requires BF16/FP32")
    if bool(cfg.train_ckpt_path) == bool(cfg.agent.checkpoint_path):
        raise ValueError("Choose exactly one Proposal init or full V4-scaled resume")
    if cfg.agent.loss.phase != "joint" or cfg.agent.config.training_phase != "joint":
        raise ValueError("Full Refinement joint required")
    if cfg.agent.loss._target_ != "navsim.agents.drivesta.losses.refinement.DriveSTALoss":
        raise ValueError("Missing synthetic long-GT mask")
    if (
        cfg.feedback.training_split != "trainval"
        or cfg.validation_run
        or cfg.trainer.params.limit_val_batches
    ):
        raise ValueError(
            "This full mixture uses external NAVTEST, no overlapping validation"
        )
    if (
        cfg.feedback.replay_samples,
        cfg.feedback.auxiliary_samples,
        cfg.feedback.auxiliary_bank_capacity,
    ) != (64, 32, 32):
        raise ValueError(
            "Scaled V4 requires replay64/dense32/bank32; no silent overrides"
        )
    if (
        cfg.feedback.limit_train_scenes is not None
        or cfg.feedback.limit_val_scenes is not None
    ):
        raise ValueError(
            "This entry uses the complete fixed mixture; scene limits are unsupported"
        )
    pl.seed_everything(cfg.seed, workers=True)
    train = DriveSTAMixedDataset(
        cfg.feedback.index_path,
        cfg.agent.config,
        cfg.feedback.history_index,
        cfg.feedback.future_index,
        cfg.feedback.synthetic_video_index,
        cfg.seed,
    )
    allowed = set(cfg.train_logs) | set(cfg.val_logs)
    if any(r["log_name"] not in allowed for r in train.real.records):
        raise ValueError("Real test log leakage")
    if any(
        r["log_name"].rsplit("-", 2)[0] not in allowed for r in train.synthetic.records
    ):
        raise ValueError("Synthetic source log outside trainval")
    if len(train) != cfg.feedback.expected_trainval_scenes:
        raise ValueError("DATA_SIZE mismatch")
    cfg.agent.scheduler_args.dataset_size = len(train)
    world = int(cfg.trainer.params.devices)
    batch = int(cfg.dataloader.params.batch_size)
    acc = int(cfg.trainer.params.accumulate_grad_batches)
    steps_epoch = math.ceil(math.ceil(math.ceil(len(train) / world) / batch) / acc)
    summary = dict(
        data_size=len(train),
        real=len(train.real),
        synthetic=len(train.synthetic),
        real_video_valid=train.real.future_valid_scenes,
        synthetic_video_valid=train.synthetic.future_valid_scenes,
        optimizer_steps=steps_epoch * int(cfg.agent.scheduler_args.num_epochs),
        steps_per_epoch=steps_epoch,
        batch=batch,
        world=world,
        accumulation=acc,
        planner_peak_lr=float(cfg.agent.lr_args.base_lr)
        * math.sqrt(batch * world * acc / cfg.agent.lr_args.base_batch_size),
        jepa_encoder_peak_lr=cfg.agent.config.jepa_encoder_lr,
        jepa_predictor_lr=cfg.agent.config.jepa_predictor_lr,
        initialization=cfg.agent.checkpoint_path,
        resume=cfg.train_ckpt_path,
    )
    if cfg.preflight_only:
        print(json.dumps(dict(status="PREFLIGHT_ONLY", **summary)))
        return
    if cfg.train_ckpt_path:
        cfg.agent.config.jepa_load_pretrained = False
    agent = instantiate(cfg.agent)
    agent.initialize()
    if cfg.agent.checkpoint_path:
        meta = agent.feedback_checkpoint_training
        if (
            meta.get("loss") != "DriveSTAProposalLoss"
            or meta.get("dataset_size") != 232672
        ):
            raise ValueError(
                "This new experiment requires the V2-scaled Proposal checkpoint; use resume for scaled V4 continuation"
            )
    params = OmegaConf.to_container(cfg.dataloader.params, resolve=True)
    if not params.get("num_workers"):
        params.pop("prefetch_factor", None)
    data = DriveSTADataModule(
        train, [], params, drivesta_collate_fn_for_agent(agent), seed=cfg.seed
    )
    protocol = dict(
        index_identity=train.index_identity,
        seed=cfg.seed,
        auxiliary_bank_capacity=32,
        auxiliary_samples=32,
        history_identity=train.history_identity,
        future_identity=train.future_identity,
        synthetic_future_identity=train.synthetic_future_identity,
        sampling="epoch_index_pairs_consumed_cursor_v1",
        sources=train.spec,
        long_gt="real_only_batch_mean_mask",
        video_sources="NAVSIM14_and_SimScale12_recorded_frames_to_official8frame_masked_loss",
    )
    module = DriveSTALightningModule(
        agent,
        len(train),
        None,
        False,
        "trainval",
        replay_samples=64,
        online=DriveSTAMixedOnlinePDM(
            train.official_cache_root,
            train.records,
            cfg.feedback.online_workers,
            cfg.feedback.online_chunk_size,
        ),
        target_policy=train.target_policy,
        data_protocol=protocol,
    )
    if (
        os.environ.get("LOCAL_RANK", "0") == "0"
        and not cfg.train_ckpt_path
        and (Path(cfg.output_dir) / "initialization.json").exists()
    ):
        raise FileExistsError("Choose a fresh output directory or explicit full resume")
    trainer_params = OmegaConf.to_container(cfg.trainer.params, resolve=True)
    if isinstance(trainer_params.get("logger"), dict):
        trainer_params["logger"] = instantiate(trainer_params["logger"])
    trainer = pl.Trainer(**trainer_params, callbacks=agent.get_training_callbacks())
    if trainer.is_global_zero:
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if not cfg.train_ckpt_path:
            summary.update(
                kind="scaled_Proposal_to_new_V4_Refinement",
                checkpoint_sha256=digest(cfg.agent.checkpoint_path),
            )
            (out / "initialization.json").write_text(json.dumps(summary, indent=2))
        OmegaConf.save(cfg, out / "training_config.yaml", resolve=True)
        print(json.dumps(summary), flush=True)
    trainer.fit(module, datamodule=data, ckpt_path=cfg.train_ckpt_path)
