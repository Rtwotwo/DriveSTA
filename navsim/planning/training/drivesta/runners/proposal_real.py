"""Train the independent expert-anchored DriveSTA agent."""

import json
import logging
from pathlib import Path


from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytorch_lightning as pl

from navsim.planning.training.drivesta.data.proposal.dataset import DriveSTAProposalDataset
from navsim.planning.training.drivesta.proposal.module import (
    DriveSTAProposalLightningModule,
)
from navsim.evaluate.drivesta.proposal.validation import OfficialPDM
from navsim.evaluate.drivesta.proposal.online import OnlinePDM
from navsim.planning.training.drivesta.data.proposal.datamodule import DriveSTAProposalDataModule
from navsim.planning.training.collate import drivesta_collate_fn_for_agent

log = logging.getLogger(__name__)


def run(cfg):
    pl.seed_everything(cfg.seed, workers=True)
    if cfg.train_ckpt_path and cfg.agent.checkpoint_path:
        raise ValueError(
            "Set either train_ckpt_path (resume) or agent.checkpoint_path (initialization)"
        )
    if cfg.agent.config.use_diffusion_decoder and not (
        cfg.train_ckpt_path or cfg.agent.checkpoint_path
    ):
        raise ValueError(
            "Refinement requires a proposal/refinement initialization checkpoint, or a feedback resume checkpoint"
        )
    kwargs = dict(
        index_path=cfg.feedback.index_path,
        use_vlm=cfg.agent.config.use_vlm_feature_cache,
        replay_samples=cfg.feedback.replay_samples,
        seed=cfg.seed,
        auxiliary_samples=cfg.feedback.auxiliary_samples,
        auxiliary_bank_capacity=cfg.feedback.auxiliary_bank_capacity,
    )
    split = cfg.feedback.training_split
    if split not in ("train", "trainval"):
        raise ValueError("feedback.training_split must be train or trainval")
    if split == "trainval" and (
        cfg.validation_run or cfg.trainer.params.limit_val_batches
    ):
        raise ValueError(
            "Full trainval optimization uses navtest for evaluation; val scenes overlap training"
        )
    train = DriveSTAProposalDataset(
        **kwargs, split=split, limit=cfg.feedback.limit_train_scenes
    )
    val = DriveSTAProposalDataset(
        **kwargs, split="val", limit=cfg.feedback.limit_val_scenes
    )
    if not len(train) or (
        (cfg.validation_run or cfg.trainer.params.limit_val_batches) and not len(val)
    ):
        raise ValueError("Empty training or validation set")
    expected = (
        cfg.feedback.expected_trainval_scenes
        if split == "trainval"
        else cfg.feedback.expected_train_scenes
    )
    if cfg.feedback.limit_train_scenes is None and len(train) != expected:
        raise ValueError(f"Expected {expected} training scenes, found {len(train)}")
    if len({r["token"] for r in train.records}) != len(train):
        raise ValueError("Duplicate training token")
    # Fail on an edited index that moves official held-out logs into training.
    official_train, official_val = set(cfg.train_logs), set(cfg.val_logs)
    allowed = official_train if split == "train" else official_train | official_val
    if any(r["log_name"] not in allowed for r in train.records):
        raise ValueError(
            "Feedback train index includes logs outside the official train split"
        )
    if any(r["log_name"] not in official_val for r in val.records):
        raise ValueError(
            "Feedback val index includes logs outside the official val split"
        )
    cfg.agent.scheduler_args.dataset_size = len(train)
    if cfg.preflight_only:
        print(
            json.dumps(
                dict(
                    preflight=True,
                    scenes=len(train),
                    index_identity=train.index_identity,
                )
            )
        )
        return
    agent = instantiate(cfg.agent)
    agent.initialize()
    source = getattr(agent, "feedback_checkpoint_training", {})
    if (cfg.validation_run or cfg.trainer.params.limit_val_batches) and source.get(
        "training_split"
    ) == "trainval":
        raise ValueError(
            "This checkpoint was trained on trainval; use TRAIN_TEST_SPLIT=navtest"
        )
    loader_params = OmegaConf.to_container(cfg.dataloader.params, resolve=True)
    if int(loader_params.get("num_workers", 0)) == 0:
        loader_params.pop("prefetch_factor", None)
    collate = drivesta_collate_fn_for_agent(agent)
    data_module = DriveSTAProposalDataModule(
        train, val, loader_params, collate, seed=cfg.seed
    )
    if cfg.train_ckpt_path:
        import torch

        resume = torch.load(cfg.train_ckpt_path, map_location="cpu", weights_only=False)
        loader_states = resume["loops"]["fit_loop"]["state_dict"].get(
            "combined_loader", []
        )
        if len(loader_states) != 1:
            raise ValueError(
                "Full resume requires exactly one saved training-loader cursor"
            )
        data_module.resume_loader_state = dict(loader_states[0])
        log.info(
            "Preloading consumed data cursor: epoch=%s samples/rank=%s",
            loader_states[0]["epoch"],
            loader_states[0]["samples"],
        )
        resume_step = int(resume["global_step"])
        del resume

    evaluator = None
    if cfg.validation_run or cfg.trainer.params.limit_val_batches:
        evaluator = OfficialPDM(cfg.feedback.official_cache_root, val.records)
    module = DriveSTAProposalLightningModule(
        agent,
        len(train),
        evaluator,
        cfg.feedback.validation_oracle,
        split,
        replay_samples=cfg.feedback.replay_samples,
        online=OnlinePDM(
            cfg.feedback.official_cache_root, train.records, cfg.feedback.online_workers
        ),
        target_policy=json.loads(Path(cfg.feedback.index_path).read_text())[
            "target_policy"
        ],
        data_protocol=dict(
            index_identity=train.index_identity,
            seed=cfg.seed,
            auxiliary_bank_capacity=cfg.feedback.auxiliary_bank_capacity,
            auxiliary_samples=cfg.feedback.auxiliary_samples,
            sampling="epoch_index_pairs_consumed_cursor_v1",
        ),
    )
    # Explicit logger version keeps resumed checkpoints in the original folder.
    import os

    trainer_params = dict(cfg.trainer.params)
    logger_version = os.environ.get("TRAIN_LOGGER_VERSION")
    if logger_version is not None:
        from pytorch_lightning.loggers import TensorBoardLogger

        if not cfg.train_ckpt_path:
            raise ValueError(
                "Explicit resume logger version requires a full checkpoint resume"
            )
        if not logger_version.isdigit():
            raise ValueError("TRAIN_LOGGER_VERSION must be a nonnegative integer")
        logger_kwargs = {}
        purge_step = os.environ.get("TRAIN_LOGGER_PURGE_STEP", str(resume_step))
        if purge_step is not None:
            if not purge_step.isdigit():
                raise ValueError("TRAIN_LOGGER_PURGE_STEP must be nonnegative")
            logger_kwargs["purge_step"] = int(purge_step)
        trainer_params["logger"] = TensorBoardLogger(
            save_dir=str(cfg.output_dir),
            name="lightning_logs",
            version=int(logger_version),
            **logger_kwargs,
        )
    trainer = pl.Trainer(**trainer_params, callbacks=agent.get_training_callbacks())
    if trainer.is_global_zero:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, Path(cfg.output_dir) / "training_config.yaml", resolve=True)
        log.info(
            "DATA_SIZE=%d, val=%d, batch/GPU=%d; LR uses actual world size, accumulation and optimizer steps",
            len(train),
            len(val),
            cfg.dataloader.params.batch_size,
        )
    if cfg.validation_run:
        trainer.validate(module, datamodule=data_module, ckpt_path=cfg.train_ckpt_path)
    else:
        trainer.fit(module, datamodule=data_module, ckpt_path=cfg.train_ckpt_path)
