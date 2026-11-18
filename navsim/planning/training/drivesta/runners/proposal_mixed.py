"""Train the independent expert-anchored DriveSTA agent."""

import json
import logging
import os
from pathlib import Path


from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytorch_lightning as pl

from navsim.planning.training.drivesta.data.proposal.mixed_dataset import (
    DriveSTAProposalMixedDataset,
    DriveSTAProposalMixedOnlinePDM,
)
from navsim.planning.training.drivesta.proposal.module import (
    DriveSTAProposalLightningModule,
)
from navsim.evaluate.drivesta.proposal.validation import OfficialPDM
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
    if cfg.agent.checkpoint_path:
        raise ValueError(
            "Proposal scaling starts from random planner weights; no pretrained planner initialization"
        )
    if cfg.agent.config.use_diffusion_decoder or cfg.agent.config.use_vlm_feature_cache:
        raise ValueError(
            "DriveSTA Proposal requires proposal/scorer pretraining without VLM or diffusion"
        )
    if cfg.validation_run or cfg.trainer.params.limit_val_batches:
        raise ValueError("Joint trainval training uses external NAVTEST evaluation")
    train = DriveSTAProposalMixedDataset(
        cfg.feedback.index_path, cfg.agent.config, cfg.seed
    )
    val = []
    split = "trainval"
    allowed = set(cfg.train_logs) | set(cfg.val_logs)
    if any(r["log_name"] not in allowed for r in train.real.records):
        raise ValueError("Real held-out log leaked into training")
    if len(train) != 232672:
        raise ValueError("Unexpected DATA_SIZE")
    cfg.agent.scheduler_args.dataset_size = len(train)
    if cfg.preflight_only:
        print(
            json.dumps(
                dict(
                    data_size=len(train),
                    real=len(train.real),
                    synthetic=len(train.synthetic),
                    batch_per_gpu=cfg.dataloader.params.batch_size,
                    gpus=cfg.trainer.params.devices,
                    planner_initialization=None,
                    loss=OmegaConf.to_container(cfg.agent.loss, resolve=True),
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
        online=DriveSTAProposalMixedOnlinePDM(
            train.official_cache_root, train.records, cfg.feedback.online_workers
        ),
        target_policy=train.target_policy,
        data_protocol=dict(
            index_identity=train.index_identity,
            seed=cfg.seed,
            auxiliary_bank_capacity=cfg.feedback.auxiliary_bank_capacity,
            auxiliary_samples=cfg.feedback.auxiliary_samples,
            sampling="epoch_index_pairs_consumed_cursor_v1",
            sources=train.spec,
        ),
    )
    if (
        os.environ.get("LOCAL_RANK", "0") == "0"
        and not cfg.train_ckpt_path
        and Path(cfg.output_dir, "initialization.json").exists()
    ):
        raise ValueError(
            "Existing run requires explicit resume; choose a fresh output directory"
        )
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())
    if trainer.is_global_zero:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        if not cfg.train_ckpt_path:
            Path(cfg.output_dir, "initialization.json").write_text(
                json.dumps(
                    dict(
                        kind="scratch_V2_proposal_data_scaling",
                        planner_checkpoint=None,
                        seed=cfg.seed,
                        backbone="generic_DINO_pretrained",
                        index_identity=train.index_identity,
                    ),
                    indent=2,
                )
                + "\n"
            )
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
