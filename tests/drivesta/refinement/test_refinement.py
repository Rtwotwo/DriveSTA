from types import SimpleNamespace
import copy
import numpy as np
import pytest
import torch
from navsim.agents.drivesta.losses.residual_wta import (
    make_pairs,
    paired_wta,
    residual_target,
    wrap,
)
from navsim.agents.drivesta.decoders.decoder_traj.dit import DiffusionTrajectoryDecoder
from navsim.agents.drivesta.decoders.decoder_score.score import ScoreDecoder
from navsim.agents.drivesta.encoders.jepa import preprocess_front, VideoCondition
from navsim.agents.drivesta.losses.refinement import score_loss, COMPONENTS


def config():
    return SimpleNamespace(
        use_diffusion_decoder=True,
        num_poses=8,
        tf_d_model=32,
        tf_d_ffn=64,
        diffusion_type="ddim",
        diffusion_num_layers=1,
        diffusion_num_heads=4,
        diffusion_d_ffn=64,
        diffusion_ego_status_dim=8,
        diffusion_use_ar_proposal_tokens=True,
        diffusion_eval_noise_seed=27,
        diffusion_pair_chunk=4,
        diffusion_num_inference_steps=2,
        pair_max_ade=3.0,
        pair_max_fde=6.0,
        pair_max_heading=0.5,
        wta_targets_per_parent=4,
        proposal_num=5,
        one_token_per_traj=True,
        noc=1.0,
        dac=1.0,
        ddc=0.0,
        ttc=5.0,
        ep=5.0,
        comfort=2.0,
        scorer_ref_num=1,
        double_score=False,
        agent_pred=False,
        area_pred=False,
        bev_map=False,
        bev_agent=False,
    )


def fixture():
    parents = torch.zeros(2, 5, 8, 3)
    bank = torch.zeros(2, 4, 8, 3)
    bank[:, 0, :, 0] = 0.5
    bank[:, 1, :, 0] = 1.0
    bank[:, 2, :, 0] = 20.0
    labels = torch.ones(2, 4, 7)
    mask = torch.tensor([[1, 1, 1, 0], [0, 0, 0, 0]], dtype=torch.bool)
    return parents, dict(
        auxiliary_trajectories=bank, auxiliary_mask=mask, auxiliary_labels=labels
    )


def test_pairing_masks_and_retained():
    p, t = fixture()
    r = make_pairs(p, t, torch.tensor([0, 4]), config())
    assert r["matched"][0].all() and not r["matched"][1].any()
    assert r["target_mask"][0].sum() == 10
    assert r["target_mask"][1].sum() == 5
    assert r["active"].sum() == 8 and not r["active"][0, 0] and not r["active"][1, 4]
    assert torch.equal(r["targets"][1], p[1, :, None].expand_as(r["targets"][1]))


def test_joint_all_parents_are_active_and_detached():
    p, t = fixture()
    p.requires_grad_()
    r = make_pairs(p.detach(), t, None, config())
    assert r["active"].all() and not r["parents"].requires_grad
    assert not r["targets"].requires_grad
    d = DiffusionTrajectoryDecoder(config())
    memory = torch.randn(2, 8, 32, requires_grad=True)
    # Nonzero output removes intentional adaLN/output initialization blocking.
    torch.nn.init.normal_(d.traj_output_proj.weight, std=0.01)
    w = paired_wta(d.forward_pairs(r, memory, torch.randn(2, 8)))
    (w["velocity"] + w["trajectory"]).backward()
    assert p.grad is None
    assert memory.grad is not None and memory.grad.abs().sum() > 0


def test_joint_loss_contains_ar_dit_and_true_score_terms():
    from navsim.agents.drivesta.losses.refinement import (
        DriveSTALoss,
        expert_loss,
        auxiliary_loss,
    )

    p, t = fixture()
    p.requires_grad_()
    pairs = make_pairs(p.detach(), t, None, config())
    decoded = (pairs["targets"] + 0.1).requires_grad_()
    velocity = torch.ones_like(decoded, requires_grad=True)
    pairs.update(
        proposals=decoded,
        pred_velocity=velocity,
        target_velocity=torch.zeros_like(velocity),
    )
    labels = torch.ones(2, 5, 7)
    labels[0, 0, 0] = 0
    labels[0, 0, 6] = 0
    logits = {k: torch.zeros(2, 5, requires_grad=True) for k in COMPONENTS}
    t.update(
        trajectory=torch.ones(2, 8, 3),
        trajectory_long=torch.ones(2, 8, 3) * 2,
        ar_auxiliary_trajectories=t["auxiliary_trajectories"],
        ar_auxiliary_mask=t["auxiliary_mask"],
        native_labels=labels,
        replay_labels=labels,
        replay_mask=torch.ones(2, 5, dtype=torch.bool),
    )
    pred = dict(
        diffusion_training=pairs,
        ar_proposals=p,
        pred_logit=logits,
        replay_pred_logit=logits,
        jepa_future=dict(loss=torch.tensor(2.0)),
    )
    loss = DriveSTALoss(phase="joint")
    loss.epoch_progress = 3
    out = loss(t, pred)
    expected = (
        out["native_score_loss"]
        + 0.25 * out["replay_loss"]
        + out["expert_loss"]
        + 0.25 * out["auxiliary_loss"]
        + 10 * out["dit_velocity"]
        + 20 * out["dit_trajectory"]
    )
    assert torch.allclose(out["loss"], expected + 2.0)
    out["loss"].backward()
    assert p.grad.abs().sum() > 0 and velocity.grad.abs().sum() > 0
    assert logits[COMPONENTS[0]].grad.abs().sum() > 0


def test_reject_high_pdm_bad_direction():
    p, t = fixture()
    t["auxiliary_labels"][0, 0, 5] = 0.5
    with pytest.raises(ValueError):
        make_pairs(p, t, torch.tensor([0, 0]), config())


def test_periodic_heading():
    p = torch.zeros(1, 8, 3)
    t = p.clone()
    p[..., 2] = 3.13
    t[..., 2] = -3.13
    assert residual_target(t, p)[..., 2].abs().max() < 0.03
    assert wrap(torch.tensor([7.0])).abs().max() < torch.pi


def test_wta_same_winner_for_both_terms_and_no_padding_gradient():
    p, t = fixture()
    r = make_pairs(p, t, torch.tensor([0, 4]), config())
    pred = r["targets"].clone()
    pred[:, :, 0, :, 0] += 2.0
    pred[:, :, 1, :, 0] += 0.1
    pred.requires_grad_()
    v = torch.full_like(pred, 3.0, requires_grad=True)
    truth = torch.zeros_like(v)
    truth[:, :, 1] = 2.0
    r.update(proposals=pred, pred_velocity=v, target_velocity=truth)
    w = paired_wta(r)
    (w["velocity"] + w["trajectory"]).backward()
    assert w["matched_velocity"] == 0.5  # scene1 mean=1, empty scene2=0
    assert not v.grad[0, :, 0].any() and v.grad[0, 1, 1].abs().sum() > 0
    assert not pred.grad[0, 0].any() and not pred.grad[1, 4].any()


def test_v_parameterization_inverts_forward_noise():
    d = DiffusionTrajectoryDecoder(config())
    x = torch.randn(2, 5, 8, 3)
    e = torch.randn_like(x)
    t = torch.tensor([0, 999])
    z = d.noise_scheduler.add_noise(x, e, t)
    v = d.noise_scheduler.get_velocity(x, e, t)
    assert torch.allclose(d._predict_x0_from_velocity(z, t, v), x, atol=1e-6)
    assert d.noise_scheduler.config.clip_sample is False


def test_shared_noise_target_binding_and_backward(monkeypatch):
    d = DiffusionTrajectoryDecoder(config())
    p, t = fixture()
    r = make_pairs(p, t, torch.tensor([0, 4]), config())
    monkeypatch.setattr(torch, "randn_like", lambda x: torch.ones_like(x) * 0.25)
    monkeypatch.setattr(
        torch,
        "randint",
        lambda high, size, device: torch.full(
            size, 100, device=device, dtype=torch.long
        ),
    )
    out = d.forward_pairs(r, torch.randn(2, 8, 32), torch.randn(2, 8))
    delta = residual_target(r["targets"], p[:, :, None]).flatten(1, 2)
    expected = d.noise_scheduler.get_velocity(
        delta, torch.full_like(delta, 0.25), torch.tensor([100, 100])
    ).reshape_as(out["target_velocity"])
    assert torch.equal(out["target_velocity"], expected)
    w = paired_wta(out)
    (w["velocity"] + w["trajectory"]).backward()
    assert (
        torch.isfinite(d.traj_output_proj.weight.grad).all()
        and d.traj_output_proj.weight.grad.norm() > 0
    )


def test_scorer_zero_gate_restores_original_and_learns_gate():
    c = config()
    c.tf_d_model = 32
    d = ScoreDecoder(c).eval()
    p = torch.randn(2, 5, 8, 3)
    s = torch.randn(2, 64, 32)
    ego = torch.randn(2, 1, 32)
    base = d(p, s, ego)
    memory = torch.randn(2, 17, 32)
    extra = d(p, s, ego, memory)
    for k in COMPONENTS:
        assert torch.equal(base["pred_logit"][k], extra["pred_logit"][k])
    sum(x.sum() for x in extra["pred_logit"].values()).backward()
    assert d.jepa_gate.grad.abs() > 0


def test_score_label_soft_values_and_zero_component_semantics():
    logits = {k: torch.zeros(2, 5, requires_grad=True) for k in COMPONENTS}
    labels = torch.ones(2, 5, 7)
    labels[0, 0, 0] = 0.0
    labels[0, 0, 6] = 0.0
    labels[0, 1, 5] = 0.5
    value, _ = score_loss(
        logits, labels, torch.ones(2, 5, dtype=torch.bool), rank_weight=0.05
    )
    value.backward()
    assert logits["no_at_fault_collisions"].grad[0, 0] > 0
    assert logits["drivable_area_compliance"].grad[0, 0] < 0


def test_history_condition_shapes_and_finite_grads():
    for pooled, number in [(False, 513), (True, 129)]:
        d = VideoCondition(32, pooled)
        x = d(torch.randn(2, 512, 1024), torch.randn(2, 4, 11))
        assert x.shape == (2, number, 32)
        x.square().mean().backward()
        assert torch.isfinite(d.project[1].weight.grad).all()


def test_front_observation_preprocess():
    x = preprocess_front(
        [
            np.zeros((900, 1600, 3), dtype=np.uint8),
            np.full((900, 1600, 3), 255, dtype=np.uint8),
        ]
    )
    assert x.shape == (3, 2, 256, 512) and x[:, 0].mean() < x[:, 1].mean()
    with pytest.raises(ValueError):
        preprocess_front([None, np.zeros((900, 1600, 3))])


def test_strict_load_cannot_bypass_untrained_dit():
    from navsim.agents.drivesta.drivesta_agent import DriveSTAAgent

    a = SimpleNamespace(
        _load_checkpoint_state=lambda: {},
        feedback_checkpoint_training={
            "loss": "DriveSTALoss",
            "refinement_phase": "scorer_ar",
        },
        _config=SimpleNamespace(use_diffusion_decoder=True, training_phase="joint"),
        _checkpoint_strict_load=True,
    )
    with pytest.raises(ValueError, match="trained joint"):
        DriveSTAAgent._load_pretrained_checkpoint(a)


def test_resume_protocol_changes_with_pairing_and_rollout():
    from navsim.planning.training.drivesta.refinement.module import DriveSTALightningModule

    c = config()
    c.training_phase = "joint"
    c.jepa_checkpoint = "official_encoder.pt"
    c.diffusion_delta_scale = 1.0
    c.ddim_num_train_timesteps = 1000
    c.ddim_beta_schedule = "scaled_linear"

    for k, v in dict(
        jepa_weight_decay=0.04,
        jepa_encoder_lr=1e-5,
        jepa_predictor_lr=1e-4,
        jepa_layer_decay=0.9,
        jepa_ema_momentum=0.996,
        jepa_predictor_dim=256,
        jepa_predictor_layers=12,
        jepa_predictor_heads=12,
        jepa_mask_tokens=10,
        jepa_video_protocol="drive_jepa_expert_video8_full14_v1",
        jepa_activation_checkpointing=True,
    ).items():
        setattr(c, k, v)
    a = SimpleNamespace(
        _config=c, loss=SimpleNamespace(protocol=lambda: {}), _lr_args={}
    )
    m = DriveSTALightningModule(a, 12)
    old = m.supervision_protocol()
    c.pair_max_ade = 30.0
    assert m.supervision_protocol() != old
    old = m.supervision_protocol()
    c.diffusion_eval_noise_seed = 123
    assert m.supervision_protocol() != old


def test_direct_agent_requires_token():
    from navsim.agents.drivesta.drivesta_agent import DriveSTAAgent

    with pytest.raises(ValueError, match="token"):
        DriveSTAAgent.compute_trajectory(
            SimpleNamespace(), SimpleNamespace(token=None)
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("diffusion_delta_scale", 2.0),
        ("ddim_num_train_timesteps", 500),
        ("ddim_beta_schedule", "linear"),
        ("diffusion_num_inference_steps", 20),
        ("diffusion_eval_noise_seed", 999),
    ],
)
def test_phase_migration_rejects_generator_semantic_drift(key, value):
    from navsim.agents.drivesta.drivesta_agent import DriveSTAAgent

    c = SimpleNamespace(
        use_diffusion_decoder=True,
        training_phase="joint",
        diffusion_delta_scale=1.0,
        ddim_num_train_timesteps=1000,
        ddim_beta_schedule="scaled_linear",
        diffusion_num_inference_steps=12,
        diffusion_eval_noise_seed=27,
    )
    saved = {k: v for k, v in vars(c).items() if k != "training_phase"}
    saved[key] = value
    a = SimpleNamespace(
        _load_checkpoint_state=lambda: {},
        _config=c,
        _checkpoint_strict_load=True,
        feedback_checkpoint_training=dict(
            loss="DriveSTALoss",
            refinement_dit_trained=True,
            refinement_phase="joint",
            supervision=dict(
                refinement=saved,
                frozen_observation_precision="fp32",
                scorer_candidate_precision="fp32",
            ),
        ),
    )
    with pytest.raises(ValueError, match="generator protocol mismatch"):
        DriveSTAAgent._load_pretrained_checkpoint(a)


def test_nonofficial_score_weight_rejected_before_model_load():
    from navsim.agents.drivesta.drivesta_agent import DriveSTAAgent

    c = SimpleNamespace(noc=1.0, dac=1.0, ddc=1.0, ttc=5.0, ep=5.0, comfort=2.0)
    with pytest.raises(ValueError, match="scoring weights"):
        DriveSTAAgent(c, lr_args={})


@pytest.mark.parametrize("source", ["DriveSTAProposalLoss", "DriveSTAProposalLoss"])
def test_original_and_scaled_proposal_loading_share_shape_checks(source):
    from navsim.agents.drivesta.drivesta_agent import DriveSTAAgent

    calls = []

    def load(state, strict):
        calls.append((state, strict))
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    m = SimpleNamespace(
        jepa=SimpleNamespace(reset_target=lambda: None),
        reference_scorer=SimpleNamespace(load_state_dict=lambda state, strict: None),
        score_decoder=SimpleNamespace(state_dict=lambda: {}),
    )
    a = SimpleNamespace(
        _load_checkpoint_state=lambda: {"original_tensor": torch.ones(1)},
        _config=SimpleNamespace(use_diffusion_decoder=True, training_phase="joint", jepa_load_pretrained=True),
        _checkpoint_strict_load=False,
        feedback_checkpoint_training=dict(loss=source),
        load_state_dict=load,
        _drivesta_model=m,
    )
    DriveSTAAgent._load_pretrained_checkpoint(a)
    assert len(calls) == 1 and calls[0][1] is False


def test_provisional_pair_v4_checkpoint_rejected_by_protocol():
    from navsim.agents.drivesta.drivesta_agent import DriveSTAAgent

    refinement = dict(
        diffusion_delta_scale=1.0,
        ddim_num_train_timesteps=1000,
        ddim_beta_schedule="scaled_linear",
        diffusion_num_inference_steps=12,
        diffusion_eval_noise_seed=27,
    )
    saved = dict(
        refinement=refinement,
        jepa_gradient_policy="expert_future_only_detach_before_both_planning_adapters",
        future_protocol="expert_future_pairs_v1_layernorm128_L1_epoch_rotation",
    )
    agent = SimpleNamespace(
        _load_checkpoint_state=lambda: {},
        _config=SimpleNamespace(use_diffusion_decoder=True, **refinement),
        feedback_checkpoint_training=dict(
            loss="DriveSTALoss",
            refinement_phase="joint",
            refinement_dit_trained=True,
            supervision=saved,
        ),
    )
    with pytest.raises(ValueError, match="incompatible JEPA auxiliary protocol"):
        DriveSTAAgent._load_pretrained_checkpoint(agent)


def test_mixed_v2_jepa_source_rejected_before_overwriting_encoder():
    from navsim.agents.drivesta.drivesta_agent import DriveSTAAgent

    refinement = dict(
        diffusion_delta_scale=1.0,
        ddim_num_train_timesteps=1000,
        ddim_beta_schedule="scaled_linear",
        diffusion_num_inference_steps=12,
        diffusion_eval_noise_seed=27,
    )
    key = "_drivesta_model.jepa.encoder.weight"
    agent = SimpleNamespace(
        _load_checkpoint_state=lambda: {key: torch.ones(1)},
        state_dict=lambda: {key: torch.zeros(1)},
        _checkpoint_strict_load=False,
        _config=SimpleNamespace(use_diffusion_decoder=True, **refinement, jepa_load_pretrained=True),
        feedback_checkpoint_training=dict(
            loss="FeedbackRefinementV2TrajectoryLoss",
            refinement_phase="joint",
            refinement_dit_trained=True,
            supervision=dict(refinement=refinement),
        ),
    )
    with pytest.raises(ValueError, match="Initialize DriveSTA from"):
        DriveSTAAgent._load_pretrained_checkpoint(agent)
