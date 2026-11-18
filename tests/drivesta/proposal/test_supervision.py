import ast
from pathlib import Path
import unittest
import numpy as np
import torch
from navsim.planning.training.drivesta.data.proposal.sampling import rotating_replay_indices
from navsim.agents.drivesta.losses.proposal import (
    expert_loss,
    auxiliary_loss,
    score_loss,
    COMPONENTS,
    compose_pdm,
)


class SupervisionTests(unittest.TestCase):
    def test_expert_best_of(self):
        pred = torch.randn(2, 64, 8, 3, requires_grad=True)
        gt = torch.randn(2, 8, 3)
        expected = (pred - gt[:, None]).norm(p=1, dim=-1).mean(-1).min(1).values.mean()
        torch.testing.assert_close(expert_loss(pred, gt), expected)

    def test_aux_empty_has_no_generator_gradient(self):
        p = torch.randn(2, 64, 8, 3, requires_grad=True)
        loss = auxiliary_loss(
            p, torch.randn(2, 8, 8, 3), torch.zeros(2, 8, dtype=torch.bool)
        )
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(p.grad.abs().sum().item(), 0)

    def test_aux_mask_ignores_padding(self):
        p = torch.randn(1, 64, 8, 3)
        a = torch.randn(1, 8, 8, 3)
        mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
        x = auxiliary_loss(p, a, mask)
        a[:, 2:] = 10000
        torch.testing.assert_close(x, auxiliary_loss(p, a, mask))

    def test_strata(self):
        y = np.zeros((400, 7))
        y[:, 6] = np.repeat([0, 0.5, 0.9, 0.98], 100)
        i = rotating_replay_indices(y, 0, "1234567890abcdef", 2)
        v = y[i, 6]
        self.assertEqual(len(set(i)), 64)
        self.assertEqual([sum(v == x) for x in [0, 0.5, 0.9, 0.98]], [8, 8, 24, 24])

    def test_strata_shortage(self):
        y = np.zeros((110, 7))
        y[100:, 6] = 1
        i = rotating_replay_indices(y, 0, "1234567890abcdef", 2)
        self.assertEqual(len(i), 18)
        self.assertEqual(sum(y[i, 6] == 0), 8)

    def test_score_components_keep_soft_and_safe_labels(self):
        logits = {k: torch.zeros(1, 1, requires_grad=True) for k in COMPONENTS}
        y = torch.tensor([[[0.0, 1.0, 0.7, 1.0, 1.0, 0.5, 0.0]]])
        loss, _ = score_loss(logits, y, torch.ones(1, 1, dtype=torch.bool), 0, 0)
        loss.backward()
        self.assertGreater(logits[COMPONENTS[0]].grad.item(), 0)
        self.assertLess(logits[COMPONENTS[1]].grad.item(), 0)
        self.assertEqual(logits[COMPONENTS[5]].grad.item(), 0)

    def test_score_padding_does_not_change_loss(self):
        d = {k: torch.randn(1, 4) for k in COMPONENTS}
        y = torch.rand(1, 4, 7)
        m = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
        a, _ = score_loss(d, y, m)
        y[:, 2:] = 0
        b, _ = score_loss(d, y, m)
        torch.testing.assert_close(a, b)

    def test_official_ddc_weight_zero(self):
        p = torch.ones(2, 6)
        p[1, 5] = 0
        torch.testing.assert_close(compose_pdm(p), torch.ones(2))

    def test_scorer_detaches_trajectory_coordinates(self):
        from omegaconf import OmegaConf
        from navsim.agents.drivesta.decoders.decoder_score.score import (
            ScoreDecoder,
        )

        cfg = OmegaConf.load(
            Path(__import__("navsim").__file__).resolve().parents[1]
            / "navsim/planning/script/config/common/agent/drivesta_proposal.yaml"
        ).config
        decoder = ScoreDecoder(cfg)
        proposals = torch.randn(2, 64, 8, 3, requires_grad=True)
        scene = torch.randn(2, 5, cfg.tf_d_model, requires_grad=True)
        ego = torch.randn(2, 1, cfg.tf_d_model, requires_grad=True)
        result = decoder(proposals, scene, ego)
        sum(x.sum() for x in result["pred_logit"].values()).backward()
        self.assertIsNone(proposals.grad)
        self.assertGreater(scene.grad.abs().sum().item(), 0)
        self.assertGreater(decoder.pos_embed[0].weight.grad.abs().sum().item(), 0)

    def test_masked_replay_excludes_padding_from_attention(self):
        from navsim.agents.drivesta.drivesta_model import score_replay

        class ContextSum(torch.nn.Module):
            def forward(self, p, s, e, v=None):
                return {
                    "pred_logit": {
                        "score": p[:, :, 0, 0] + p[:, :, 0, 0].sum(-1, keepdim=True)
                    }
                }

        p = torch.randn(2, 4, 8, 3)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
        scene = torch.zeros(2, 1, 1)
        ego = scene.clone()
        a = score_replay(ContextSum(), p, mask, scene, ego)["score"]
        p[~mask] = 10000
        b = score_replay(ContextSum(), p, mask, scene, ego)["score"]
        torch.testing.assert_close(a, b)

    def test_agent_import_independence(self):
        root = Path(__import__("navsim").__file__).resolve().parent / "agents/drivesta"
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = (
                    [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else (
                        [n.name for n in node.names]
                        if isinstance(node, ast.Import)
                        else []
                    )
                )
                for n in names:
                    self.assertFalse(
                        "clones" in n
                        or (
                            n.startswith("navsim.agents.")
                            and not n.startswith(
                                (
                                    "navsim.agents.drivesta",
                                    "navsim.agents.abstract_agent",
                                )
                            )
                        ),
                        (path, n),
                    )


if __name__ == "__main__":
    unittest.main()
