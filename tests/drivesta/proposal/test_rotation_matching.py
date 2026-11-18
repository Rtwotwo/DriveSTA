import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from torch.utils.data import Dataset
from navsim.planning.training.drivesta.data.proposal.sampling import (
    cyclic_indices,
    rotating_replay_indices,
    EpochDistributedSampler,
    EpochDataLoader,
)
from navsim.agents.drivesta.losses.proposal import (
    auxiliary_loss,
    frontier_rank_loss,
    trajectory_distance,
    DriveSTAProposalLoss,
    COMPONENTS,
)

TOKEN = "0123456789abcdef"


class IndexedDataset(Dataset):
    def __len__(self):
        return 17

    def __getitem__(self, key):
        return torch.tensor(key)


class RotationTests(unittest.TestCase):
    def test_32_targets_covered_in_four_epochs(self):
        chunks = [cyclic_indices(np.arange(32), 8, e, TOKEN) for e in range(4)]
        self.assertEqual(len(set(np.concatenate(chunks))), 32)
        np.testing.assert_array_equal(
            chunks[0], cyclic_indices(np.arange(32), 8, 4, TOKEN)
        )

    def test_variable_banks_never_duplicate(self):
        for n in range(33):
            chunks = [cyclic_indices(np.arange(n), 8, e, TOKEN) for e in range(5)]
            for c in chunks:
                self.assertEqual(len(set(c)), min(n, 8))
            self.assertEqual(len(set(np.concatenate(chunks))), n)

    def test_resume_and_seed(self):
        a = cyclic_indices(np.arange(32), 8, 3, TOKEN, 2)
        np.testing.assert_array_equal(a, cyclic_indices(np.arange(32), 8, 3, TOKEN, 2))
        self.assertFalse(
            np.array_equal(a, cyclic_indices(np.arange(32), 8, 3, TOKEN, 17))
        )

    def test_replay_rotation_exposes_each_stratum(self):
        y = np.zeros((256, 7))
        y[:, 6] = np.repeat([0, 0.5, 0.9, 1.0], 64)
        seen = set()
        for epoch in range(8):
            indices = rotating_replay_indices(y, epoch, TOKEN)
            self.assertEqual(len(set(indices)), 64)
            self.assertEqual(
                [sum(y[indices, 6] == v) for v in (0, 0.5, 0.9, 1.0)], [8, 8, 24, 24]
            )
            seen.update(indices)
        self.assertEqual(len(seen), 256)

    def test_ddp_partition_and_epoch(self):
        ds = IndexedDataset()
        all_rows = []
        for rank in range(4):
            s = EpochDistributedSampler(ds, num_replicas=4, rank=rank, seed=2)
            s.set_epoch(7)
            rows = list(s)
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(e == 7 for i, e in rows))
            all_rows.extend(i for i, e in rows)
        self.assertEqual(set(all_rows), set(range(17)))

    def test_persistent_workers_receive_new_epoch(self):
        ds = IndexedDataset()
        s = EpochDistributedSampler(ds, num_replicas=1, rank=0)
        loader = EpochDataLoader(
            ds, batch_size=4, sampler=s, num_workers=2, persistent_workers=True
        )
        for epoch in (0, 1, 6):
            s.set_epoch(epoch)
            rows = torch.cat(list(loader))
            self.assertTrue((rows[:, 1] == epoch).all())
        del loader

    def test_consumed_cursor_ignores_prefetch(self):
        ds = IndexedDataset()
        s = EpochDistributedSampler(ds, num_replicas=1, rank=0, seed=2)
        loader = EpochDataLoader(
            ds, batch_size=4, sampler=s, num_workers=2, persistent_workers=True
        )
        it = iter(loader)
        first = next(it)
        loader.acknowledge(0, 4)
        state = loader.state_dict()
        restored = EpochDataLoader(
            ds,
            batch_size=4,
            sampler=EpochDistributedSampler(ds, num_replicas=1, rank=0, seed=2),
        )
        restored.load_state_dict(state)
        remaining = torch.cat(list(restored))
        self.assertEqual(len(remaining), 13)
        self.assertFalse(set(first[:, 0].tolist()) & set(remaining[:, 0].tolist()))
        self.assertEqual(
            set(torch.cat([first, remaining])[:, 0].tolist()), set(range(17))
        )
        del it, loader


class MatchingRankingTests(unittest.TestCase):
    def test_only_native_rank_enters_total_and_warms_up(self):
        torch.manual_seed(2)
        proposals = torch.randn(1, 64, 8, 3, requires_grad=True)
        pred = dict(
            proposals=proposals,
            proposal_list=[proposals],
            pred_logit={k: torch.randn(1, 64, requires_grad=True) for k in COMPONENTS},
            replay_pred_logit={
                k: torch.randn(1, 64, requires_grad=True) for k in COMPONENTS
            },
        )
        targets = dict(
            trajectory=torch.zeros(1, 8, 3),
            trajectory_long=torch.zeros(1, 8, 3),
            auxiliary_trajectories=torch.zeros(1, 8, 8, 3),
            auxiliary_mask=torch.zeros(1, 8, dtype=torch.bool),
            native_labels=torch.rand(1, 64, 7),
            replay_labels=torch.rand(1, 64, 7),
            replay_mask=torch.ones(1, 64, dtype=torch.bool),
        )
        baseline = DriveSTAProposalLoss(rank_weight=0.0)
        ranked = DriveSTAProposalLoss(rank_weight=0.05)
        for epoch, scale in [(0, 0), (1, 0.5), (2, 1)]:
            baseline.epoch_progress = ranked.epoch_progress = epoch
            a = baseline(targets, pred)
            b = ranked(targets, pred)
            torch.testing.assert_close(
                b["loss"] - a["loss"],
                0.05 * scale * b["native_rank_loss"],
                atol=2e-6,
                rtol=1e-5,
            )
            torch.testing.assert_close(a["replay_score_loss"], b["replay_score_loss"])

    def test_runtime_configs_match_approved_weights(self):
        from omegaconf import OmegaConf

        root = Path(__import__("navsim").__file__).resolve().parents[1]
        for profile, rank, size in [
            ("base_proposal", 0.0, 103263),
            ("scale_proposal", 0.05, 232672),
        ]:
            cfg = OmegaConf.load(
                root / "navsim/planning/script/config/checkpoints" / (profile + ".yaml")
            ).agent
            self.assertEqual(
                cfg._target_,
                "navsim.agents.drivesta.drivesta_agent.DriveSTAAgent",
            )
            self.assertEqual(cfg.loss.rank_weight, rank)
            self.assertEqual(cfg.loss.rank_warmup_epochs, 2.0)
            self.assertEqual(cfg.loss.matching, "injective")
            self.assertEqual(cfg.loss.auxiliary_weight, 0.25)
            self.assertEqual(cfg.loss.replay_weight, 0.25)
            self.assertEqual(cfg.config.proposal_num, 64)
            self.assertEqual(cfg.scheduler_args.dataset_size, size)
            training = OmegaConf.load(
                root
                / "navsim/planning/script/config/training"
                / ("drivesta_" + profile + ".yaml")
            )
            self.assertEqual(training.feedback.expected_trainval_scenes, size)
            self.assertFalse(training.trainer.params.use_distributed_sampler)

    def test_injective_matching_uses_two_predictions(self):
        p = torch.zeros(1, 3, 8, 3)
        p[0, 1, :, 0] = 2
        p[0, 2, :, 0] = 10
        p.requires_grad_()
        h = torch.zeros(1, 2, 8, 3)
        h[0, 0, :, 0] = 0.1
        h[0, 1, :, 0] = 0.2
        mask = torch.ones(1, 2, dtype=torch.bool)
        nearest = auxiliary_loss(p, h, mask, matching="nearest")
        matched = auxiliary_loss(p, h, mask)
        matched.backward()
        self.assertAlmostEqual(nearest.item(), 0.15, places=5)
        self.assertAlmostEqual(matched.item(), 0.95, places=5)
        self.assertGreater(p.grad[0, 0].abs().sum(), 0)
        self.assertGreater(p.grad[0, 1].abs().sum(), 0)
        self.assertEqual(p.grad[0, 2].abs().sum(), 0)

    def test_heading_wrap(self):
        a = torch.zeros(1, 8, 3)
        b = a.clone()
        a[..., 2] = np.pi - 0.01
        b[..., 2] = -np.pi + 0.01
        self.assertAlmostEqual(trajectory_distance(a, b).item(), 0.005, places=5)

    def test_empty_or_nan_padding_ignored(self):
        p = torch.randn(2, 64, 8, 3, requires_grad=True)
        h = torch.full((2, 8, 8, 3), float("nan"))
        loss = auxiliary_loss(p, h, torch.zeros(2, 8, dtype=torch.bool))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(p.grad.abs().sum(), 0)

    def test_rank_no_tie_or_small_gap_preferences(self):
        for y in [torch.ones(2, 64), torch.tensor([[0.997, 1.0]]).expand(2, -1)]:
            p = torch.rand_like(y, requires_grad=True)
            loss, pairs = frontier_rank_loss(p, y, torch.ones_like(y, dtype=torch.bool))
            loss.backward()
            self.assertEqual(loss.item(), 0)
            self.assertEqual(pairs.item(), 0)
            self.assertEqual(p.grad.abs().sum(), 0)

    def test_frontier_union_and_gradient_direction(self):
        p = torch.tensor([[0.9, 0.2, 0.4, 0.3]], requires_grad=True)
        y = torch.tensor([[0.1, 1.0, 0.5, 0.3]], requires_grad=True)
        loss, pairs = frontier_rank_loss(
            p, y, torch.ones_like(p, dtype=torch.bool), top_k=1
        )
        loss.backward()
        # All ordered positive pairs except the nonfrontier pair (2,3).
        self.assertEqual(pairs.item(), 5)
        self.assertGreater(p.grad[0, 0], 0)
        self.assertLess(p.grad[0, 1], 0)
        self.assertIsNone(y.grad)

    def test_rank_scene_normalization_and_padding(self):
        p = torch.tensor([[0.8, 0.2, 0.5, 99.0]])
        y = torch.tensor([[0.2, 1.0, 0.5, 0.0]])
        mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
        a, _ = frontier_rank_loss(p, y, mask, top_k=1)
        b, _ = frontier_rank_loss(p[:, :3], y[:, :3], mask[:, :3], top_k=1)
        c, _ = frontier_rank_loss(
            p.repeat(3, 1), y.repeat(3, 1), mask.repeat(3, 1), top_k=1
        )
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(a, c)


if __name__ == "__main__":
    unittest.main()
