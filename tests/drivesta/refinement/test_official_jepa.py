"""Source-equivalence, masking and rectangular positional indexing checks."""

import torch
from pathlib import Path
from navsim.agents.drivesta.encoders.jepa import official_masks
from navsim.agents.drivesta.encoders.jepa_vendor.models.vision_transformer import (
    VisionTransformer,
)
from navsim.agents.drivesta.encoders.jepa_vendor.models.predictor import vit_predictor


def test_official_masks_disjoint_reproducible_and_nonempty():
    torch.manual_seed(3)
    before = torch.get_rng_state()
    first = official_masks(torch.tensor([29, 31]), "cpu")
    assert torch.equal(before, torch.get_rng_state())
    second = official_masks(torch.tensor([29, 31]), "cpu")
    for (c, p), (c2, p2) in zip(first, second):
        assert torch.equal(c, c2) and torch.equal(p, p2)
        assert c.shape[1] > 0 and p.shape[1] > 0
        for row in range(2):
            assert not set(c[row].tolist()) & set(p[row].tolist())
            assert c[row].max() < 2048 and p[row].max() < 2048


def test_masked_pixels_cannot_reach_visible_encoder_tokens():
    torch.manual_seed(4)
    model = VisionTransformer(
        img_size=(32, 64),
        patch_size=16,
        num_frames=8,
        tubelet_size=2,
        embed_dim=48,
        depth=1,
        num_heads=4,
        use_rope=True,
        handle_nonsquare_inputs=True,
    ).eval()
    clip = torch.randn(1, 3, 8, 32, 64)
    # One complete spatial tubelet, including every channel/pixel in it.
    changed = clip.clone()
    changed[:, :, 0:2, 0:16, 0:16] = 1000
    visible = torch.arange(1, 32)[None]
    with torch.no_grad():
        assert torch.equal(model(clip, masks=visible), model(changed, masks=visible))
        assert not torch.equal(model(clip), model(changed))


def test_square_predictor_explicit_grid_matches_original_forward():
    # Same official implementation with only the rectangle fix reverted.
    from navsim.agents.drivesta.encoders.jepa_vendor.models import predictor as module

    source = (
        Path(module.__file__)
        .read_text()
        .replace(
            ", T=self.grid_depth, H_patches=self.grid_height, W_patches=self.grid_width",
            "",
        )
    )
    namespace = dict(__name__=module.__name__, __package__=module.__package__)
    exec(compile(source, "original_square_predictor.py", "exec"), namespace)
    kwargs = dict(
        img_size=(32, 32),
        num_frames=8,
        embed_dim=48,
        predictor_embed_dim=48,
        depth=1,
        num_heads=4,
        use_rope=True,
        use_mask_tokens=True,
        num_mask_tokens=10,
    )
    model = vit_predictor(**kwargs).eval()
    reference = namespace["vit_predictor"](**kwargs).eval()
    reference.load_state_dict(model.state_dict(), strict=True)
    c = torch.tensor([[0, 3, 6, 9, 12, 15]])
    p = torch.tensor([[1, 2, 4, 5, 7, 8, 10, 11, 13, 14]])
    x = torch.randn(1, 6, 48)
    assert torch.equal(model(x, c, p), reference(x, c, p))


def test_rectangle_predictor_uses_actual_width_and_checkpoint_gradients():
    kwargs = dict(
        img_size=(32, 64),
        num_frames=8,
        embed_dim=48,
        predictor_embed_dim=48,
        depth=1,
        num_heads=4,
        use_rope=True,
        use_mask_tokens=True,
        num_mask_tokens=10,
    )
    model = vit_predictor(**kwargs)
    reference = vit_predictor(**kwargs, use_activation_checkpointing=True)
    reference.load_state_dict(model.state_dict(), strict=True)
    c = torch.arange(0, 32, 2)[None]
    p = torch.arange(1, 32, 2)[None]
    x = torch.randn(1, 16, 48, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    first = model(x, c, p)
    second = reference(y, c, p)
    assert torch.equal(first, second)
    first.square().mean().backward()
    second.square().mean().backward()
    assert torch.equal(x.grad, y.grad)
