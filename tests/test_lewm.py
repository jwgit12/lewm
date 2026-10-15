import torch

import torch.nn.functional as F

from lewm import LeWM, LeWMConfig, LeWMLoss, VisualProbe


def tiny_model():
    cfg = LeWMConfig(
        action_dim=4, img_size=32, patch_size=8, encoder_depth=2,
        proj_hidden_dim=64, predictor_depth=2, predictor_mlp_dim=64,
    )
    return LeWM(cfg)


def test_train_step():
    model, criterion = tiny_model(), LeWMLoss(num_proj=64)
    pixels, actions = torch.randn(8, 4, 3, 32, 32), torch.randn(8, 4, 4)

    emb, pred = model(pixels, actions)
    assert emb.shape == (8, 4, 192) and pred.shape == (8, 3, 192)

    losses = criterion(emb, pred)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert all(p.grad is not None for p in model.parameters())


def test_predictor_is_causal():
    model = tiny_model().eval()
    emb, actions = torch.randn(2, 3, 192), torch.randn(2, 3, 4)
    pred = model.predict(emb, actions)

    emb[:, -1], actions[:, -1] = 0.0, 0.0
    assert torch.allclose(model.predict(emb, actions)[:, :-1], pred[:, :-1], atol=1e-6)


def test_rollout():
    model = tiny_model().eval()
    with torch.no_grad():
        out = model.rollout(torch.randn(2, 3, 3, 32, 32), torch.randn(2, 7, 4))
    assert out.shape == (2, 8, 192)


def test_visual_probe():
    model = tiny_model()
    probe = VisualProbe(in_dim=192, img_size=32, patch_size=8, dim=64, depth=2, heads=4, mlp_dim=128)
    pixels = torch.randn(2, 3, 3, 32, 32)

    cls = model.encoder(pixels.flatten(0, 1)).detach()  # (6, 192) CLS tokens, no gradient into LeWM
    recon = probe(cls)
    assert recon.shape == (6, 3, 32, 32)
    assert probe(cls.view(2, 3, -1)).shape == (2, 3, 3, 32, 32)

    F.mse_loss(recon, pixels.flatten(0, 1)).backward()
    assert all(p.grad is not None for p in probe.parameters())
    assert all(p.grad is None for p in model.parameters())
