# lewm

A minimal PyTorch implementation of [LeWorldModel (LeWM)](https://github.com/lucas-maes/le-wm). It gives you the model and the loss, and you write your own training loop. The only dependency is `torch`.

```bash
pip install -e .
```

## Usage

```python
import torch
from lewm import LeWM, LeWMConfig, LeWMLoss

cfg = LeWMConfig(action_dim=10)          # every hyperparameter lives here; defaults = paper
model = LeWM(cfg)
criterion = LeWMLoss(sigreg_weight=0.09)
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)

for pixels, actions in loader:           # (B, T, 3, 224, 224), (B, T, action_dim)
    emb, pred = model(pixels, actions)   # T = history_size + num_preds (default 4)
    losses = criterion(emb, pred)        # {"loss", "pred_loss", "sigreg_loss"}

    opt.zero_grad()
    losses["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
```

Paper training setup: AdamW with lr 5e-5 and weight decay 1e-3, linear warmup followed by cosine decay, batch size 128, bf16 autocast, gradient clipping at 1.0, and 100 epochs.

## Data

- `pixels` are float images normalized with ImageNet mean and std, resized to `img_size`. Shape: `(B, T, C, H, W)`.
- `actions[:, t]` is the action taken at frame `t`. The original data uses frameskip 5, so each step is the 5 skipped actions concatenated, and `action_dim = 5 * raw_action_dim`. The model sets NaN actions to 0.
- The original also z-score normalizes actions and proprioception.

## What's inside

- `LeWM.encode(pixels)` maps `(B, T, C, H, W)` to `(B, T, D)`. It runs a ViT, takes the CLS token, and passes it through a BatchNorm MLP projector.
- `LeWM.predict(emb, actions)` maps `(B, T, D)` to `(B, T, D)`. It is a causal transformer with AdaLN-zero action conditioning, and output `t` is the predicted embedding after action `t`.
- `LeWM.forward(pixels, actions)` returns `(emb, pred)` for training, where `pred[:, t]` predicts `emb[:, t + num_preds]`.
- `LeWM.rollout(pixels, actions)` imagines future embeddings autoregressively for planning. Call it in eval mode. A planning cost is, for example, `(rollout[:, -1] - goal_emb).pow(2).sum(-1)`.
- `LeWMLoss(emb, pred)` computes `MSE(pred, emb[:, num_preds:]) + λ · SIGReg(emb)`. There is no stop-gradient and no EMA target: the model trains end to end.

You can swap the encoder for any module that maps `(N, C, H, W)` to `(N, encoder_dim)`, for example `model.encoder = MyCNN()`.

## Visual probe

`VisualProbe` reconstructs an image from a single embedding, which shows what visual information a latent keeps. It is the decoder from App. D of the paper. The embedding is projected to the probe width and used as the only key and value token. One learnable query per output patch reads from it through a stack of cross-attention + MLP blocks, and each query is then projected linearly to a `patch_size × patch_size × C` pixel patch. The probe is a diagnostic and never part of LeWM's loss, so train it on detached embeddings, either after LeWM is trained or alongside it.

```python
import torch.nn.functional as F
from lewm import VisualProbe

probe = VisualProbe(in_dim=cfg.encoder_dim)      # decodes the ViT CLS token, as in the paper
opt = torch.optim.AdamW(probe.parameters(), lr=1e-4)

for pixels, _ in loader:
    x = pixels.flatten(0, 1)                     # (B*T, 3, 224, 224)
    with torch.no_grad():
        cls = model.encoder(x)                   # (B*T, encoder_dim), no gradient reaches LeWM
    loss = F.mse_loss(probe(cls), x)

    opt.zero_grad()
    loss.backward()
    opt.step()
```

Reconstructions are in the same normalized space as `pixels`. To view one, undo the ImageNet normalization: `img * std + mean`.

To probe during LeWM training without a second encoder pass, decode `emb.detach()` from `model(pixels, actions)` and use `in_dim=cfg.embed_dim`. A probe trained on that projector space can also decode the imagined futures from `model.rollout`.

### API

**`VisualProbe(in_dim=192, img_size=224, patch_size=16, out_channels=3, dim=256, depth=4, heads=8, mlp_dim=1024)`**

| Argument | Default | Meaning |
|---|---|---|
| `in_dim` | 192 | Size of the embedding to decode: `encoder_dim` for the CLS token, `embed_dim` for `encode` or `rollout` outputs. |
| `img_size` | 224 | Output height and width. Must be divisible by `patch_size`. |
| `patch_size` | 16 | Side length of the pixel patch each query decodes. There are `(img_size // patch_size)²` queries, 196 by default. |
| `out_channels` | 3 | Output image channels. |
| `dim` | 256 | Probe width. The embedding is projected to this size. |
| `depth` | 4 | Number of cross-attention + MLP blocks. |
| `heads` | 8 | Attention heads per block. |
| `mlp_dim` | 1024 | Hidden size of each block's MLP. |

**`probe(z)`** maps `(..., in_dim)` to `(..., out_channels, img_size, img_size)`. Any leading dimensions work, so `(B, T, D)` embeddings decode to `(B, T, C, H, W)` frames.

The default probe has 3.46M parameters. These details come from the paper: the CLS input, one learnable query per patch, cross-attention layers with residual MLPs, and the linear projection to 16×16×3 patches. The paper does not specify the following, so they were chosen here: `dim`, `depth`, `heads`, `mlp_dim`, pre-norm LayerNorms, no self-attention between queries (the paper describes only cross-attention), and the MSE loss.

Note that the context is a single token, so the attention softmax is always 1 and each block adds the same projected embedding to every query. Patches differ only through their learned queries and the MLPs, which means each patch is decoded independently of the others, and `heads` does not change what the probe can represent.

## Save / load

```python
from dataclasses import asdict

torch.save({"config": asdict(cfg), "model": model.state_dict()}, "lewm.pt")

ckpt = torch.load("lewm.pt")
model = LeWM(LeWMConfig(**ckpt["config"]))
model.load_state_dict(ckpt["model"])
```
