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

## Save / load

```python
from dataclasses import asdict

torch.save({"config": asdict(cfg), "model": model.state_dict()}, "lewm.pt")

ckpt = torch.load("lewm.pt")
model = LeWM(LeWMConfig(**ckpt["config"]))
model.load_state_dict(ckpt["model"])
```
