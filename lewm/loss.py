import torch
import torch.nn.functional as F
from torch import nn


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer: Epps-Pulley normality test on random 1D projections."""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        phi = torch.exp(-t.square() / 2)  # characteristic function of N(0, 1)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, x):
        """x: (..., N, D) with N samples -> scalar statistic, averaged over leading dims and projections."""
        A = torch.randn(x.size(-1), self.num_proj, device=x.device, dtype=x.dtype)
        A = A / A.norm(dim=0)
        x_t = (x @ A).unsqueeze(-1) * self.t  # (..., N, num_proj, knots)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return ((err @ self.weights) * x.size(-2)).mean()


class LeWMLoss(nn.Module):
    """LeWM objective: next-embedding MSE + sigreg_weight * SIGReg(embeddings)."""

    def __init__(self, sigreg_weight=0.09, knots=17, num_proj=1024):
        super().__init__()
        self.sigreg_weight = sigreg_weight
        self.sigreg = SIGReg(knots, num_proj)

    def forward(self, emb, pred):
        """emb: (B, T, D) and pred: (B, T - num_preds, D) from `LeWM.forward` -> dict of losses."""
        target = emb[:, -pred.size(1):]  # = emb[:, num_preds:], no stop-gradient: trained end-to-end
        pred_loss = F.mse_loss(pred, target)
        sigreg_loss = self.sigreg(emb.transpose(0, 1))  # per time step, over the batch
        loss = pred_loss + self.sigreg_weight * sigreg_loss
        return {"loss": loss, "pred_loss": pred_loss, "sigreg_loss": sigreg_loss}
