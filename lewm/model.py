import torch
import torch.nn.functional as F
from torch import nn

from .config import LeWMConfig


class ViTEncoder(nn.Module):
    """Pre-norm ViT returning the CLS token. Same architecture and init as the HF ViTModel used by LeWM."""

    def __init__(self, img_size, patch_size, in_channels, dim, depth, heads, mlp_dim):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        layer = nn.TransformerEncoderLayer(
            dim, heads, mlp_dim, dropout=0.0, activation="gelu",
            layer_norm_eps=1e-12, batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim, eps=1e-12)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.trunc_normal_(m.in_proj_weight, std=0.02)
                nn.init.zeros_(m.in_proj_bias)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """x: (N, C, H, W) -> (N, dim)"""
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1) + self.pos_embed
        return self.norm(self.blocks(x)[:, 0])


class Attention(nn.Module):
    """Causal multi-head self-attention."""

    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, 3 * heads * dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(heads * dim_head, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.to_qkv(self.norm(x)).view(B, T, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=True)
        return self.to_out(out.transpose(1, 2).reshape(B, T, -1))


class ConditionalBlock(nn.Module):
    """Transformer block conditioned on action embeddings via AdaLN-zero."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c).chunk(6, dim=-1)
        x = x + gate1 * self.attn(self.norm1(x) * (1 + scale1) + shift1)
        x = x + gate2 * self.mlp(self.norm2(x) * (1 + scale2) + shift2)
        return x


class Predictor(nn.Module):
    """Causal transformer: output t predicts the next embedding from inputs <= t."""

    def __init__(self, num_frames, dim, depth, heads, dim_head, mlp_dim, dropout, emb_dropout):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_frames, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.blocks = nn.ModuleList(
            ConditionalBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, c):
        """x: (B, T, dim) embeddings, c: (B, T, dim) action embeddings -> (B, T, dim)"""
        if x.size(1) > self.pos_embed.size(1):
            raise ValueError(f"predictor got {x.size(1)} frames, at most history_size={self.pos_embed.size(1)} allowed")
        x = self.dropout(x + self.pos_embed[:, : x.size(1)])
        for block in self.blocks:
            x = block(x, c)
        return self.norm(x)


def mlp(in_dim, hidden_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim)
    )


class LeWM(nn.Module):
    """LeWorldModel: ViT encoder + projector, action encoder, action-conditioned predictor + projector."""

    def __init__(self, config: LeWMConfig):
        super().__init__()
        self.config = c = config
        self.encoder = ViTEncoder(
            c.img_size, c.patch_size, c.in_channels,
            c.encoder_dim, c.encoder_depth, c.encoder_heads, c.encoder_mlp_dim,
        )
        self.projector = mlp(c.encoder_dim, c.proj_hidden_dim, c.embed_dim)
        self.action_encoder = nn.Sequential(
            nn.Linear(c.action_dim, c.action_hidden_dim),
            nn.Linear(c.action_hidden_dim, c.action_mlp_ratio * c.embed_dim),
            nn.SiLU(),
            nn.Linear(c.action_mlp_ratio * c.embed_dim, c.embed_dim),
        )
        self.predictor = Predictor(
            c.history_size, c.embed_dim, c.predictor_depth, c.predictor_heads, c.predictor_dim_head,
            c.predictor_mlp_dim, c.predictor_dropout, c.predictor_emb_dropout,
        )
        self.pred_proj = mlp(c.embed_dim, c.proj_hidden_dim, c.embed_dim)

    def encode(self, pixels):
        """pixels: (B, T, C, H, W) -> embeddings (B, T, embed_dim)"""
        B, T = pixels.shape[:2]
        return self.projector(self.encoder(pixels.flatten(0, 1))).view(B, T, -1)

    def predict(self, emb, actions):
        """emb: (B, T, D), actions: (B, T, A) -> (B, T, D); output t predicts the embedding after action t."""
        actions = torch.nan_to_num(actions)  # stable-worldmodel datasets pad sequence boundaries with NaN
        x = self.predictor(emb, self.action_encoder(actions))
        return self.pred_proj(x.flatten(0, 1)).view_as(x)

    def forward(self, pixels, actions):
        """Training pass.

        pixels: (B, T, C, H, W) and actions: (B, T, A), with T = history_size + num_preds.
        Returns `emb` (B, T, D) for every frame and `pred` (B, T - num_preds, D), where
        pred[:, t] is the prediction of emb[:, t + num_preds]. Feed both to `LeWMLoss`.
        """
        n = self.config.num_preds
        emb = self.encode(pixels)
        return emb, self.predict(emb[:, :-n], actions[:, :-n])

    def rollout(self, pixels, actions):
        """Imagine future embeddings autoregressively (call in eval mode).

        pixels: (B, t0, C, H, W) observed frames, actions: (B, T, A) with T >= t0.
        Returns (B, T + 1, D): t0 encoded frames followed by T + 1 - t0 predicted ones.
        """
        H = self.config.history_size
        emb = self.encode(pixels)
        for t in range(emb.size(1), actions.size(1) + 1):
            lo = max(0, t - H)
            next_emb = self.predict(emb[:, lo:], actions[:, lo:t])[:, -1:]
            emb = torch.cat([emb, next_emb], dim=1)
        return emb
