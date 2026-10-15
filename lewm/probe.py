import torch
from torch import nn


class CrossAttentionBlock(nn.Module):
    """Pre-norm block: queries cross-attend to a context, then a residual MLP. No self-attention."""

    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))

    def forward(self, x, ctx):
        x = x + self.attn(self.norm1(x), ctx, ctx, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class VisualProbe(nn.Module):
    """Decodes an image from one embedding (LeWM App. D): learnable per-patch queries cross-attend to it.

    A diagnostic, not part of LeWM's objective: feed it detached embeddings so no gradient reaches the model.
    """

    def __init__(self, in_dim=192, img_size=224, patch_size=16, out_channels=3, dim=256, depth=4, heads=8, mlp_dim=1024):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.grid = img_size // patch_size
        self.in_proj = nn.Linear(in_dim, dim)
        self.queries = nn.Parameter(torch.zeros(1, self.grid**2, dim))
        self.blocks = nn.ModuleList(CrossAttentionBlock(dim, heads, mlp_dim) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.to_pixels = nn.Linear(dim, patch_size**2 * out_channels)
        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(self, z):
        """z: (..., in_dim) -> images (..., out_channels, img_size, img_size)"""
        lead, z = z.shape[:-1], z.reshape(-1, z.size(-1))
        ctx = self.in_proj(z).unsqueeze(1)  # (N, 1, dim): the embedding is the single key/value token
        x = self.queries.expand(z.size(0), -1, -1)
        for block in self.blocks:
            x = block(x, ctx)
        g, p, C = self.grid, self.patch_size, self.out_channels
        x = self.to_pixels(self.norm(x)).view(-1, g, g, p, p, C)  # one (p, p, C) pixel patch per query
        return x.permute(0, 5, 1, 3, 2, 4).reshape(*lead, C, g * p, g * p)
