from dataclasses import dataclass


@dataclass
class LeWMConfig:
    """All LeWM model hyperparameters. Defaults reproduce the paper (ViT-tiny encoder, ~18M params)."""

    # size of one action step; with frameskip, stack the skipped actions: frameskip * raw_action_dim
    action_dim: int

    # sequences: the model sees T = history_size + num_preds frames per sample
    history_size: int = 3  # context frames fed to the predictor
    num_preds: int = 1  # predict the embedding this many steps ahead

    # ViT encoder
    img_size: int = 224
    patch_size: int = 14
    in_channels: int = 3
    encoder_dim: int = 192
    encoder_depth: int = 12
    encoder_heads: int = 3
    encoder_mlp_dim: int = 768

    # latent space: projector output, predictor width and action embedding size
    embed_dim: int = 192
    proj_hidden_dim: int = 2048  # hidden size of the projector / pred_proj MLPs

    # action encoder
    action_hidden_dim: int = 10
    action_mlp_ratio: int = 4

    # predictor: causal transformer conditioned on actions via AdaLN-zero
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_dim_head: int = 64
    predictor_mlp_dim: int = 2048
    predictor_dropout: float = 0.1
    predictor_emb_dropout: float = 0.0
