# env_net_transformer_gphsplit.py
# -*- coding: utf-8 -*-
"""
Env_net with optional PRIME:
- Baseline: feature = TransformerEncoder(output)[-1]
- PRIME:    feature = PRIME(output)  (IMA aggregation + TR gated reconstruction)

Run this file directly to test both modes:
    python env_net_transformer_gphsplit.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1) PRIME Components
# ==========================================
class IntensityMemoryAttention(nn.Module):
    """
    Input-conditioned temporal aggregation.
    Input : temporal_features [T, B, D]
    Output: embedded [B, D], alpha [B, T]
    """
    def __init__(self, feature_dim: int, hidden_dim: int = 128, temperature: float = 1.0):
        super().__init__()
        self.temperature = float(temperature)
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, temporal_features: torch.Tensor):
        # [T,B,D] -> [B,T,D]
        x = temporal_features.permute(1, 0, 2)
        scores = self.scorer(x).squeeze(-1)  # [B,T]
        alpha = F.softmax(scores / self.temperature, dim=1)  # [B,T]
        embedded = torch.sum(x * alpha.unsqueeze(-1), dim=1)  # [B,D]
        return embedded, alpha


class TakensReconstructor(nn.Module):
    """
    Safe reconstructor with gated residual.
    Input : x [B, D]
    Output: s [B, D]
    """
    def __init__(self, feature_dim: int, hidden_dim: int = 256, gate_bias: float = -3.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.gate = nn.Linear(feature_dim * 2, feature_dim)
        nn.init.constant_(self.gate.bias, float(gate_bias))

    def forward(self, x: torch.Tensor):
        r = self.mlp(x)  # [B,D]
        g = torch.sigmoid(self.gate(torch.cat([x, r], dim=-1)))  # [B,D]
        s = x + g * (r - x)
        return s


class PRIME(nn.Module):
    """
    PRIME = IMA (temporal aggregation) + TR (safe reconstruction)
    Input : temporal_features [T, B, D]
    Output: reconstructed [B, D], embedded [B, D], alpha [B, T]
    """
    def __init__(self, feature_dim: int, hidden_attn: int = 128, hidden_rec: int = 256):
        super().__init__()
        self.ima = IntensityMemoryAttention(feature_dim=feature_dim, hidden_dim=hidden_attn)
        self.tr = TakensReconstructor(feature_dim=feature_dim, hidden_dim=hidden_rec)

    def forward(self, temporal_features: torch.Tensor):
        embedded, alpha = self.ima(temporal_features)    # [B,D], [B,T]
        reconstructed = self.tr(embedded)                # [B,D]
        return reconstructed, embedded, alpha


# ==========================================
# 2) Env_net
# ==========================================
class Env_net(nn.Module):
    """
    env_data: dict of tensors, each [B, T, dim_k]
    gph:      [B, 1, T, H, W]  (or any shape that can be permuted into [B, T, 1, H, W])

    returns:
        feature:        [B, 64]   (baseline=last step; prime=reconstructed)
        output:         [T, B, 64] (Transformer encoder output)
        embedded_state: [B, 64] or None (PRIME embedded_state; baseline None)
    """
    def __init__(
        self,
        obs_len: int = 8,
        use_prime: bool = False,
        prime_hidden_attn: int = 128,
        prime_hidden_rec: int = 256,
        dim_f6: int = 6,
        dim_f10: int = 10,
    ):
        super().__init__()
        self.obs_len = int(obs_len)
        self.use_prime = bool(use_prime)
        self.last_alpha = None

        embed_dim = 16
        self.data_embed = nn.ModuleDict()

        # -------- original keys --------
        self.data_embed['wind'] = nn.Linear(1, embed_dim)
        self.data_embed['intensity_class'] = nn.Linear(6, embed_dim)
        self.data_embed['move_velocity'] = nn.Linear(1, embed_dim)
        self.data_embed['month'] = nn.Linear(12, embed_dim)
        self.data_embed['location_long'] = nn.Linear(36, embed_dim)
        self.data_embed['location_lat'] = nn.Linear(12, embed_dim)
        self.data_embed['history_direction12'] = nn.Linear(8, embed_dim)
        self.data_embed['history_direction24'] = nn.Linear(8, embed_dim)
        self.data_embed['history_inte_change24'] = nn.Linear(4, embed_dim)

        # -------- new keys --------
        self.data_embed['feature6'] = nn.Linear(int(dim_f6), embed_dim)
        self.data_embed['feature10'] = nn.Linear(int(dim_f10), embed_dim)

        # GPH embedding (per time step)
        self.GPH_embed = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(8, 8)  # 64x64 -> 8x8 if input is 64x64
        )

        # env_f_in = (#keys * embed_dim) + (gph_flat = 8*8)
        env_f_in = len(self.data_embed) * embed_dim + 8 * 8

        self.evn_extract = nn.Sequential(
            nn.Linear(env_f_in, env_f_in // 2),
            nn.ReLU(inplace=True),
            nn.Linear(env_f_in // 2, env_f_in // 2),
            nn.ReLU(inplace=True),
            nn.Linear(env_f_in // 2, 64)
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        if self.use_prime:
            self.prime = PRIME(
                feature_dim=64,
                hidden_attn=int(prime_hidden_attn),
                hidden_rec=int(prime_hidden_rec),
            )

        self.init_weights()

    def init_weights(self):
        def init_kaiming(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in")
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
        self.apply(init_kaiming)

    def forward(self, env_data: dict, gph: torch.Tensor):
        # ---- gph shape normalize: [B, 1, T, H, W] -> [B, T, 1, H, W]
        if gph.dim() != 5:
            raise ValueError(f"gph must be 5D [B,1,T,H,W] (or permutable), got {tuple(gph.shape)}")
        # assume input is [B,1,T,H,W] and permute to [B,T,1,H,W]
        gph = gph.permute(0, 2, 1, 3, 4)  # [B,T,1,H,W]

        B, T, C, H, W = gph.shape
        if T != self.obs_len:
            # not fatal, but warn-like assertion for debugging
            # You can remove this if your pipeline uses other T.
            pass

        # ---- env feature embeddings
        embed_list = []
        for key in self.data_embed:
            if key not in env_data:
                raise KeyError(f"env_data missing key '{key}'. Provided keys: {list(env_data.keys())}")
            xk = env_data[key]
            if xk.dim() != 3:
                raise ValueError(f"env_data['{key}'] must be [B,T,dim], got {tuple(xk.shape)}")
            if xk.shape[0] != B or xk.shape[1] != T:
                raise ValueError(
                    f"env_data['{key}'] shape mismatch. Expected [B={B},T={T},*], got {tuple(xk.shape)}"
                )
            embed_list.append(self.data_embed[key](xk))  # [B,T,16]

        # ---- gph per-step embedding
        gph_list = []
        for t in range(T):
            # gph[:,t] -> [B,1,H,W]
            gph_feat_t = self.GPH_embed(gph[:, t]).reshape(B, -1)  # [B,64] for 8*8
            gph_list.append(gph_feat_t)
        gph_feature = torch.stack(gph_list, dim=1)  # [B,T,64]
        embed_list.append(gph_feature)

        # ---- concat -> [B,T,env_f_in]
        embed = torch.cat(embed_list, dim=2)

        # ---- MLP extract -> Transformer encoder
        feature_in = self.evn_extract(embed).permute(1, 0, 2)  # [T,B,64]
        output = self.encoder(feature_in)                      # [T,B,64]

        # ---- PRIME / baseline
        if self.use_prime:
            feature, embedded_state, alpha = self.prime(output)  # feature: [B,64]
            self.last_alpha = alpha
        else:
            feature = output[-1]     # [B,64]
            embedded_state = None
            self.last_alpha = None

        return feature, output, embedded_state
