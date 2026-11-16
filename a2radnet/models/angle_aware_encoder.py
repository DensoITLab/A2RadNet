"""Angle-Aware Encoder (AAE) and Doppler Latent Reducer (DLR).

Channel-first input [B, 2C, R, D]; RPA output [B, L, R, Ap].
State-dict attribute names remain compatible with the research checkpoints.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def cyclic_padding(x: torch.Tensor, shift_bins: int) -> torch.Tensor:
    """
    Args:
        x: [B, C, R, D]
        shift_bins: int, how many bins to shift (positive means right shift)

    Returns:
        x_padded: [B, C, R, D + shift_bins]
    """
    if shift_bins == 0:
        return x
    elif shift_bins > 0:
        pad = x[:, :, :, -shift_bins:]
        x_padded = torch.cat([pad, x], dim=-1)
        return x_padded
    else:
        raise ValueError('Negative shift not supported yet')

class DilatedCircularProjection(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, dilation: int, dropout: float=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cyclic_padding(x, shift_bins=self.conv.dilation[1] * (self.conv.kernel_size[1] - 1))
        x = self.conv(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

class PointwiseProjection(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, dropout: float=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

class AngleProjection(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, doppler_bins: int, doppler_period: int, cyclic_shift: bool=True):
        super().__init__()
        num_tx_like = doppler_bins // doppler_period
        if cyclic_shift:
            self.input_proj = DilatedCircularProjection(in_dim=in_dim, out_dim=out_dim, kernel_size=(1, num_tx_like), dilation=(1, doppler_period))
        else:
            self.input_proj = PointwiseProjection(in_dim=in_dim, out_dim=out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, C, R, D) = x.shape
        x = self.input_proj(x)
        return x.permute(0, 2, 3, 1)

class DopplerLatentReducer(nn.Module):
    """
    Input:
        [B, R, D, C]
        C: angle feature dimension per hypothesis
        D: Doppler dimension (hypothesis dimension)

    Output:
        [B, R, M, C]
        M: number of latents to reduce to
    """

    def __init__(self, feat_dim: int, num_latents: int, mode: str='attn'):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_latents = num_latents
        self.mode = mode
        self.reducer_attn_dim = feat_dim
        if mode == 'attn':
            self.latents = nn.Parameter(torch.randn(1, 1, self.num_latents, self.feat_dim) * 0.02)
            self.q = nn.Linear(self.feat_dim, self.reducer_attn_dim)
            self.k = nn.Linear(self.feat_dim, self.reducer_attn_dim)
            self.v = nn.Linear(self.feat_dim, self.feat_dim)
            self.out = nn.Linear(self.feat_dim, self.feat_dim)
        else:
            raise ValueError(f'Unknown mode={mode}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, R, D, C) = x.shape
        kv = x.reshape(B * R, D, C)
        q = self.q(self.latents.expand(B, R, -1, -1).reshape(B * R, self.num_latents, self.reducer_attn_dim))
        k = self.k(kv)
        v = self.v(kv)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.reducer_attn_dim ** 0.5
        attn = F.softmax(attn, dim=-1)
        y = torch.matmul(attn, v)
        y = self.out(y)
        return y.view(B, R, self.num_latents, C)

class AngleAwareEncoder(nn.Module):
    """
    Explicit MIMO-aware MLP frontend that operates on the raw antenna channels and Doppler bins with a built-in inductive bias for the periodic DDM structure.

    Input:
        x: [B, 32, R, D]
           channel order = [real_16, imag_16]

    Output:
        y: [B, out_channels, R, D]

    Main idea:
        1) split channels into [16 antennas, 2 complex]

    Notes:
        - single-head linear attention
        - no explicit anchor-wise KV expansion
        - relative periodic structure is induced by circular RoPE
    """

    def __init__(self, doppler_bins=256, doppler_period=16, num_ant=16, complex_dim=2, embed_dim=32, out_channels=32, num_latents=8, num_blocks=2, dropout=0.1, doppler_reducer=False, output_order='rd', angle_estimator_type=1, use_cyclic_shift=True):
        super().__init__()
        if angle_estimator_type != 1 or doppler_reducer not in (0, 1):
            raise ValueError('Only the paper AAE projection and DLR variants are supported.')
        if output_order not in ('ra', 'rd'):
            raise ValueError('output_order must be ra or rd')
        self.doppler_bins = doppler_bins
        self.doppler_period = doppler_period
        self.num_tx_like = doppler_bins // doppler_period
        self.num_ant = num_ant
        self.complex_dim = complex_dim
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.num_latents = num_latents
        self.use_cyclic_shift = use_cyclic_shift
        self.angle_estimator = AngleProjection(in_dim=self.num_ant * self.complex_dim, out_dim=self.out_channels, doppler_bins=doppler_bins, doppler_period=doppler_period, cyclic_shift=use_cyclic_shift)
        self.reducer = None
        if doppler_reducer == 1:
            self.reducer = DopplerLatentReducer(feat_dim=self.out_channels, num_latents=num_latents)
        self.output_order = output_order

    def forward(self, x):
        """
        x: [B, 32, R, D]
        returns: [B, out_channels, R, D]
        """
        (B, C, R, D) = x.shape
        assert C == self.num_ant * self.complex_dim, f'Expected {self.num_ant * self.complex_dim} channels, got {C}'
        P = self.doppler_period
        assert D % P == 0, f'D={D} must be divisible by doppler_period={P}'
        T = D // P
        assert T == self.num_tx_like, f'Expected num_tx_like={self.num_tx_like}, got T={T}'
        x = self.angle_estimator(x)
        if self.reducer is not None:
            x = self.reducer(x)
        if self.output_order == 'rd':
            return x.permute(0, 3, 1, 2)
        elif self.output_order == 'ra':
            return x.permute(0, 2, 1, 3)
        else:
            raise ValueError(f'Unknown output_order={self.output_order}')
