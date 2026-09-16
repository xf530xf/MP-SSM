"""Six spatial permutations and the selective recurrence in paper Eqs. 7-9.

No convolution or attention substitute is used by the reference backend. It is
intended for small correctness tests; use the official CUDA scan for training.
"""

from functools import lru_cache
import math

import torch
from torch import nn
from torch.nn import functional as F


@lru_cache(maxsize=16)
@torch.inference_mode(False)
def scan_indices(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CPU gather and inverse indices, each of shape [6, H*W].

    Image coordinates have rows increasing downwards. NE: increasing r+c,
    decreasing r within each diagonal. NW: increasing c-r, decreasing r
    within each diagonal. Diagonals are concatenated without resetting state.
    Every route visits each pixel exactly once, including rectangular grids.
    """
    if height < 1 or width < 1:
        raise ValueError("Spatial dimensions must be positive")
    raster = list(range(height * width))
    columns = [r * width + c for c in range(width) for r in range(height)]
    northeast, northwest = [], []
    for diagonal in range(height + width - 1):
        for r in range(min(height - 1, diagonal), max(-1, diagonal - width), -1):
            northeast.append(r * width + diagonal - r)
    for diagonal in range(-(height - 1), width):
        for r in range(min(height - 1, width - 1 - diagonal), max(-1, -diagonal - 1), -1):
            northwest.append(r * width + r + diagonal)
    indices = torch.tensor([raster, raster[::-1], columns, columns[::-1], northeast, northwest])
    return indices, torch.argsort(indices, dim=-1)


def selective_scan_reference(u, delta, A, B, C, D, delta_bias=None):
    """Differentiable FP32 recurrence; same grouped layout as Mamba's CUDA API.

    u/delta: [batch, groups*channels, length]; A: [groups*channels, state];
    B/C: [batch, groups, state, length]; D/bias: [groups*channels].
    delta is converted with softplus, matching delta_softplus=True on CUDA.
    """
    dtype = u.dtype
    batch, total_channels, length = u.shape
    groups, state_size = B.shape[1:3]
    if length < 1 or total_channels % groups:
        raise ValueError("Invalid scan sequence or channel grouping")
    channels = total_channels // groups
    u = u.float().reshape(batch, groups, channels, length)
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float().view(1, -1, 1)
    delta = F.softplus(delta).reshape(batch, groups, channels, length)
    A = A.float().reshape(groups, channels, state_size)
    B, C = B.float(), C.float()
    state = torch.zeros(batch, groups, channels, state_size, device=u.device, dtype=torch.float32)
    outputs = []
    for t in range(length):
        dt = delta[..., t].unsqueeze(-1)
        state = torch.exp(dt * A.unsqueeze(0)) * state + dt * B[..., t].unsqueeze(2) * u[..., t].unsqueeze(-1)
        outputs.append((state * C[..., t].unsqueeze(2)).sum(-1))
    y = torch.stack(outputs, dim=-1) + D.float().reshape(1, groups, channels, 1) * u
    return y.reshape(batch, total_channels, length).to(dtype)


@lru_cache(maxsize=1)
def require_cuda_scan():
    try:
        import selective_scan_cuda  # noqa: F401 -- verify the compiled extension itself
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "CUDA selective scan is unavailable. Install a CUDA-compatible PyTorch and "
            "mamba-ssm compiled extension (see README.md). For current source builds: "
            "MAMBA_KEEP_CUDA_BUILD=TRUE pip install mamba-ssm --no-build-isolation. "
            "Only use scan_backend=reference for small correctness tests."
        ) from exc
    return selective_scan_fn


class SelectiveScan2D(nn.Module):
    def __init__(self, channels: int, d_state: int = 16, expand: int = 2, backend: str = "cuda"):
        super().__init__()
        if backend not in {"cuda", "reference"}:
            raise ValueError("scan_backend must be cuda or reference")
        if min(channels, d_state, expand) < 1:
            raise ValueError("Scan dimensions must be positive")
        self.backend, self.channels, self.d_state = backend, channels, d_state
        self.inner = channels * expand
        self.rank = math.ceil(channels / 16)
        self.in_proj = nn.Conv2d(channels, self.inner, 1, bias=False)
        self.out_proj = nn.Conv2d(self.inner, channels, 1, bias=False)
        self.x_proj = nn.Parameter(torch.empty(6, self.rank + 2 * d_state, self.inner))
        self.dt_proj = nn.Parameter(torch.empty(6, self.inner, self.rank))
        self.dt_bias = nn.Parameter(torch.empty(6, self.inner))
        self.A_log = nn.Parameter(torch.arange(1, d_state + 1).float().log().repeat(6, self.inner, 1))
        self.D = nn.Parameter(torch.ones(6, self.inner))
        for weight in self.x_proj:
            nn.init.xavier_uniform_(weight)
        nn.init.uniform_(self.dt_proj, -self.rank ** -0.5, self.rank ** -0.5)
        # Inverse softplus produces positive initial steps in [0.001, 0.1].
        dt = torch.exp(torch.rand(6, self.inner) * math.log(100) + math.log(0.001))
        with torch.no_grad():
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        # Device cache is bounded and is not serialized in checkpoints.
        self._index_key = None
        self._indices = None

    def forward(self, x):
        batch, _, height, width = x.shape
        key = (height, width, x.device)
        if self._index_key != key:
            # Indices may be first requested during validation or benchmarking.
            # Keep cached tensors ordinary tensors so a later training forward
            # can save them for index_select's backward (including CUDA copies).
            with torch.inference_mode(False):
                self._indices = tuple(t.to(x.device) for t in scan_indices(height, width))
            self._index_key = key
        indices, inverse = self._indices
        projected = self.in_proj(x)
        flat = projected.flatten(2)
        sequences = torch.stack([flat.index_select(-1, route) for route in indices], dim=1)
        params = torch.einsum("bkdl,kod->bkol", sequences, self.x_proj)
        dt, B, C = torch.split(params, [self.rank, self.d_state, self.d_state], dim=2)
        dt = torch.einsum("bkrl,kdr->bkdl", dt, self.dt_proj)
        u = sequences.reshape(batch, 6 * self.inner, -1).contiguous()
        dt = dt.reshape_as(u).contiguous()
        A, D = -self.A_log.float().exp().flatten(0, 1), self.D.float().flatten()
        if self.backend == "cuda":
            if not x.is_cuda:
                raise RuntimeError("scan_backend=cuda requires a CUDA tensor; use reference for a CPU smoke test")
            fn = require_cuda_scan()
            y = fn(u, dt, A, B.contiguous(), C.contiguous(), D,
                   delta_bias=self.dt_bias.float().flatten(), delta_softplus=True)
        else:
            y = selective_scan_reference(u, dt, A, B, C, D, self.dt_bias.flatten())
        y = y.reshape(batch, 6, self.inner, height * width)
        merged = sum(y[:, r].index_select(-1, inverse[r]) for r in range(6))
        return self.out_proj(merged.reshape(batch, self.inner, height, width).to(projected.dtype))
