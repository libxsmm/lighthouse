# RUN: %PYTHON %S/kernel_bench.py --input %s --dump-kernel imported | FileCheck %s --check-prefix=IMPORT
# RUN: %PYTHON %S/kernel_bench.py --input %s --dump-kernel xegpu-wg | FileCheck %s --check-prefix=WG --implicit-check-not=memref.copy --implicit-check-not=memref.alloc
# REQUIRES: torch, torch_mlir

# IMPORT: math.rsqrt
# IMPORT: linalg.matmul
# IMPORT: tensor.concat
# WG: gpu.alloc {{.*}}memref<4x512x2048xbf16>
# WG-NOT: gpu.alloc
# WG: gpu.alloc {{.*}}memref<2048x512xbf16>
# WG-NOT: gpu.alloc
# WG: gpu.func @main_kernel
# WG: vector.multi_reduction
# WG: math.rsqrt
# WG-NOT: xegpu.dpas
# WG: gpu.func @main_kernel
# WG: xegpu.dpas
# WG-NOT: vector.multi_reduction
# WG: gpu.func @main_kernel
# WG: arith.negf
# WG-NOT: gpu.func

"""Llama 3.2 1B input RMSNorm, K projection, and RoPE.

RMSNorm is shared by Q/K/V in a full decoder. This single-output stepping
stone retains only K and uses separate normalization, projection, and RoPE
kernels. Variance and normalization arithmetic use fp32, as in the decoder.
"""

import torch
import torch.nn as nn

from llama_rope import get_rope_tables, rotate_half


batch_size = 4
n_ctx = 512
dim = 2048
num_kv_heads = 8
d_head = 64


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(dim))
        self.k_proj = nn.Linear(dim, num_kv_heads * d_head, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        batch, sequence, _ = x.shape
        h = x.float()
        variance = h.pow(2).mean(dim=-1, keepdim=True)
        h = self.norm_weight * (h * torch.rsqrt(variance + 1e-5)).to(x.dtype)
        k = self.k_proj(h).view(batch, sequence, num_kv_heads, d_head).transpose(1, 2)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        return k * cos + rotate_half(k) * sin


def get_inputs():
    x = torch.randn(batch_size, n_ctx, dim)
    cos, sin = get_rope_tables(torch.arange(n_ctx))
    return [x, cos, sin]


def get_init_inputs():
    return []
