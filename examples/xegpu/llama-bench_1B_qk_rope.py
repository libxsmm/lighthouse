# RUN: %PYTHON %S/kernel_bench.py --input %s --input-shapes 1x8x2048 8x64 8x64 --dump-kernel imported | FileCheck %s
# RUN: %PYTHON %S/kernel_bench.py --input %s --dump-kernel tiled | FileCheck %s --check-prefix=TILED --implicit-check-not=tensor.concat
# RUN: %PYTHON %S/kernel_bench.py --input %s --dump-kernel vectorized | FileCheck %s --check-prefix=VECTOR --implicit-check-not=tensor.concat --implicit-check-not=vector.extract_strided_slice --implicit-check-not=vector.insert_strided_slice
# RUN: %PYTHON %S/kernel_bench.py --input %s --dump-kernel bufferized | FileCheck %s --check-prefix=BUFFER --implicit-check-not=memref.alloc --implicit-check-not=memref.copy
# RUN: %PYTHON %S/kernel_bench.py --input %s --dump-kernel gpu-outlining | FileCheck %s --check-prefix=OUTLINE
# RUN: %PYTHON %S/kernel_bench.py --input %s --dump-kernel xegpu-initial | FileCheck %s --check-prefix=XEGPU --implicit-check-not=vector.extract_strided_slice --implicit-check-not=vector.insert_strided_slice --implicit-check-not=memref.copy
# RUN: %PYTHON %S/kernel_bench.py --input %s --dump-kernel xegpu-wg | FileCheck %s --check-prefix=WG --implicit-check-not=vector.extract_strided_slice --implicit-check-not=vector.insert_strided_slice --implicit-check-not=memref.copy
# REQUIRES: torch, torch_mlir

# CHECK-LABEL: func.func private @main(
# CHECK-COUNT-2: linalg.matmul
# CHECK: tensor.extract_slice
# CHECK: arith.negf
# CHECK: arith.mulf
# CHECK: arith.addf
# CHECK: return

# Projections and RoPE are tiled into separate workgroup loops (kernels).
# TILED-LABEL: func.func private @main(
# TILED: scf.forall {{.*}} -> (tensor<2048x2048xbf16>)
# TILED: scf.for
# TILED: linalg.matmul
# TILED: arith.truncf
# TILED: scf.forall {{.*}} -> (tensor<2048x512xbf16>)
# TILED: scf.for
# TILED: linalg.matmul
# TILED: arith.truncf
# TILED: scf.forall {{.*}} -> (tensor<4x32x512x64xbf16>)
# TILED-NOT: linalg.matmul
# TILED: linalg.transpose
# TILED: arith.negf
# TILED: arith.addf
# TILED-COUNT-2: tensor.insert_slice
# TILED: scf.forall {{.*}} -> (tensor<4x8x512x64xbf16>)
# TILED-NOT: linalg.matmul
# TILED: linalg.transpose
# TILED: arith.negf
# TILED: arith.addf
# TILED-COUNT-2: tensor.insert_slice
# TILED: return

# VECTOR-LABEL: func.func private @main(
# VECTOR-COUNT-2: arith.truncf
# VECTOR: scf.forall {{.*}} -> (tensor<4x32x512x64xbf16>)
# VECTOR-COUNT-2: vector.transfer_write {{.*}}vector<128x32xbf16>
# VECTOR-NOT: vector.transfer_write
# VECTOR: scf.forall {{.*}} -> (tensor<4x8x512x64xbf16>)
# VECTOR-COUNT-2: vector.transfer_write {{.*}}vector<128x32xbf16>
# VECTOR-NOT: vector.transfer_write
# VECTOR: return

# Only the projection results passed between kernels are allocated.
# BUFFER-LABEL: func.func private @main(
# BUFFER: gpu.alloc () : memref<2048x2048xbf16>
# BUFFER: scf.forall
# BUFFER: gpu.alloc () : memref<2048x512xbf16>
# BUFFER-COUNT-3: scf.forall
# BUFFER: return

# OUTLINE-COUNT-2: gpu.func @main_kernel{{.*}}known_block_size = array<i32: 512, 1, 1>
# OUTLINE-COUNT-2: gpu.func @main_kernel{{.*}}known_block_size = array<i32: 16, 1, 1>

# XEGPU-COUNT-2: gpu.alloc
# XEGPU-NOT: gpu.alloc
# XEGPU-COUNT-2: xegpu.dpas
# XEGPU: gpu.func @main_kernel{{.*}}memref<4x32x512x64xbf16>
# XEGPU-NOT: gpu.alloc
# XEGPU-COUNT-2: xegpu.store_nd {{.*}}vector<128x32xbf16>
# XEGPU: gpu.func @main_kernel{{.*}}memref<4x8x512x64xbf16>
# XEGPU-NOT: gpu.alloc
# XEGPU-COUNT-2: xegpu.store_nd {{.*}}vector<128x32xbf16>
# XEGPU-NOT: gpu.alloc

# WG-COUNT-2: xegpu.dpas
# WG-LABEL: gpu.func @main_kernel{{.*}}memref<4x32x512x64xbf16>
# WG-NOT: gpu.alloc
# WG: xegpu.load_nd {{.*}}layout = #xegpu.layout<sg_layout = [1, 1, 1], sg_data = [128, 1, 32]>
# WG: xegpu.load_nd {{.*}}layout = #xegpu.layout<sg_layout = [1, 1], sg_data = [128, 32]>
# WG: xegpu.load_nd {{.*}}layout = #xegpu.layout<sg_layout = [1, 1, 1], sg_data = [128, 1, 32]>
# WG: xegpu.load_nd {{.*}}layout = #xegpu.layout<sg_layout = [1, 1], sg_data = [128, 32]>
# WG: xegpu.load_nd {{.*}}layout = #xegpu.layout<sg_layout = [1, 1], sg_data = [128, 32]>
# WG: xegpu.load_nd {{.*}}layout = #xegpu.layout<sg_layout = [1, 1], sg_data = [128, 32]>
# WG-COUNT-2: xegpu.store_nd {{.*}}layout = #xegpu.layout<sg_layout = [1, 1], sg_data = [128, 32]>
# WG-LABEL: gpu.func @main_kernel{{.*}}memref<4x8x512x64xbf16>
# WG-NOT: gpu.alloc
# WG: xegpu.load_nd {{.*}}layout = #xegpu.layout<sg_layout = [1, 1, 1], sg_data = [128, 1, 32]>
# WG: xegpu.load_nd {{.*}}layout = #xegpu.layout<sg_layout = [1, 1], sg_data = [128, 32]>
# WG: xegpu.load_nd {{.*}}layout = #xegpu.layout<sg_layout = [1, 1, 1], sg_data = [128, 1, 32]>
# WG: xegpu.load_nd {{.*}}layout = #xegpu.layout<sg_layout = [1, 1], sg_data = [128, 32]>
# WG: xegpu.load_nd {{.*}}layout = #xegpu.layout<sg_layout = [1, 1], sg_data = [128, 32]>
# WG: xegpu.load_nd {{.*}}layout = #xegpu.layout<sg_layout = [1, 1], sg_data = [128, 32]>
# WG-COUNT-2: xegpu.store_nd {{.*}}layout = #xegpu.layout<sg_layout = [1, 1], sg_data = [128, 32]>
# WG-NOT: gpu.alloc

import torch
import torch.nn as nn

from llama_rope import get_rope_tables, rotate_half


batch_size = 4
n_ctx = 512
dim = 2048
num_heads = 32
num_kv_heads = 8
d_head = 64


class Model(nn.Module):
    """Llama 3.2 1B Q/K projections followed by RoPE, for epilogue fusion.

    The input x is already normalized: [batch, sequence, dim]. Precomputed
    cos/sin tables [sequence, d_head] are shared across the batch. Returns
    rotated Q [batch, num_heads, sequence, d_head] and K [batch, num_kv_heads,
    sequence, d_head]. V, RMSNorm, and attention are excluded.
    """

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(dim, num_heads * d_head, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * d_head, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, sequence, _ = x.shape
        q = self.q_proj(x).view(batch, sequence, num_heads, d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch, sequence, num_kv_heads, d_head).transpose(1, 2)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def get_inputs():
    x = torch.randn(batch_size, n_ctx, dim)
    cos, sin = get_rope_tables(torch.arange(n_ctx))
    return [x, cos, sin]


def get_init_inputs():
    return []
