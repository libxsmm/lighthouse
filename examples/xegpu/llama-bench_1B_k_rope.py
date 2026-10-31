# RUN: %PYTHON %S/kernel_bench.py --input %s --input-shapes 1x8x2048 8x64 8x64 --dump-kernel imported | FileCheck %s
# RUN: %PYTHON %S/kernel_bench.py --input %s --dump-kernel tiled | FileCheck %s --check-prefix=TILED --implicit-check-not=tensor.concat
# RUN: %PYTHON %S/kernel_bench.py --input %s --dump-kernel vectorized | FileCheck %s --check-prefix=VECTOR --implicit-check-not=tensor.concat --implicit-check-not=vector.extract_strided_slice --implicit-check-not=vector.insert_strided_slice
# RUN: %PYTHON %S/kernel_bench.py --input %s --dump-kernel bufferized | FileCheck %s --check-prefix=BUFFER --implicit-check-not=memref.alloc --implicit-check-not=memref.copy
# RUN: %PYTHON %S/kernel_bench.py --input %s --dump-kernel gpu-outlining | FileCheck %s --check-prefix=OUTLINE
# RUN: %PYTHON %S/kernel_bench.py --input %s --dump-kernel xegpu-initial | FileCheck %s --check-prefix=XEGPU --implicit-check-not=vector.extract_strided_slice --implicit-check-not=vector.insert_strided_slice --implicit-check-not=memref.copy
# RUN: %PYTHON %S/kernel_bench.py --input %s --dump-kernel xegpu-wg | FileCheck %s --check-prefix=WG --implicit-check-not=vector.extract_strided_slice --implicit-check-not=vector.insert_strided_slice --implicit-check-not=memref.copy
# RUN: %PYTHON %S/kernel_bench.py --input %s --identity-weights --power-of-two-inputs 0 --dump-kernel xegpu-wg | FileCheck %s --check-prefix=WG --implicit-check-not=vector.extract_strided_slice --implicit-check-not=vector.insert_strided_slice --implicit-check-not=memref.copy
# REQUIRES: torch, torch_mlir

# CHECK-LABEL: func.func private @main(
# CHECK: linalg.matmul
# CHECK-NOT: linalg.matmul
# CHECK: tensor.extract_slice
# CHECK: arith.negf
# CHECK: arith.mulf
# CHECK: arith.addf
# CHECK: return

# The K projection and RoPE are tiled into separate workgroup loops (kernels).
# TILED-LABEL: func.func private @main(
# TILED: scf.forall {{.*}} -> (tensor<2048x512xbf16>)
# TILED: scf.for
# TILED: linalg.matmul
# TILED: arith.truncf
# TILED: scf.forall {{.*}} -> (tensor<4x8x512x64xbf16>)
# TILED-NOT: linalg.matmul
# TILED: tensor<128x32xbf16>
# TILED: arith.negf
# TILED: arith.addf
# TILED-COUNT-2: tensor.insert_slice
# TILED: return

# VECTOR-LABEL: func.func private @main(
# VECTOR: vector.transfer_read {{.*}}tensor<4x512x2048xbf16>, vector<128x16xbf16>
# VECTOR-NEXT: {{.*}}vector.contract
# VECTOR: arith.truncf
# VECTOR: scf.forall {{.*}} -> (tensor<4x8x512x64xbf16>)
# VECTOR: vector.transfer_read {{.*}}tensor<4x512x8x64xbf16>, vector<128x32xbf16>
# VECTOR-NOT: vector.shape_cast
# VECTOR-COUNT-2: vector.transfer_write {{.*}}vector<128x32xbf16>
# VECTOR-NOT: vector.transfer_write
# VECTOR: return

# Only the projection result passed between kernels is allocated.
# BUFFER-LABEL: func.func private @main(
# BUFFER: gpu.alloc () : memref<2048x512xbf16>
# BUFFER-NOT: gpu.alloc
# BUFFER-COUNT-2: scf.forall
# BUFFER: return

# OUTLINE: gpu.func @main_kernel{{.*}}known_block_size = array<i32: 512, 1, 1>
# OUTLINE: gpu.func @main_kernel{{.*}}known_block_size = array<i32: 16, 1, 1>
# OUTLINE-NOT: gpu.func

# XEGPU: gpu.alloc
# XEGPU-NOT: gpu.alloc
# XEGPU: xegpu.dpas
# XEGPU: gpu.func @main_kernel{{.*}}memref<4x8x512x64xbf16>
# XEGPU-NOT: gpu.alloc
# XEGPU-COUNT-2: xegpu.store_nd {{.*}}vector<128x32xbf16>
# XEGPU-NOT: gpu.alloc

# WG: xegpu.dpas
# WG-LABEL: gpu.func @main_kernel{{.*}}memref<4x8x512x64xbf16>
# WG-NOT: gpu.alloc
# WG: xegpu.load_nd {{.*}}layout = #xegpu.layout<sg_layout = [1, 1], sg_data = [128, 32]>
# WG: xegpu.load_nd {{.*}}layout = #xegpu.layout<sg_layout = [1, 1], sg_data = [128, 32]>
# WG: xegpu.load_nd {{.*}}layout = #xegpu.layout<sg_layout = [1, 1], sg_data = [128, 32]>
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
num_kv_heads = 8
d_head = 64


class Model(nn.Module):
    """Llama 3.2 1B K projection followed by a separate RoPE kernel.

    The input x is already normalized: [batch, sequence, dim]. Precomputed
    cos/sin tables [sequence, d_head] are shared across the batch. Returns
    rotated K [batch, num_kv_heads, sequence, d_head]. Q, V, RMSNorm, and
    attention are excluded to keep a single result tensor.
    """

    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(dim, num_kv_heads * d_head, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        batch, sequence, _ = x.shape
        k = self.k_proj(x).view(batch, sequence, num_kv_heads, d_head).transpose(1, 2)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        return k * cos + rotate_half(k) * sin


def get_inputs():
    x = torch.randn(batch_size, n_ctx, dim)
    cos, sin = get_rope_tables(torch.arange(n_ctx))
    return [x, cos, sin]


def get_init_inputs():
    return []
