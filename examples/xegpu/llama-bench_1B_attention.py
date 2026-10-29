# RUN: %PYTHON %S/kernel_bench.py --input %s --dump-kernel xegpu-wg | FileCheck %s --check-prefix=WG --implicit-check-not=memref.alloc --implicit-check-not=memref.copy --implicit-check-not=gpu.alloc
# REQUIRES: torch, torch_mlir

# WG: gpu.func @main_kernel
# WG: scf.for %[[QUERY:[a-zA-Z0-9_]+]] =
# WG: vector.broadcast %[[QUERY]] : index
# WG: scf.for %[[KEY:[a-zA-Z0-9_]+]] =
# WG: vector.broadcast %[[KEY]] : index
# WG: arith.cmpi sle, {{.*}} : vector<128x64xi64>
# WG: arith.select
# WG: xegpu.dpas
# WG: vector.multi_reduction <maxnumf>
# WG: math.exp
# WG: xegpu.dpas
# WG: xegpu.store_nd
# WG-NOT: gpu.func

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Llama3.2-1B scaled dot-product attention (GQA + causal), the realistic
    counterpart to KernelBench level1-97.

    Differences from the synthetic level1-97 SDPA:

      * d_head = 64 (Llama3.2-1B), the per-head reduction dimension, within the
        current <= 64 lowering limit.
      * Grouped-query attention: 32 query heads share 8 key/value heads
        (num_heads / num_kv_heads = 4). K/V carry fewer heads than Q and are
        broadcast across the group (enable_gqa=True).
      * Causal masking (is_causal=True), as in autoregressive decoding/prefill.

    RoPE is intentionally NOT applied here. Q/K/V are taken as
    already-projected inputs, matching level1-97.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        # Q: (batch, num_heads, n_ctx, d_head); K/V: (batch, num_kv_heads, n_ctx, d_head)
        return torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, is_causal=True, enable_gqa=True
        )


# Llama3.2-1B attention config.
batch_size = 4
n_ctx = 512
num_heads = 32  # query heads
num_kv_heads = 8  # grouped key/value heads (GQA group size = 4)
d_head = 64  # per-head reduction dim; within the current <= 64 cap


def get_inputs():
    Q = torch.rand(batch_size, num_heads, n_ctx, d_head)
    K = torch.rand(batch_size, num_kv_heads, n_ctx, d_head)
    V = torch.rand(batch_size, num_kv_heads, n_ctx, d_head)
    return [Q, K, V]


def get_init_inputs():
    return []
