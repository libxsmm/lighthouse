# RUN: %PYTHON %s | FileCheck %s

"""Test elementwise workgroup tile inference."""

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

import lighthouse.dialects as lh_dialects
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate


PAYLOAD = """
module {
  func.func @payload(%arg: tensor<4x32x512x64xf32>,
                     %out: tensor<4x32x512x64xf32>)
                     -> tensor<4x32x512x64xf32> {
    %result = linalg.generic {
        indexing_maps=[affine_map<(b,h,s,d)->(b,h,s,d)>,
                       affine_map<(b,h,s,d)->(b,h,s,d)>],
        iterator_types=["parallel","parallel","parallel","parallel"]}
      ins(%arg : tensor<4x32x512x64xf32>)
      outs(%out : tensor<4x32x512x64xf32>) {
    ^bb0(%a: f32, %o: f32):
      linalg.yield %a : f32
    } -> tensor<4x32x512x64xf32>
    return %result : tensor<4x32x512x64xf32>
  }
}
"""


def build_schedule():
    with schedule_boilerplate() as (schedule, named_sequence):
        generic = lh_transform.match_op(named_sequence.bodyTarget, "linalg.generic")
        generic = transform_ext.extract_handle(generic, 0)
        wg_tile = transform_ext.infer_xegpu_elementwise_params(generic)
        structured.TileUsingForallOp(generic, tile_sizes=wg_tile)
        transform.yield_()
    return schedule


def test():
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(PAYLOAD)
        schedule = build_schedule()
        schedule.body.operations[0].apply(payload.operation)
        print(payload)


# CHECK: scf.forall
# CHECK-SAME: in (4, 32, 4)
# CHECK: tensor.extract_slice
# CHECK-SAME: [1, 1, 128, 64]


test()
