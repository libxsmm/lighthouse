# RUN: %PYTHON %s | FileCheck %s

"""Single-subgroup memory layouts, schedule dispatch and rejection tests."""

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule import schedule_boilerplate
from lighthouse.schedule.xegpu.annotate_layouts_schedule import (
    annotate_layouts_schedule,
)


SOURCE = """
module {
  gpu.module @kernels {
    gpu.func @elementwise(%input: memref<128x1x32xbf16>,
                         %table: memref<32xbf16>,
                         %output: memref<128x32xbf16>) kernel
        attributes {known_block_size = array<i32: 16, 1, 1>} {
      %a_desc = xegpu.create_nd_tdesc %input : memref<128x1x32xbf16>
        -> !xegpu.tensor_desc<128x1x32xbf16>
      %b_desc = xegpu.create_nd_tdesc %table : memref<32xbf16>
        -> !xegpu.tensor_desc<32xbf16>
      %out_desc = xegpu.create_nd_tdesc %output : memref<128x32xbf16>
        -> !xegpu.tensor_desc<128x32xbf16>
      %a = xegpu.load_nd %a_desc[0,0,0]
        : !xegpu.tensor_desc<128x1x32xbf16> -> vector<128x1x32xbf16>
      %b = xegpu.load_nd %b_desc[0]
        : !xegpu.tensor_desc<32xbf16> -> vector<32xbf16>
      %flat = vector.shape_cast %a
        : vector<128x1x32xbf16> to vector<128x32xbf16>
      %broadcast = vector.broadcast %b
        : vector<32xbf16> to vector<128x32xbf16>
      %sum = arith.addf %flat, %broadcast : vector<128x32xbf16>
      xegpu.store_nd %sum, %out_desc[0,0]
        : vector<128x32xbf16>, !xegpu.tensor_desc<128x32xbf16>
      xegpu.store_nd %sum, %out_desc[0,0]
        : vector<128x32xbf16>, !xegpu.tensor_desc<128x32xbf16>
      gpu.return
    }
  }
}
"""


def operations(module, name):
    found = []

    def collect(op):
        if op.name == name:
            found.append(op)
        return ir.WalkResult.ADVANCE

    module.operation.walk(collect)
    return found


def apply(module, *, dispatch=False):
    if dispatch:
        schedule = annotate_layouts_schedule()
    else:
        with schedule_boilerplate() as (schedule, sequence):
            funcs = lh_transform.match_op(sequence.bodyTarget, "gpu.func").results[0]
            transform_ext.annotate_elementwise_layouts(funcs)
            transform.yield_()
    schedule.body.operations[0].apply(module.operation)
    assert module.operation.verify()


def test_layouts():
    module = ir.Module.parse(SOURCE)
    apply(module)
    loads = operations(module, "xegpu.load_nd")
    stores = operations(module, "xegpu.store_nd")
    assert str(loads[0].attributes["layout"]) == (
        "#xegpu.layout<sg_layout = [1, 1, 1], sg_data = [128, 1, 32]>"
    )
    assert "layout" not in loads[1].attributes
    assert all(
        str(s.attributes["layout"])
        == "#xegpu.layout<sg_layout = [1, 1], sg_data = [128, 32]>"
        for s in stores
    )
    before = str(module)
    apply(module)
    assert str(module) == before
    print("Memory anchors annotated; broadcast inputs deferred; idempotent")


# CHECK: Memory anchors annotated; broadcast inputs deferred; idempotent


def test_schedule():
    module = ir.Module.parse(SOURCE)
    apply(module, dispatch=True)
    stores = operations(module, "xegpu.store_nd")
    assert len(stores) == 2 and all("layout" in s.attributes for s in stores)
    load = operations(module, "xegpu.load_nd")[1]
    assert str(load.attributes["layout"]).startswith("#xegpu.slice<")
    (broadcast,) = operations(module, "vector.broadcast")
    assert broadcast.attributes["layout_result_0"] == stores[0].attributes["layout"]
    before = str(module)
    apply(module, dispatch=True)
    assert str(module) == before
    print("Schedule dispatches elementwise and reuses broadcast annotations")


# CHECK: Schedule dispatches elementwise and reuses broadcast annotations


def test_rejections():
    reduction = SOURCE.replace(
        "      gpu.return",
        "      %zero = arith.constant 0.0 : bf16\n"
        "      %r = vector.reduction <add>, %b, %zero : vector<32xbf16> into bf16\n"
        "      gpu.return",
    )
    no_store = "\n".join(
        line
        for line in SOURCE.splitlines()
        if "xegpu.store_nd" not in line
        and ": vector<128x32xbf16>, !xegpu.tensor_desc" not in line
    )
    for source in (
        SOURCE.replace("16, 1, 1", "32, 1, 1"),
        SOURCE.replace("attributes {known_block_size = array<i32: 16, 1, 1>}", ""),
        reduction,
        no_store,
    ):
        module = ir.Module.parse(source)
        before = str(module)
        try:
            apply(module)
        except ValueError as error:
            assert "silenceable failure" in str(error)
        else:
            raise AssertionError("Expected unsupported kernel to fail")
        assert str(module) == before
    print("Multi-subgroup, unknown block size, reduction and storeless rejected")


# CHECK: Multi-subgroup, unknown block size, reduction and storeless rejected


def test_conflict():
    module = ir.Module.parse(SOURCE)
    loads = operations(module, "xegpu.load_nd")
    loads[0].attributes["layout"] = ir.Attribute.parse(
        "#xegpu.layout<sg_layout=[2,1,1], sg_data=[64,1,32]>"
    )
    before = str(module)
    try:
        apply(module)
    except ValueError as error:
        assert "conflicting memory layout" in str(error)
    else:
        raise AssertionError("Expected conflicting layout to fail")
    assert str(module) == before
    print("Conflicting layouts fail without partial annotations")


# CHECK: Conflicting layouts fail without partial annotations


if __name__ == "__main__":
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        test_layouts()
        test_schedule()
        test_rejections()
        test_conflict()
