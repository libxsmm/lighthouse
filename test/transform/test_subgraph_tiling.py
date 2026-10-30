# RUN: %PYTHON %s | FileCheck %s

"""Subgraph tagging and per-subgraph workgroup tiling."""

import numpy as np
from mlir import ir
from mlir.dialects import transform
from mlir.passmanager import PassManager

import lighthouse.dialects as lh_dialects
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.dialects.transform.transform_ext.utils.subgraph_analysis import (
    SUBGRAPH_ANCHOR_ATTR_NAME,
    SUBGRAPH_ID_ATTR_NAME,
    SUBGRAPH_KIND_ATTR_NAME,
)
from lighthouse.execution.runner import Runner
from lighthouse.ingress.mlir_gen import generate_gpu_matmul_payload
from lighthouse.pipeline.helper import match
from lighthouse.schedule import schedule_boilerplate
from lighthouse.schedule.xegpu import cleanup_schedule
from lighthouse.schedule.xegpu.wg_tiling_schedule import apply_subgraph_tiling


# Two GEMMs share the accumulator fill. The first has a ReLU epilogue, the
# second feeds a token reshape + elementwise DAG fused into its epilogue.
SOURCE = """
#id2 = affine_map<(m,n)->(m,n)>
#id3 = affine_map<(b,s,d)->(b,s,d)>
func.func @payload(%x: tensor<256x128xf32>, %w1: tensor<128x256xf32>,
                   %w2: tensor<128x256xf32>, %r1: memref<256x256xf32>,
                   %r2: memref<4x64x256xf32>) {
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<256x256xf32>
  %init = linalg.fill ins(%zero : f32) outs(%e : tensor<256x256xf32>) -> tensor<256x256xf32>
  %mm1 = linalg.matmul ins(%x, %w1 : tensor<256x128xf32>, tensor<128x256xf32>)
                       outs(%init : tensor<256x256xf32>) -> tensor<256x256xf32>
  %relu = linalg.generic {indexing_maps=[#id2,#id2],
      iterator_types=["parallel","parallel"]}
    ins(%mm1 : tensor<256x256xf32>) outs(%e : tensor<256x256xf32>) {
  ^bb0(%a: f32, %o: f32):
    %m = arith.maximumf %a, %zero : f32
    linalg.yield %m : f32
  } -> tensor<256x256xf32>
  %mm2 = linalg.matmul ins(%x, %w2 : tensor<256x128xf32>, tensor<128x256xf32>)
                       outs(%init : tensor<256x256xf32>) -> tensor<256x256xf32>
  %heads = tensor.expand_shape %mm2 [[0,1],[2]] output_shape [4,64,256]
    : tensor<256x256xf32> into tensor<4x64x256xf32>
  %e3 = tensor.empty() : tensor<4x64x256xf32>
  %neg = linalg.generic {indexing_maps=[#id3,#id3],
      iterator_types=["parallel","parallel","parallel"]}
    ins(%heads : tensor<4x64x256xf32>) outs(%e3 : tensor<4x64x256xf32>) {
  ^bb0(%a: f32, %o: f32):
    %n = arith.negf %a : f32
    linalg.yield %n : f32
  } -> tensor<4x64x256xf32>
  %sum = linalg.generic {indexing_maps=[#id3,#id3,#id3],
      iterator_types=["parallel","parallel","parallel"]}
    ins(%neg, %heads : tensor<4x64x256xf32>, tensor<4x64x256xf32>)
    outs(%e3 : tensor<4x64x256xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %s = arith.mulf %a, %b : f32
    linalg.yield %s : f32
  } -> tensor<4x64x256xf32>
  bufferization.materialize_in_destination %relu in restrict writable %r1
    : (tensor<256x256xf32>, memref<256x256xf32>) -> ()
  bufferization.materialize_in_destination %sum in restrict writable %r2
    : (tensor<4x64x256xf32>, memref<4x64x256xf32>) -> ()
  return
}
"""

REDUCTION = SOURCE.replace(
    "  return\n",
    """  %re = tensor.empty() : tensor<4x64xf32>
  %red = linalg.reduce ins(%sum : tensor<4x64x256xf32>)
    outs(%re : tensor<4x64xf32>) dimensions = [2]
    (%in: f32, %acc: f32) {
      %r = arith.addf %in, %acc : f32
      linalg.yield %r : f32
    }
  return
""",
)

NORMALIZED = SOURCE.replace(
    "  %mm1 = linalg.matmul",
    """  %ne = tensor.empty() : tensor<256x128xf32>
  %square = linalg.generic {indexing_maps=[#id2,#id2],
      iterator_types=["parallel","parallel"]}
    ins(%x : tensor<256x128xf32>) outs(%ne : tensor<256x128xf32>) {
  ^bb0(%a: f32, %o: f32):
    %v = arith.mulf %a, %a : f32
    linalg.yield %v : f32
  } -> tensor<256x128xf32>
  %se = tensor.empty() : tensor<256xf32>
  %si = linalg.fill ins(%zero : f32) outs(%se : tensor<256xf32>) -> tensor<256xf32>
  %norm_sum = linalg.generic {
      indexing_maps=[#id2,affine_map<(m,n)->(m)>],
      iterator_types=["parallel","reduction"]}
    ins(%square : tensor<256x128xf32>) outs(%si : tensor<256xf32>) {
  ^bb0(%a: f32, %acc: f32):
    %v = arith.addf %a, %acc : f32
    linalg.yield %v : f32
  } -> tensor<256xf32>
  %norm = linalg.generic {
      indexing_maps=[#id2,affine_map<(m,n)->(m)>,#id2],
      iterator_types=["parallel","parallel"]}
    ins(%x, %norm_sum : tensor<256x128xf32>, tensor<256xf32>)
    outs(%ne : tensor<256x128xf32>) {
  ^bb0(%a: f32, %s: f32, %o: f32):
    %scale = math.rsqrt %s : f32
    %v = arith.mulf %a, %scale : f32
    linalg.yield %v : f32
  } -> tensor<256x128xf32>
  %mm1 = linalg.matmul""",
).replace("ins(%x, %w", "ins(%norm, %w")


def operations(module, prefix):
    found = []

    def visit(op):
        if op.name.startswith(prefix):
            found.append(op)
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit)
    return found


def describe(op):
    attrs = op.attributes
    if SUBGRAPH_ID_ATTR_NAME not in attrs:
        return f"{op.name} untagged"
    kind = ir.StringAttr(attrs[SUBGRAPH_KIND_ATTR_NAME]).value
    id = ir.IntegerAttr(attrs[SUBGRAPH_ID_ATTR_NAME]).value
    anchor = " anchor" if SUBGRAPH_ANCHOR_ATTR_NAME in attrs else ""
    roles = [r for r in ("producer", "consumer") if f"test.{r}" in attrs]
    return f"{op.name} {kind} {id}{anchor}" + "".join(f" {r}" for r in roles)


def run(source, build):
    module = ir.Module.parse(source)
    with schedule_boilerplate() as (schedule, sequence):
        func = lh_transform.match_op(sequence.bodyTarget, "func.func").results[0]
        build(func)
        transform.yield_([])
    schedule.body.operations[0].apply(module.operation)
    assert module.operation.verify()
    return module


def tag_and_mark(func):
    func = transform_ext.tag_subgraphs(func, kinds=["gemm", "elementwise"])
    anchors = match(func, op_attrs={SUBGRAPH_ANCHOR_ATTR_NAME: ir.UnitAttr.get()})
    with lh_transform.foreach(anchors) as anchor:
        producers, consumers = transform_ext.get_subgraph_ops(anchor)
        transform.annotate(producers, "test.producer")
        transform.annotate(consumers, "test.consumer")
        transform.yield_()


def test_tagging():
    module = run(SOURCE, tag_and_mark)
    for op in operations(module, "linalg.") + operations(module, "tensor.expand"):
        if op.name != "linalg.yield":
            print(describe(op))


# CHECK-LABEL: test_tagging
# The shared fill is cloned for the first GEMM, the second claims the original.
# CHECK-NEXT: linalg.fill gemm 1 producer
# CHECK-NEXT: linalg.fill gemm 2 producer
# CHECK-NEXT: linalg.matmul gemm 1 producer
# CHECK-NEXT: linalg.generic gemm 1 anchor
# CHECK-NEXT: linalg.matmul gemm 2 anchor
# The token reshape and DAG are fused forward into the GEMM loop.
# CHECK-NEXT: linalg.generic gemm 2 consumer
# CHECK-NEXT: linalg.generic gemm 2 consumer
# CHECK-NEXT: tensor.expand_shape gemm 2 consumer


def test_untagged():
    try:
        run(
            REDUCTION, lambda f: transform_ext.tag_subgraphs(f, ["gemm", "elementwise"])
        )
    except Exception:
        print("Reduction left untagged: tagging failed")


# CHECK-LABEL: test_untagged
# CHECK-NEXT: Reduction left untagged: tagging failed


def tile(func):
    apply_subgraph_tiling(func, force_wg_tile=[128, 128], force_k_tile=32)


def test_tiling():
    module = run(SOURCE, tile)
    loops = [op for op in operations(module, "scf.forall") if op.name == "scf.forall"]
    print(f"{len(loops)} workgroup loops")
    for loop in loops:
        names = sorted(
            {op.name for op in operations(loop, "linalg.")} - {"linalg.yield"}
        )
        print(", ".join(names))
    tagged = [
        o for o in operations(module, "") if SUBGRAPH_ID_ATTR_NAME in o.attributes
    ]
    assert not tagged, "subgraph tags must be cleared"


# CHECK-LABEL: test_tiling
# CHECK-NEXT: 2 workgroup loops
# CHECK-NEXT: linalg.fill, linalg.generic, linalg.matmul
# CHECK-NEXT: linalg.fill, linalg.generic, linalg.matmul


def test_numerics(source=SOURCE):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(256, 128)).astype(np.float32)
    w1 = rng.normal(size=(128, 256)).astype(np.float32)
    w2 = rng.normal(size=(128, 256)).astype(np.float32)
    normalized = (
        x if source == SOURCE else x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
    )
    heads = (normalized @ w2).reshape(4, 64, 256)
    expected = [np.maximum(normalized @ w1, 0), -heads * heads]
    module = run(source, tile)
    PassManager.parse(
        "builtin.module(empty-tensor-to-alloc-tensor,"
        "one-shot-bufferize{bufferize-function-boundaries "
        "function-boundary-type-conversion=identity-layout-map},"
        "scf-forall-to-for,buffer-deallocation-pipeline,"
        "convert-linalg-to-loops,expand-strided-metadata,lower-affine,"
        "convert-scf-to-cf,convert-math-to-llvm)"
    ).run(module.operation)
    Runner.make_function_callable(module, "payload")
    PassManager.parse(
        "builtin.module(finalize-memref-to-llvm,convert-to-llvm,"
        "reconcile-unrealized-casts)"
    ).run(module.operation)
    outputs = [np.empty_like(e) for e in expected]
    Runner(module).execute("payload", [x, w1, w2, *outputs])
    for output, reference in zip(outputs, expected):
        np.testing.assert_allclose(output, reference, rtol=1e-5, atol=1e-5)
    print("Tiled subgraphs match NumPy")


# CHECK-LABEL: test_numerics
# CHECK-NEXT: Tiled subgraphs match NumPy


def test_mixed_reduction():
    module = run(NORMALIZED, tile)
    loops = [op for op in operations(module, "scf.forall") if op.name == "scf.forall"]
    assert len(loops) == 3, str(module)
    assert len(operations(loops[0], "math.rsqrt")) == 1
    assert not operations(loops[0], "linalg.matmul")
    for loop in loops[1:]:
        assert not operations(loop, "math.rsqrt"), (
            "normalization must not be recomputed"
        )
    assert len(operations(module, "linalg.matmul")) == 2
    test_numerics(NORMALIZED)
    print("Shared reduction is materialized once before both GEMM consumers")


# CHECK-LABEL: test_mixed_reduction
# CHECK-NEXT: Tiled subgraphs match NumPy
# CHECK-NEXT: Shared reduction is materialized once before both GEMM consumers


def test_shared_normalization_input():
    source = (
        NORMALIZED.replace(
            "%r2: memref<4x64x256xf32>)",
            "%r2: memref<4x64x256xf32>, %residual_out: memref<256x128xf32>)",
        )
        .replace(
            "  %ne = tensor.empty()",
            """  %residual_empty = tensor.empty() : tensor<256x128xf32>
  %residual = linalg.generic {indexing_maps=[#id2,#id2,#id2],
      iterator_types=["parallel","parallel"]}
    ins(%x, %x : tensor<256x128xf32>, tensor<256x128xf32>)
    outs(%residual_empty : tensor<256x128xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %sum = arith.addf %a, %b : f32
    linalg.yield %sum : f32
  } -> tensor<256x128xf32>
  bufferization.materialize_in_destination %residual in restrict writable %residual_out
    : (tensor<256x128xf32>, memref<256x128xf32>) -> ()
  %ne = tensor.empty()""",
        )
        .replace(
            "ins(%x : tensor<256x128xf32>)", "ins(%residual : tensor<256x128xf32>)"
        )
        .replace("ins(%x, %norm_sum", "ins(%residual, %norm_sum")
    )
    module = run(
        source,
        lambda func: transform_ext.tag_subgraphs(
            func, ["reduction", "gemm", "elementwise"]
        ),
    )
    add = operations(module, "arith.addf")[0]
    assert (
        ir.StringAttr(add.parent.attributes[SUBGRAPH_KIND_ATTR_NAME]).value
        == "elementwise"
    )
    module = run(source, tile)
    loops = [op for op in operations(module, "scf.forall") if op.name == "scf.forall"]
    assert len(loops) == 4, str(module)
    normalization = next(loop for loop in loops if operations(loop, "math.rsqrt"))
    assert len(operations(normalization, "arith.addf")) == 1
    assert len(operations(module, "arith.addf")) == 4  # Residual, reduction, two GEMMs.
    print("Shared residual stays outside normalization during partitioning and tiling")


# CHECK-LABEL: test_shared_normalization_input
# CHECK-NEXT: Shared residual stays outside normalization during partitioning and tiling


def test_late_bias_relu_setup():
    rng = np.random.default_rng(42)
    a = rng.integers(-3, 4, size=(256, 64)).astype(np.float32)
    b = rng.integers(-3, 4, size=(64, 256)).astype(np.float32)
    bias = rng.integers(-3, 4, size=256).astype(np.float32)
    initial = rng.integers(-3, 4, size=(256, 256)).astype(np.float32)
    for accumulate in (False, True):
        module = generate_gpu_matmul_payload(
            func_name="payload",
            M=256,
            N=256,
            K=64,
            ab_type=ir.F32Type.get(),
            c_type=ir.F32Type.get(),
            transpose_a=False,
            transpose_b=False,
            has_bias=True,
            has_relu=True,
            accumulate_c=accumulate,
        )
        cleanup = cleanup_schedule()
        cleanup.body.operations[0].apply(module.operation)
        module = run(str(module), tile)
        (loop,) = [
            op for op in operations(module, "scf.forall") if op.name == "scf.forall"
        ]
        (matmul,) = operations(module, "linalg.matmul")
        (relu,) = operations(module, "arith.maximumf")
        assert matmul.parent.name == "scf.for"
        assert relu.parent.parent == loop
        PassManager.parse(
            "builtin.module(empty-tensor-to-alloc-tensor,"
            "one-shot-bufferize{bufferize-function-boundaries "
            "function-boundary-type-conversion=identity-layout-map},"
            "scf-forall-to-for,buffer-deallocation-pipeline,"
            "convert-linalg-to-loops,expand-strided-metadata,lower-affine,"
            "convert-scf-to-cf,convert-math-to-llvm)"
        ).run(module.operation)
        Runner.make_function_callable(module, "payload")
        PassManager.parse(
            "builtin.module(finalize-memref-to-llvm,convert-to-llvm,"
            "reconcile-unrealized-casts)"
        ).run(module.operation)
        output = initial.copy()
        Runner(module).execute("payload", [output, a, b, bias])
        expected = np.maximum(a @ b + (initial if accumulate else 0) + bias, 0)
        np.testing.assert_array_equal(output, expected)
    print("Late bias/ReLU setup fuses into one GEMM kernel and matches NumPy")


# CHECK-LABEL: test_late_bias_relu_setup
# CHECK-NEXT: Late bias/ReLU setup fuses into one GEMM kernel and matches NumPy


with ir.Context(), ir.Location.unknown():
    lh_dialects.register_and_load()
    for test in (
        test_tagging,
        test_untagged,
        test_tiling,
        test_numerics,
        test_mixed_reduction,
        test_shared_normalization_input,
        test_late_bias_relu_setup,
    ):
        print(test.__name__)
        test()
