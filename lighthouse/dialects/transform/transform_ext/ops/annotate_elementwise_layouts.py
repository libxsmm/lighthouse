from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.schedule.xegpu.matmul_constraints import NB_WORKITEMS


class AnnotateElementwiseLayoutsOp(
    TransformExtensionDialect.Operation, name="annotate_elementwise_layouts"
):
    """Anchor loads and stores in single-subgroup elementwise GPU kernels.

    Every memory tile belongs to the single subgroup, so its sg_data is the
    complete vector shape and sg_layout is all ones. Instruction and lane
    layouts are left to XeGPU layout propagation. This also handles unit
    dimensions retained on loads before vector.shape_cast.
    Direct broadcast inputs are left for annotate_epilogue_broadcasts, which
    derives their sliced layouts from the stores.

    Requires a gpu.func with known_block_size of one subgroup, at least one
    store_nd, and no contractions or reductions. Unsupported kernels fail
    silenceably without changing any annotations.
    """

    target: ext.Operand[transform.AnyOpType]

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(op, _rewriter, _results, state) -> DiagnosedSilenceableFailure:
            annotations = []
            for target in state.get_payload_ops(op.target):
                if (
                    target.operation.name != "gpu.func"
                    or "known_block_size" not in target.attributes
                    or list(target.attributes["known_block_size"])
                    != [NB_WORKITEMS, 1, 1]
                ):
                    return DiagnosedSilenceableFailure.SilenceableFailure
                memory_ops = []
                unsupported = False

                def visit(nested):
                    nonlocal unsupported
                    if nested.name in (
                        "xegpu.dpas",
                        "vector.contract",
                        "vector.multi_reduction",
                        "vector.reduction",
                    ):
                        unsupported = True
                    if nested.name in ("xegpu.load_nd", "xegpu.store_nd"):
                        memory_ops.append(nested)
                    return ir.WalkResult.ADVANCE

                target.walk(visit)
                if unsupported or not any(
                    m.name == "xegpu.store_nd" for m in memory_ops
                ):
                    return DiagnosedSilenceableFailure.SilenceableFailure
                for memory in memory_ops:
                    if memory.name == "xegpu.load_nd" and any(
                        use.owner.name == "vector.broadcast"
                        for use in memory.results[0].uses
                    ):
                        continue
                    value = (
                        memory.results[0]
                        if memory.name == "xegpu.load_nd"
                        else memory.operands[0]
                    )
                    ty = value.type
                    if not isinstance(ty, ir.VectorType) or ty.scalable or not ty.rank:
                        return DiagnosedSilenceableFailure.SilenceableFailure
                    shape = list(ty.shape)
                    layout = ir.Attribute.parse(
                        f"#xegpu.layout<sg_layout = {[1] * ty.rank}, sg_data = {shape}>"
                    )
                    if (
                        "layout" in memory.attributes
                        and memory.attributes["layout"] != layout
                    ):
                        memory.location.emit_error(
                            "annotate_elementwise_layouts: conflicting memory layout"
                        )
                        return DiagnosedSilenceableFailure.DefiniteFailure
                    annotations.append((memory, layout))
            for memory, layout in annotations:
                memory.attributes["layout"] = layout
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op):
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.modifies_payload()
            )


def annotate_elementwise_layouts(target: ir.Value[transform.AnyOpType]):
    return AnnotateElementwiseLayoutsOp(target=target)
