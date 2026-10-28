from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect

# Upper bounds for the elements of one workgroup tile and of its innermost dim.
_MAX_WG_TILE_ELEMS = 8192
_MAX_WG_TILE_INNER = 256


def _largest_pow2_divisor(extent: int, bound: int) -> int:
    size = 1
    while size * 2 <= bound and extent % (size * 2) == 0:
        size *= 2
    return size


def elementwise_wg_tile(shape: list[int]) -> list[int]:
    """Workgroup tile for an elementwise iteration space.

    Tiles the two innermost dims into a row-major block of at most
    `_MAX_WG_TILE_ELEMS` elements (keeping the innermost dim whole up to
    `_MAX_WG_TILE_INNER`) and the outer dims by one. Zero marks an untiled dim.
    """
    tile = [1] * len(shape)
    inner = shape[-1]
    if inner > _MAX_WG_TILE_INNER:
        inner = _largest_pow2_divisor(inner, _MAX_WG_TILE_INNER)
    tile[-1] = inner
    if len(shape) > 1:
        tile[-2] = _largest_pow2_divisor(shape[-2], _MAX_WG_TILE_ELEMS // inner)
    return [0 if t == extent else t for t, extent in zip(tile, shape)]


class InferXeGPUElementwiseParamsOp(
    TransformExtensionDialect.Operation, name="infer_xegpu_elementwise_params"
):
    """
    Infer the XeGPU workgroup tile of an elementwise anchor op.

    The tile is derived from the static shape of the single result of the
    target (see `elementwise_wg_tile`). Zero marks an untiled dim.

    Args:
        target: Handle to a single elementwise anchor op.
    Return:
        Param holding the wg tile sizes, one i64 per iteration dim.
    """

    target: ext.Operand[transform.AnyOpType]
    wg_tile: ext.Result[transform.AnyParamType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "InferXeGPUElementwiseParamsOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            targets = state.get_payload_ops(op.target)
            if len(targets) != 1 or len(targets[0].results) != 1:
                return DiagnosedSilenceableFailure.SilenceableFailure
            ty = targets[0].results[0].type
            if not isinstance(ty, ir.RankedTensorType) or not ty.has_static_shape:
                return DiagnosedSilenceableFailure.SilenceableFailure
            tile = elementwise_wg_tile(list(ty.shape))
            i64 = ir.IntegerType.get_signless(64)
            results.set_params(op.wg_tile, [ir.IntegerAttr.get(i64, t) for t in tile])
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(
            _op: "InferXeGPUElementwiseParamsOp",
        ) -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.only_reads_payload()
            )


def infer_xegpu_elementwise_params(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value:
    """
    snake_case wrapper to create an InferXeGPUElementwiseParamsOp.

    Args:
        target: Handle to a single elementwise anchor op.
    Return:
        Param holding the wg tile sizes.
    """
    return InferXeGPUElementwiseParamsOp(target=target).wg_tile
