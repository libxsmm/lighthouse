"""Forward tile propagation for closed, tensor-only epilogue DAGs.

This is a prototype of a consumer-fusion transform, not a RoPE matcher. Planning
is separate from rewriting: unsupported tile images leave the payload unchanged.
An MLIR implementation should obtain these images from operation interfaces.
"""

from dataclasses import dataclass
from math import prod

from mlir import ir
from mlir.dialects import affine, ext, linalg, scf, tensor, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.utils.mlir import indexing_maps


@dataclass(frozen=True)
class Tile:
    """Global rectangular tile; affine dimensions denote the forall IVs."""

    offsets: tuple[ir.AffineExpr, ...]
    sizes: tuple[int, ...]


def _shape(value):
    ty = value.type
    if (
        not isinstance(ty, ir.RankedTensorType)
        or ty.encoding is not None
        or any(size <= 0 for size in ty.shape)
    ):
        return None
    return tuple(ty.shape)


def _zero():
    return ir.AffineExpr.get_constant(0)


def _groups(op):
    return [[int(dim) for dim in group] for group in op.reassociation]


def _tensor_only(operation):
    pure = True

    def check(nested):
        nonlocal pure
        if nested.name.split(".")[0] not in (
            "arith",
            "math",
            "complex",
            "affine",
            "linalg",
            "tensor",
            "scf",
        ) or any(
            isinstance(v.type, ir.ShapedType)
            and not isinstance(v.type, ir.RankedTensorType)
            for v in nested.operands
        ):
            pure = False
        return ir.WalkResult.ADVANCE

    operation.walk(check)
    return pure


def _generic_tiles(op, tiles):
    maps = indexing_maps(op)
    rank = len(op.iterator_types)
    body = op.regions[0].blocks[0]
    if (
        len(op.results) != 1
        or any(str(it) != "#linalg.iterator_type<parallel>" for it in op.iterator_types)
        or any(not m.is_projected_permutation or m.n_symbols for m in maps)
        or any(
            not isinstance(expr, ir.AffineDimExpr)
            for mapping in maps
            for expr in mapping.results
        )
        or not maps[-1].is_permutation
        or any(
            nested.operation.name.split(".")[0] not in ("arith", "math", "complex")
            and nested.operation.name != "linalg.yield"
            for nested in body.operations
        )
    ):
        return None
    dimensions = [None] * rank
    for value, mapping in zip(op.operands, maps):
        if value not in tiles:
            continue
        tile = tiles[value]
        for expr, offset, size in zip(mapping.results, tile.offsets, tile.sizes):
            entry = (offset, size)
            if dimensions[expr.position] not in (None, entry):
                return None
            dimensions[expr.position] = entry
    if any(entry is None for entry in dimensions):
        return None
    operands = []
    for value, mapping in zip(op.operands, maps):
        shape = _shape(value)
        if shape is None:
            if isinstance(value.type, ir.ShapedType) or mapping.results:
                return None
            operands.append(None)
            continue
        entries = [dimensions[expr.position] for expr in mapping.results]
        tile = Tile(tuple(e[0] for e in entries), tuple(e[1] for e in entries))
        if any(size > extent for size, extent in zip(tile.sizes, shape)):
            return None
        operands.append(tile)
    return operands[-1], operands


def _image(op, tiles):
    """Return the result tile and external operand tiles, or reject the image."""
    if len(op.results) != 1 or _shape(op.results[0]) is None:
        return None
    if isinstance(op, linalg.GenericOp):
        return _generic_tiles(op, tiles)
    if isinstance(op, linalg.TransposeOp):
        tile = tiles.get(op.operands[0])
        if tile is None:
            return None
        perm = list(op.permutation)
        result = Tile(
            tuple(tile.offsets[i] for i in perm), tuple(tile.sizes[i] for i in perm)
        )
        return result, [tile, result]
    if isinstance(op, (tensor.ExpandShapeOp, tensor.CollapseShapeOp)):
        tile = tiles.get(op.src)
        if tile is None:
            return None
        source_shape, result_shape = _shape(op.src), _shape(op.result)
        offsets, sizes = [], []
        if isinstance(op, tensor.ExpandShapeOp):
            for offset, size, group in zip(tile.offsets, tile.sizes, _groups(op)):
                shape = [result_shape[i] for i in group]
                pivot = next(
                    (
                        i
                        for i in range(len(shape))
                        if size % prod(shape[i + 1 :]) == 0
                        and prod(shape[i:]) % size == 0
                    ),
                    None,
                )
                if pivot is None:
                    return None
                for i, extent in enumerate(shape):
                    stride = prod(shape[i + 1 :])
                    local_size = (
                        1 if i < pivot else size // stride if i == pivot else extent
                    )
                    offsets.append(
                        _zero()
                        if local_size == extent
                        else ir.AffineExpr.get_floor_div(
                            offset, ir.AffineExpr.get_constant(stride)
                        )
                        % extent
                    )
                    sizes.append(local_size)
        else:
            for group in _groups(op):
                pivot = next((i for i in group if tile.sizes[i] != 1), group[-1])
                if any(tile.sizes[i] != source_shape[i] for i in group if i > pivot):
                    return None
                offsets.append(
                    sum(
                        (
                            tile.offsets[i]
                            * prod(source_shape[j] for j in group if j > i)
                            for i in group
                        ),
                        _zero(),
                    )
                )
                sizes.append(prod(tile.sizes[i] for i in group))
        return Tile(tuple(offsets), tuple(sizes)), [tile]
    if isinstance(op, tensor.ExtractSliceOp):
        tile = tiles.get(op.source)
        if (
            tile is None
            or len(tile.sizes) != len(_shape(op.result))
            or op.offsets
            or op.sizes
            or op.strides
            or any(int(s) != 1 for s in op.static_strides)
        ):
            return None
        offsets, sizes = [], []
        for extent, start, size, offset, tile_size in zip(
            _shape(op.source),
            op.static_offsets,
            op.static_sizes,
            tile.offsets,
            tile.sizes,
        ):
            if start == 0 and size == extent:
                offsets.append(offset)
                sizes.append(tile_size)
            elif tile_size == extent and offset == _zero():
                offsets.append(_zero())
                sizes.append(int(size))
            else:
                return None
        return Tile(tuple(offsets), tuple(sizes)), [tile]
    if isinstance(op, tensor.ConcatOp):
        inputs = list(op.inputs)
        if any(value not in tiles for value in inputs):
            return None
        dim = int(op.dim)
        first = tiles[inputs[0]]
        for value in inputs:
            tile = tiles[value]
            if tile.sizes[dim] != _shape(value)[dim] or tile.offsets[dim] != _zero():
                return None
            if any(
                (tile.offsets[i], tile.sizes[i]) != (first.offsets[i], first.sizes[i])
                for i in range(len(first.sizes))
                if i != dim
            ):
                return None
        sizes = list(first.sizes)
        sizes[dim] = sum(tiles[value].sizes[dim] for value in inputs)
        return Tile(first.offsets, tuple(sizes)), [tiles[value] for value in inputs]
    return None


def _plan(loop):
    if (
        len(loop.results) != 1
        or loop.dynamicLowerBound
        or loop.dynamicUpperBound
        or loop.dynamicStep
    ):
        return None
    body = loop.regions[0].blocks[0]
    rank = len(loop.staticStep)
    parallel = body.operations[-1]
    if parallel.operation.name != "scf.forall.in_parallel":
        return None
    inserts = list(parallel.regions[0].blocks[0].operations)
    if len(inserts) != 1 or inserts[0].operation.name != "tensor.parallel_insert_slice":
        return None
    insert = inserts[0]
    if (
        insert.sizes
        or insert.strides
        or any(int(s) != 1 for s in insert.static_strides)
    ):
        return None
    shape = _shape(loop.results[0])
    sizes = tuple(int(s) for s in insert.static_sizes)
    if (
        shape is None
        or len(sizes) != len(shape)
        or _shape(insert.source) != sizes
        or insert.dest != body.arguments[rank]
        or any(size <= 0 or extent % size for extent, size in zip(shape, sizes))
    ):
        return None
    offsets, dynamic = [], iter(insert.offsets)
    used_ivs = set()
    for offset in insert.static_offsets:
        if ir.ShapedType.is_dynamic_size(int(offset)):
            value = next(dynamic)
            if value not in body.arguments[:rank]:
                return None
            index = list(body.arguments[:rank]).index(value)
            size, extent = sizes[len(offsets)], shape[len(offsets)]
            step = int(loop.staticStep[index])
            if (
                index in used_ivs
                or int(loop.staticLowerBound[index]) != 0
                or not (step == size or size == extent and step > size)
                or int(loop.staticUpperBound[index]) != extent
            ):
                return None
            used_ivs.add(index)
            offsets.append(_zero() if size == extent else ir.AffineExpr.get_dim(index))
        elif offset == 0 and sizes[len(offsets)] == shape[len(offsets)]:
            offsets.append(_zero())
        else:
            return None
    if len(used_ivs) != rank or len(sizes) != len(shape):
        return None
    if not _tensor_only(loop.operation):
        return None

    tiles = {loop.results[0]: Tile(tuple(offsets), sizes)}
    nodes, operands = [], {}
    block = loop.operation.block
    siblings = list(block.operations)
    position = next(
        i for i, op in enumerate(siblings) if op.operation == loop.operation
    )
    supported = (
        linalg.GenericOp,
        linalg.TransposeOp,
        tensor.ExpandShapeOp,
        tensor.CollapseShapeOp,
        tensor.ExtractSliceOp,
        tensor.ConcatOp,
    )
    for candidate in siblings[position + 1 :]:
        if not any(value in tiles for value in candidate.operands):
            continue
        if not isinstance(candidate, supported):
            continue
        image = _image(candidate, tiles)
        if image is None:
            return None
        tile, operand_tiles = image
        tiles[candidate.results[0]] = tile
        operands[candidate.operation] = operand_tiles
        nodes.append(candidate)
    if not nodes:
        return None
    node_ops = {op.operation for op in nodes}
    outputs = [
        value
        for value in tiles
        if any(use.owner.operation not in node_ops for use in value.uses)
    ]
    if not outputs:
        return None
    last = next(
        i for i, op in enumerate(siblings) if op.operation == nodes[-1].operation
    )
    # Moving the tiled computation must not cross a call or memory effect.
    if any(not _tensor_only(op.operation) for op in siblings[position + 1 : last]):
        return None
    positions = {op.operation: i for i, op in enumerate(siblings)}
    for value in outputs:
        if any(
            use.owner.operation not in node_ops
            and positions.get(use.owner.operation, -1) <= last
            for use in value.uses
        ):
            return None
    return insert, tiles, nodes, operands, outputs


def _type(value, tile):
    return ir.RankedTensorType.get(tile.sizes, value.type.element_type)


def _offsets(tile, ivs):
    result = []
    for expr in tile.offsets:
        if isinstance(expr, ir.AffineConstantExpr):
            result.append(expr.value)
        else:
            result.append(
                affine.AffineApplyOp(ir.AffineMap.get(len(ivs), 0, [expr]), ivs).result
            )
    return result


def _slice(value, tile, ivs):
    offsets = _offsets(tile, ivs)
    dynamic = [x for x in offsets if isinstance(x, ir.Value)]
    static = [
        ir.ShapedType.get_dynamic_size() if isinstance(x, ir.Value) else x
        for x in offsets
    ]
    return tensor.ExtractSliceOp(
        _type(value, tile),
        value,
        dynamic,
        [],
        [],
        static,
        tile.sizes,
        [1] * len(tile.sizes),
    ).result


def _rewrite(loop, plan, rewriter):
    insert, tiles, nodes, operand_tiles, outputs = plan
    old_body = loop.regions[0].blocks[0]
    rank = len(loop.staticStep)
    with loop.location, ir.InsertionPoint(nodes[-1]):
        destinations = [
            tensor.EmptyOp(_shape(value), value.type.element_type).result
            for value in outputs
        ]
        new_loop = scf.ForallOp(
            list(loop.staticLowerBound),
            list(loop.staticUpperBound),
            list(loop.staticStep),
            destinations,
            mapping=loop.mapping,
        )
        structural = {
            "operandSegmentSizes",
            "staticLowerBound",
            "staticUpperBound",
            "staticStep",
            "mapping",
        }
        for name, attribute in loop.attributes.items():
            if name not in structural:
                new_loop.attributes[name] = attribute
        body = new_loop.regions[0].blocks[0]
        ivs = list(body.arguments[:rank])
        values = dict(zip(old_body.arguments[:rank], ivs))
        values[old_body.arguments[rank]] = loop.outputs[0]
        with ir.InsertionPoint(body):
            for op in list(old_body.operations)[:-1]:
                cloned = op.operation.clone()

                def remap(operation):
                    for i, value in enumerate(operation.operands):
                        operation.operands[i] = values.get(value, value)
                    return ir.WalkResult.ADVANCE

                cloned.walk(remap)
                values.update(zip(op.results, cloned.results))
            values[loop.results[0]] = values.get(insert.source, insert.source)
            for op in nodes:
                tile = tiles[op.results[0]]
                if isinstance(op, (linalg.GenericOp, linalg.TransposeOp)):
                    mapped = []
                    for i, (value, operand_tile) in enumerate(
                        zip(op.operands, operand_tiles[op.operation])
                    ):
                        if value in values:
                            mapped.append(values[value])
                        elif operand_tile is None:
                            mapped.append(value)
                        elif i == len(op.operands) - 1 and (
                            isinstance(op, linalg.TransposeOp)
                            or not list(op.regions[0].blocks[0].arguments[-1].uses)
                        ):
                            mapped.append(
                                tensor.EmptyOp(
                                    operand_tile.sizes, value.type.element_type
                                ).result
                            )
                        else:
                            mapped.append(_slice(value, operand_tile, ivs))
                    cloned = op.operation.clone()
                    for i, value in enumerate(mapped):
                        cloned.operands[i] = value
                    cloned.results[0].set_type(_type(op.results[0], tile))
                    result = cloned.results[0]
                elif isinstance(op, tensor.ExpandShapeOp):
                    result = tensor.ExpandShapeOp(
                        _type(op.result, tile),
                        values[op.src],
                        _groups(op),
                        [],
                        tile.sizes,
                    ).result
                elif isinstance(op, tensor.CollapseShapeOp):
                    result = tensor.CollapseShapeOp(
                        _type(op.result, tile), values[op.src], _groups(op)
                    ).result
                elif isinstance(op, tensor.ExtractSliceOp):
                    local = Tile(
                        tuple(
                            ir.AffineExpr.get_constant(int(x))
                            for x in op.static_offsets
                        ),
                        tile.sizes,
                    )
                    result = _slice(values[op.source], local, ivs)
                else:
                    result = tensor.ConcatOp(
                        _type(op.result, tile),
                        int(op.dim),
                        [values[v] for v in op.inputs],
                    ).result
                values[op.results[0]] = result
            output_offsets = [_offsets(tiles[value], ivs) for value in outputs]
            parallel = scf.InParallelOp()
            parallel.regions[0].blocks.append()
            with ir.InsertionPoint(parallel.regions[0].blocks[0]):
                for value, dest, offsets in zip(
                    outputs, body.arguments[rank:], output_offsets
                ):
                    tensor.ParallelInsertSliceOp(
                        values[value],
                        dest,
                        [x for x in offsets if isinstance(x, ir.Value)],
                        [],
                        [],
                        [
                            ir.ShapedType.get_dynamic_size()
                            if isinstance(x, ir.Value)
                            else x
                            for x in offsets
                        ],
                        tiles[value].sizes,
                        [1] * len(tiles[value].sizes),
                    )
        for value, replacement in zip(outputs, new_loop.results):
            value.replace_all_uses_with(replacement)
        for op in reversed(nodes):
            rewriter.erase_op(op)
        rewriter.erase_op(loop)
    return new_loop.operation


class FuseForallEpilogueOp(
    TransformExtensionDialect.Operation, name="fuse_forall_epilogue"
):
    """Fuse a static tensor epilogue DAG into a single-output scf.forall.

    Supports parallel projected-permutation generics (without linalg.index),
    transpose, rectangular expand/collapse, static non-rank-reducing unit-stride
    slices, and concat along an untiled dimension. Tiles must divide the full
    tensor and preserve complete slice/concat dimensions. External users are
    preserved as additional loop results. Each DAG node is cloned exactly once.

    Unsupported plans are a successful no-match, returning the original loop.
    Invalid targets are diagnosed. The loop handle is consumed; nested handles
    must be reacquired from the returned loop.
    """

    target: ext.Operand[transform.AnyOpType]
    result: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(op, rewriter, results, state):
            targets = state.get_payload_ops(op.target)
            if len(targets) != 1 or targets[0].operation.name != "scf.forall":
                op.location.emit_error("fuse_forall_epilogue expects one scf.forall")
                return DiagnosedSilenceableFailure.DefiniteFailure
            loop = targets[0].opview
            plan = _plan(loop)
            result = loop.operation if plan is None else _rewrite(loop, plan, rewriter)
            results.set_ops(op.result, [result])
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op):
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op):
            return (
                transform.consumes_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def fuse_forall_epilogue(target: ir.Value[transform.AnyOpType]) -> ir.Value:
    return FuseForallEpilogueOp(target=target).result
