"""Generate MLIR transform schedule that tiles the payload for WG and reduction dims."""

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured, tensor
import lighthouse.transform as lh_transform
from lighthouse.pipeline.helper import (
    apply_registered_pass,
    match,
)
from lighthouse.schedule import schedule_boilerplate
from lighthouse.dialects.transform import transform_ext
from .attention_fusion import transform_to_flash_attention
from .lowering_common import get_payload_func
from lighthouse.dialects.transform.transform_ext.utils.subgraph_analysis import (
    SUBGRAPH_ANCHOR_ATTR_NAME,
    SUBGRAPH_KIND_ATTR_NAME,
)


def _subgraph_anchors(func: ir.Value, kind: str) -> ir.Value:
    """Match the anchors of the subgraphs of `kind` tagged by `tag_subgraphs`."""
    return match(
        func,
        op_attrs={
            SUBGRAPH_KIND_ATTR_NAME: ir.StringAttr.get(kind),
            SUBGRAPH_ANCHOR_ATTR_NAME: ir.UnitAttr.get(),
        },
    )


def _tile_subgraph(
    anchor: ir.Value,
    producers: ir.Value,
    consumers: ir.Value,
    wg_tile: list | ir.Value,
) -> ir.Value:
    """Tile a subgraph anchor into an scf.forall and fuse the other ops of the
    subgraph into it: producers backward, consumers forward. Returns the loop."""
    anytype = transform.AnyOpType.get()
    _, wg_loop = structured.TileUsingForallOp(anchor, tile_sizes=wg_tile).results
    _, wg_loop = structured.structured_fuse_into_containing_op(
        anytype, anytype, producer_op=producers, containing_op=wg_loop
    )
    return transform_ext.fuse_forall_epilogue(wg_loop, consumers)


def apply_subgraph_tiling(
    func: ir.Operation,
    force_wg_tile: list[int] | None = None,
    force_k_tile: int | None = None,
    device: str | None = None,
) -> ir.Operation:
    """Partition reduction, GEMM and elementwise subgraphs and tile each into
    its own workgroup-parallel scf.forall, i.e. its own kernel.

    Normalization reductions are claimed before GEMM to materialize their
    outputs once, including when shared by multiple projections.
    GEMM subgraphs then claim their epilogue chains; the
    remaining elementwise ops form separate subgraphs. Pure elementwise graphs
    are also supported. Fails silenceably if a linalg op is left unpartitioned.
    """

    func = transform_ext.tag_subgraphs(func, kinds=["reduction", "gemm", "elementwise"])

    with lh_transform.foreach(_subgraph_anchors(func, "reduction")) as anchor:
        # Standalone reductions keep the existing schedule and its overrides.
        transform_ext.extract_handle(
            lh_transform.match_op(func, "linalg.matmul"), 0, silenceable=True
        )
        producers, _ = transform_ext.get_subgraph_ops(anchor)
        reduction = transform_ext.filter_reduction_ops(producers)
        wg_tile, _, reduction_tile = transform_ext.infer_xegpu_reduction_params(
            reduction
        )
        _, [wg_loop], _ = lh_transform.tile(
            anchor,
            tile_sizes=wg_tile,
            fuse_producers=True,
            use_forall=True,
            apply_cleanup=False,
        )
        lh_transform.cleanup(wg_loop)
        wg_loop = transform_ext.fuse_elementwise_region(wg_loop)
        lh_transform.cleanup(wg_loop)
        _tile_reduction_body(wg_loop, reduction_tile)
        transform.yield_()
    transform.apply_dce(func)

    with lh_transform.foreach(_subgraph_anchors(func, "gemm")) as anchor:
        producers, consumers = transform_ext.get_subgraph_ops(anchor)
        matmul_op = transform_ext.filter_by_name(
            transform.merge_handles([anchor, producers]), "linalg.matmul"
        )
        params = transform_ext.infer_xegpu_gemm_params(
            matmul_op,
            force_wg_tile=force_wg_tile,
            force_k_tile=force_k_tile,
            device=device,
        )
        wg_m = transform_ext.get_param_dict_entry(params, "wg_m")
        wg_n = transform_ext.get_param_dict_entry(params, "wg_n")
        k_tile = transform_ext.get_param_dict_entry(params, "k_tile")

        wg_loop = _tile_subgraph(anchor, producers, consumers, [wg_m, wg_n])

        # k loop tiling
        wg_matmul = match(wg_loop, ops={"linalg.matmul"})
        lh_transform.tile(
            wg_matmul,
            tile_sizes=[0, 0, k_tile],
            fuse_producers=True,
        )
        lh_transform.cleanup(wg_loop)
        transform.yield_()
    transform.apply_dce(func)

    with lh_transform.foreach(_subgraph_anchors(func, "elementwise")) as anchor:
        producers, consumers = transform_ext.get_subgraph_ops(anchor)
        wg_tile = transform_ext.infer_xegpu_elementwise_params(anchor)
        wg_loop = _tile_subgraph(anchor, producers, consumers, wg_tile)
        lh_transform.cleanup(wg_loop)
        transform.yield_()

    transform_ext.clear_subgraph_tags(func)
    func = transform_ext.split_concat_output(func)
    lh_transform.cleanup(func)
    return func


def _tile_reduction_body(wg_loop, reduction_tile):
    """Tile a normalization kernel's output and reductions within its WG."""
    anytype = transform.AnyOpType.get()

    def fuse_producers(target, loop):
        producers = transform_ext.filter_by_name(
            transform_ext.filter_elementwise(transform_ext.trace_producers(target)),
            "linalg.generic",
        )
        _, loop = structured.structured_fuse_into_containing_op(
            anytype, anytype, producer_op=producers, containing_op=loop
        )
        return loop

    generic_ops = match(wg_loop, ops={"linalg.generic"})
    leaf = transform_ext.extract_handle(
        transform_ext.filter_elementwise(generic_ops), -1, silenceable=True
    )
    reductions = transform_ext.reverse_handles(
        transform_ext.filter_reduction_ops(generic_ops)
    )
    res = structured.TileUsingForOp(leaf, sizes=reduction_tile).results
    loop = fuse_producers(res[0], res[1])
    transform_ext.sink_extract_slice_into_loop(loop)
    with lh_transform.foreach(reductions) as reduction:
        tiled, loops, _ = lh_transform.tile(
            reduction,
            tile_sizes=reduction_tile,
            fuse_producers=False,
            use_forall=False,
            apply_cleanup=False,
        )
        fuse_producers(tiled, loops[0])
        transform.apply_dce(wg_loop)
        transform.yield_()


def apply_reduction_tiling(
    func: ir.Operation,
    force_wg_tile: list[int] | None = None,
    force_wg_subtile: list[int] | None = None,
    force_reduction_tile: list[int] | None = None,
) -> ir.Operation:
    """Apply reduction tiling to the given function."""

    anytype = transform.AnyOpType.get()

    # Fuse elementwise ops first to simplify the graph.
    func = apply_registered_pass(func, "linalg-fuse-elementwise-ops")

    # TODO implement actual pattern matching and anchor op inference
    anchor_op = lh_transform.match_op(func, "linalg.generic")
    anchor_op = transform_ext.extract_handle(anchor_op, 0, silenceable=True)
    wg_tile, _, reduction_tile = transform_ext.infer_xegpu_reduction_params(
        anchor_op,
        force_wg_tile=force_wg_tile,
        force_reduction_tile=force_reduction_tile,
    )

    # Adding persistent reduction loop is only possible via forced args for now.
    wg_subtile = None
    if force_wg_subtile is not None:
        wg_subtile = force_wg_subtile
    apply_wg_subtile = wg_subtile is not None and sum(wg_subtile) > 0

    # WG row tiling
    generic_ops = structured.structured_match(anytype, func, ops=["linalg.generic"])
    leaf_generic = transform_ext.extract_handle(generic_ops, -1, silenceable=True)
    _, [wg_loop], _ = lh_transform.tile(
        leaf_generic,
        tile_sizes=wg_tile,
        fuse_producers=True,
        use_forall=True,
        apply_cleanup=False,
    )
    lh_transform.cleanup(func)

    if apply_wg_subtile:
        # Add a persistent loop to reduce wg tile size.
        generic_ops = structured.structured_match(anytype, func, ops=["linalg.generic"])
        leaf_generic = transform_ext.extract_handle(generic_ops, -1)
        lh_transform.tile(
            leaf_generic,
            tile_sizes=wg_subtile,
            fuse_producers=True,
            use_forall=False,
            apply_cleanup=False,
        )
        lh_transform.cleanup(func)

    wg_loop = transform_ext.extract_handle(lh_transform.match_op(func, "scf.forall"), 0)
    _tile_reduction_body(wg_loop, reduction_tile)
    func = apply_registered_pass(func, "linalg-fuse-elementwise-ops")

    with ir.InsertionPoint(transform.apply_patterns(func).patterns):
        structured.apply_patterns_linalg_fold_unit_extent_dims_via_slices()

    # Cleanup after tiling and fusion.
    lh_transform.cleanup(func)
    return func


def apply_attention_tiling(
    func: ir.Operation,
    force_wg_tile: int | None = None,
    force_reduction_tile: int | None = None,
) -> ir.Operation:
    """Tile attention, including causal score producers, into one WG kernel.

    Producer fusion tiles the mask with QK scores and preserves global query
    and key offsets in linalg.index. No full score or mask tensor is stored.
    """
    anytype = transform.AnyOpType.get()

    # Clean up to fully expanded shapes by eliminating expand/collapse shape pairs.
    # Generalize all transpose ops.
    transpose_ops = lh_transform.match_op(func, "linalg.transpose")
    structured.structured_generalize(anytype, transpose_ops)
    for i in range(3):
        # Fold reshape ops by expansion and fold extract_slice into tensor.empty
        # Repeat until no more folding.
        func = transform_ext.fold_reshape_ops_by_expansion(func)
        with ir.InsertionPoint(transform.apply_patterns(func).patterns):
            tensor.apply_patterns_tensor_fold_tensor_empty(fold_single_use_only=False)
    lh_transform.cleanup(func)

    # Find the subtract and exp reduction loops and fuse them.
    subf_op = transform_ext.extract_handle(
        lh_transform.match_op(func, "arith.subf"),
        0,
        silenceable=True,
    )
    subf_loop = transform.get_parent_op(anytype, subf_op, op_name="linalg.generic")
    exp_op = transform_ext.extract_handle(
        lh_transform.match_op(func, "math.exp"),
        0,
        silenceable=True,
    )
    exp_loop = transform.get_parent_op(anytype, exp_op, op_name="linalg.generic")
    transform_ext.fuse_elementwise_op(subf_loop, exp_loop)

    lh_transform.cleanup(func)

    # Move the normalizing divide past @V to form the deferred-divide chain.
    contraction_ops = transform_ext.filter_contraction_ops(
        structured.structured_match(anytype, func, ops=["linalg.generic"])
    )
    pv_contraction = transform_ext.extract_handle(contraction_ops, -1, silenceable=True)
    transform_ext.sink_normalization_past_contraction(pv_contraction)
    lh_transform.cleanup(func)

    # Apply WG tiling.
    linalg_ops = lh_transform.match_op(func, ["linalg.generic", "linalg.batch_matmul"])
    leaf_linalg_op = transform_ext.extract_handle(linalg_ops, -1, silenceable=True)
    wg_tile, _sg_tile, reduction_tile = transform_ext.infer_xegpu_attention_params(
        leaf_linalg_op,
        force_wg_tile=force_wg_tile,
        force_reduction_tile=force_reduction_tile,
    )
    lh_transform.tile(
        leaf_linalg_op,
        tile_sizes=wg_tile,
        fuse_producers=True,
        use_forall=True,
        apply_cleanup=False,
    )
    lh_transform.cleanup(func)

    func = transform_to_flash_attention(anytype, func, reduction_tile)

    # Inline splat-constant generic inputs (e.g. masked-select zero) as
    # scalars so they are not vectorized into reads of constant tensors.
    func = transform_ext.fold_scalar_or_splat_constant(func)
    lh_transform.cleanup(func)

    # Write the result through an expanded view of the output memref so the GQA
    # collapse_shape leaves the write path and bufferization elides the output
    # buffer + copy.
    func = transform_ext.expand_output_destination(func)
    lh_transform.cleanup(func)

    return func


def wg_tiling_schedule(
    wg_tile: list[int] | None = None,
    k_tile: int | None = None,
    reduction_tile: list[int] | None = None,
    payload_func_name: str | None = None,
    wg_subtile: list[int] | None = None,
    device: str | None = None,
) -> ir.Module:
    """Tile the payload for workgroup parallelism and the reduction dimension."""

    with schedule_boilerplate() as (schedule, named_seq):
        op_names = ["linalg.generic", "linalg.matmul"]
        payload_func = get_payload_func(
            named_seq.bodyTarget,
            op_name=op_names,
            func_name=payload_func_name,
        )

        # Use alternatives op to try different lowerings. The defined regions
        # are applied in order, if one fails with a SilenceableFailure, the
        # next alternative is tried. The last region emits a definite failure
        # in case none of the patterns match.
        alt = lh_transform.alternatives(
            payload_func, num_alternatives=4, result_types=[payload_func.type]
        )
        with alt.region(0) as func:
            func = apply_subgraph_tiling(
                func,
                force_wg_tile=wg_tile,
                force_k_tile=k_tile,
                device=device,
            )
            transform.yield_([func])
        with alt.region(1) as func:
            func = apply_attention_tiling(
                func, force_wg_tile=wg_tile, force_reduction_tile=reduction_tile
            )
            transform.yield_([func])
        with alt.region(2) as func:
            func = apply_reduction_tiling(
                func,
                force_wg_tile=wg_tile,
                force_wg_subtile=wg_subtile,
                force_reduction_tile=reduction_tile,
            )
            transform.yield_([func])
        with alt.region(3) as func:
            transform_ext.emit_definite_failure(
                func, message="wg_tiling: Could not apply any of the defined patterns."
            )
            transform.yield_([func])
        payload_func = alt.results[0]

        # Fold unit dimensions.
        # Generalize named ops so unit dims in transpose ops are folded too.
        structured.structured_generalize(
            transform.AnyOpType.get(),
            lh_transform.match_op(payload_func, "linalg.transpose").results[0],
        )
        with ir.InsertionPoint(transform.apply_patterns(payload_func).patterns):
            structured.apply_patterns_linalg_fold_unit_extent_dims_via_slices()
            tensor.apply_patterns_tensor_reassociative_reshape_folding()
        lh_transform.cleanup(payload_func)

        transform.yield_()

    return schedule
