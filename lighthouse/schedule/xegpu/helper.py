"""Helper functions for XeGPU scheduling."""

from mlir import ir
from mlir.dialects import transform

from lighthouse.pipeline.helper import apply_registered_pass, PipelineInterrupt


def bundle_xegpu_to_binary(
    mod: ir.Value, stop_at_stage: str = ""
) -> ir.Value[transform.AnyOpType]:
    """Schedule for lowering xegpu wg level to binary."""
    # upstream xegpu/xevm pipeline is payload independent.
    mod = apply_registered_pass(
        mod, "gpu-lower-to-xevm-pipeline", options={"xegpu-op-level": "workgroup"}
    )

    if stop_at_stage == "final":
        raise PipelineInterrupt()

    return mod
