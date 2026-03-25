import os

from mlir import ir
import lighthouse.pipeline.stage as lhs
from lighthouse.pipeline.helper import import_mlir_module, remove_args_and_opts
from lighthouse.pipeline.descriptor import PipelineDescriptor


class Driver:
    """
    A simple driver that runs the optimization pipeline on a given workload.
    This is a high-level interface that abstracts away the details of the optimization pipeline,
    and provides a simple interface for running the pipeline on a given workload.

    The pipeline is flexible until the first time it is run, at which point it becomes fixed and cannot be modified until reset is called.
    This is to allow running the same pipeline on different modules, without accidentally modifying the pipeline after it has been run.

    Calling reset() will clear the pipeline and the module, allowing for a new pipeline to be constructed and run on a new module.
    """

    def __init__(self, filename: str, stages: list[str] = []):
        # The context is shared across the entire pipeline, and is used to create the PassManager and Transform Schedules.
        # The module is owned by the Driver to encapsulate its use through the pipeline.
        # It is returned at the end of run() after being transformed by the stages in the pipeline.
        self.context = ir.Context()
        self.module = None
        if filename:
            self.import_payload(filename)
        self.pipeline: list[lhs.Stage] = []
        self.pipeline_fixed = False
        self.bundles = lhs.PassBundles
        if stages:
            self.add_stages(stages)

    def import_payload(self, path: str) -> None:
        """Import the payload module and set it as the current module in the pipeline."""
        if self.module is not None:
            raise ValueError("Module already imported. Reset to start again.")
        self.module = import_mlir_module(path, self.context)

    def add_stage(self, stage_name: str) -> None:
        if self.pipeline_fixed:
            raise ValueError("Pipeline is fixed. Reset to start again.")

        # Stages can contain arguments and options, clean up for os checks
        filename = remove_args_and_opts(stage_name)

        if stage_name in self.bundles:
            # Pass Bundle
            self.pipeline.append(lhs.PassStage(self.bundles[stage_name], self.context))

        elif os.path.exists(filename):
            # Transform or YAML
            if filename.endswith(".mlir") or filename.endswith(".py"):
                self.pipeline.append(
                    lhs.TransformStage(lhs.Transform(stage_name), self.context)
                )
            elif filename.endswith(".yaml"):
                desc = PipelineDescriptor(stage_name)
                for s in desc.get_stages():
                    self.add_stage(s)
            else:
                _, ext = os.path.splitext(filename)
                raise ValueError(f"Unknown file type '{ext}' for stage '{stage_name}'.")

        else:
            # Assume random strings represent a single pass
            # Will crash later if the pass name is not registered.
            self.pipeline.append(lhs.PassStage([lhs.Pass(stage_name)], self.context))

    def add_stages(self, stages: list[str]) -> None:
        for s in stages:
            self.add_stage(s)

    def reset(self) -> None:
        """Reset the pipeline to an empty state, allowing for new stages to be added."""
        self.pipeline = list[lhs.Stage]()
        self.module = None
        self.pipeline_fixed = False

    def run(self) -> ir.Module:
        if self.module is None:
            raise ValueError("Module must not be empty.")
        if len(self.pipeline) == 0:
            raise ValueError("Pipeline must have at least one stage.")
        for stage in self.pipeline:
            self.module = stage.apply(self.module)

        # The pipeline is now fixed and cannot be modified until reset is called.
        # This is to prevent accidental modifications to the pipeline after it has been run,
        # and to ensure that different pipelines are not run on different modules.
        self.pipeline_fixed = True

        # We don't want to run the pipeline twice on the same module,
        # so we clear the module from the driver after running the pipeline,
        # and return it to the caller.
        # To use the pipeline again, the caller must import a new module into the driver.
        module = self.module
        self.module = None
        return module
