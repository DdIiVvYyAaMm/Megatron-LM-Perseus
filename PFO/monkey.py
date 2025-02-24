from typing import Any
from zeus.optimizer.pipeline_frequency import PipelineFrequencyOptimizer

# the calls below were working! tested with DummyPipelineFrequencyOptimizer

def instrument_megatron(pfo: PipelineFrequencyOptimizer) -> None:
    """Monkey-patch Megatron-LM's pipeline schedule to integrate frequency optimization.
    
    Args:
        pfo: Initialized PipelineFrequencyOptimizer instance
    """
    import megatron.core.pipeline_parallel.schedules as megatron_schedule

    # Save original functions
    original_forward_step = megatron_schedule.forward_step
    original_backward_step = megatron_schedule.backward_step

    def wrapped_forward_step(*args: Any, **kwargs: Any) -> Any:
        """Wrapped forward pass with frequency instrumentation."""

        print("[DEBUG] Forward hook triggered on rank")
        pfo.on_instruction_begin("forward")
        result = original_forward_step(*args, **kwargs)
        pfo.on_instruction_end("forward")
        return result

    def wrapped_backward_step(*args: Any, **kwargs: Any) -> Any:
        """Wrapped backward pass with frequency instrumentation."""
        
        print("[DEBUG] Backward hook triggered on rank {mpu.get_global_rank()}")
        pfo.on_instruction_begin("backward")
        result = original_backward_step(*args, **kwargs) 
        pfo.on_instruction_end("backward")
        return result

    # Monkey-patch Megatron's schedule
    megatron_schedule.forward_step = wrapped_forward_step
    megatron_schedule.backward_step = wrapped_backward_step