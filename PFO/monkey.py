from typing import Any
from zeus.optimizer.pipeline_frequency import PipelineFrequencyOptimizer
from megatron.core import mpu

def instrument_megatron(pfo) -> None:
    """Monkey-patch Megatron-LM's pipeline schedule to integrate frequency optimization.
    
    Args:
        pfo: Initialized PipelineFrequencyOptimizer instance
    """
    import megatron.core.pipeline_parallel.schedules as megatron_schedule
    import megatron.training.training as megatron_training

    # Save original functions
    orig_training = megatron_training.train
    original_forward_step = megatron_schedule.forward_step
    original_backward_step = megatron_schedule.backward_step

    def wrapped_forward_step(*args: Any, **kwargs: Any) -> Any:
        """Wrapped forward pass with frequency instrumentation."""

        print(f"[DEBUG] Forward hook triggered on rank{mpu.get_data_parallel_rank()}")
        pfo.on_instruction_begin("forward")
        result = original_forward_step(*args, **kwargs)
        pfo.on_instruction_end("forward")
        return result

    def wrapped_backward_step(*args: Any, **kwargs: Any) -> Any:
        """Wrapped backward pass with frequency instrumentation."""
        
        print(f"[DEBUG] Backward hook triggered on rank {mpu.get_data_parallel_rank()}\n")
        pfo.on_instruction_begin("backward")
        result = original_backward_step(*args, **kwargs) 
        pfo.on_instruction_end("backward")
        return result
    
    def train_and_send_payload(*args, **kwargs):
        try:
            # run the normal megatron training loop
            return orig_training(*args, **kwargs)
        finally:
            # no matter how train() exits – clean up and send final payloads
            print("[DEBUG] on_train_end – sending timing/energy", flush=True)
            pfo.on_train_end()
    
   
    import megatron.core.pipeline_parallel as pipeline_parallel
    orig_get_fb = pipeline_parallel.get_forward_backward_func
    orig_get_fb_training = megatron_training.get_forward_backward_func

    def get_fb_hooked_for_training():
        fb = orig_get_fb_training()

        def fb_wrapped(*args, **kwargs):
            rank = mpu.get_data_parallel_rank()
            print(f"[DEBUG] on_step_begin hook on DP rank {rank}", flush=True)
            pfo.on_step_begin()
            try:
                return fb(*args, **kwargs)
            finally:
                print(f"[DEBUG] on_step_end hook on DP rank {rank}", flush=True)
                pfo.on_step_end()
        return fb_wrapped

    # Monkey-patch Megatron's schedule
    megatron_training.get_forward_backward_func = get_fb_hooked_for_training
    pipeline_parallel.get_forward_backward_func = get_fb_hooked_for_training
    megatron_training.train = train_and_send_payload
    megatron_schedule.forward_step = wrapped_forward_step
    megatron_schedule.backward_step = wrapped_backward_step
    