# megatron.core.pipeline_parallel.schedule
def forward_step(...):
    pass

def backward_step(...):
    pass


# ----------------------------# ----------------------------# ----------------------------#


# zeus.optimizer.pipeline_frequency.contrib.megatron
def instrument_megatron(pfo: PipelineFrequencyOptimizer) -> None:
    import megatron.core.pipeline_parallel.schedule as megatron_schedule

    def forward_step(...):
        # pfo.on_forward_step()
        pfo.on_instruction_begin('forward')
        
        megatron_schedule.forward_step(...)
        
        pfo.on_instruction_end('forward')

    def backward_step(...):
        # pfo.on_backward_step()
        pfo.on_instruction_begin('backward')
        
        megatron_schedule.backward_step(...)
        
        pfo.on_instruction_end('backward')




    megatron_schedule.forward_step = forward_step
    megatron_schedule.backward_step = backward_step