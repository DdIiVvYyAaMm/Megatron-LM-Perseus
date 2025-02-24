# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
from functools import partial
from contextlib import nullcontext
import inspect
import zeus
from zeus.optimizer.pipeline_frequency import PipelineFrequencyOptimizer
from PFO.monkey import instrument_megatron
from megatron.core.utils import get_model_config

from typing import List, Optional, Tuple, Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.rerun_state_machine import get_rerun_state_machine
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)

# Straggler detector
stimer = StragglerDetector()

# Pipeline Frequency Optimizer
pfo_initialized = False

import torch.distributed as dist

class DummyPipelineFrequencyOptimizer:
    """A no-op version of PipelineFrequencyOptimizer that does not hit any server."""

    def __init__(
        self,
        rank: int,
        dp_rank: int,
        pp_rank: int,
        tp_rank: int,
        device_id: int,
        dp_degree: int,
        pp_degree: int,
        tp_degree: int,
        world_size: int,
        server_url: str,      # We'll ignore this
        job_metadata: str|None = None,
    ) -> None:
        # ----------------------------
        # 1) Store whatever you need
        # ----------------------------
        self.rank = rank
        self.dp_rank = dp_rank
        self.pp_rank = pp_rank
        self.tp_rank = tp_rank
        self.device_id = device_id
        self.dp_degree = dp_degree
        self.pp_degree = pp_degree
        self.tp_degree = tp_degree
        self.world_size = world_size
        self.server_url = server_url       # No-op usage
        self.job_metadata = job_metadata

        # ----------------------------
        # 2) Fake the server logic
        # ----------------------------
        # For example, you might store a dummy job_id and skip actual registration:
        self.job_id = "dummy_job_id"

        # If your pipeline hooks rely on the "frequency_controller" existing, 
        # you can either create a real one or a fake one. Let's do a trivial stub:
        self.frequency_controller = None

        # If your pipeline code calls "self._get_frequency_schedule()" or 
        # references "self.freq_schedule", define it here:
        self.freq_schedule = []
        self.freq_schedule_iter = iter(self.freq_schedule)

        print(f"[DummyPipelineFrequencyOptimizer] Initialized on rank={rank} with NO server calls.")

    # ---------------------------------
    # 3) Implement or override methods
    # ---------------------------------
    def on_step_begin(self):
        """Called at the start of each pipeline 'step'."""
        # do nothing (or your test logic)
        print(f"[DummyPFO rank={self.rank}] on_step_begin called")

    def on_instruction_begin(self, stage_name: str):
        """Called at the start of each forward/backward instruction."""
        print(f"[DummyPFO rank={self.rank}] on_instruction_begin: {stage_name}")

    def on_instruction_end(self, stage_name: str):
        """Called at the end of each forward/backward instruction."""
        print(f"[DummyPFO rank={self.rank}] on_instruction_end: {stage_name}")

    def on_step_end(self):
        """Called at the end of each pipeline 'step'."""
        print(f"[DummyPFO rank={self.rank}] on_step_end called")


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    global pfo_initialized
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,

            # record stack information for the trace events
            trace_alloc_record_context=True)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else: # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except:
                raise RuntimeError("--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")

        with build_model_context(**build_model_context_args):
            model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
                rotary_base=args.rotary_base,
                rope_scaling=args.use_rope_scaling
            )

    if not pfo_initialized:
        pass
        # job_metadata = args.model_type
        # "+".join([
        #     args.model_type,
        #     # args.train_schedule,
        #     # f"mbs{args.micro_batch_size}",
        #     # f"nmb{args.gradient_accumulation_steps}",
        # ])

        # pfo = PipelineFrequencyOptimizer(
        #     rank=torch.distributed.get_rank(),
        #     dp_rank=mpu.get_data_parallel_rank(),  # utils.py has mpu.get_data_parallel_rank, and mpu = parallel_state, which is defined in parallel_state.py at Megatron/Core 
        #     pp_rank=mpu.get_pipeline_model_parallel_rank(),
        #     tp_rank=mpu.get_expert_tensor_parallel_rank(), # Cant find slice parallel rank !!!! but there is get_pipeline_model_split_parallel_rank
        #     device_id=torch.cuda.current_device(),
        #     dp_degree=mpu.get_data_parallel_world_size(),
        #     pp_degree=mpu.get_pipeline_model_parallel_world_size(),
        #     tp_degree=mpu.get_expert_tensor_parallel_world_size(),
        #     world_size=torch.distributed.get_world_size(),
        #     server_url=args.pfo_server_url,
        #     job_metadata=None,
        # )

        pfo = DummyPipelineFrequencyOptimizer(
            rank=torch.distributed.get_rank(),
            dp_rank=mpu.get_data_parallel_rank(),
            pp_rank=mpu.get_pipeline_model_parallel_rank(),
            tp_rank=mpu.get_expert_tensor_parallel_rank(),
            device_id=torch.cuda.current_device(),
            dp_degree=mpu.get_data_parallel_world_size(),
            pp_degree=mpu.get_pipeline_model_parallel_world_size(),
            tp_degree=mpu.get_expert_tensor_parallel_world_size(),
            world_size=torch.distributed.get_world_size(),
            server_url="http://fake-url",
            job_metadata=None,
        )

        instrument_megatron(pfo)
        pfo_initialized = True
        print_rank_0(
            f"PFO initialized - DP{pfo.dp_rank} PP{pfo.pp_rank} TP{pfo.tp_rank} "
            f"on device {pfo.device_id}"
        )
        
        # Get the Megatron config object for the newly created model
        config = get_model_config(model)

        # Attach pfo as a field on the config
        config.pfo = pfo

    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=False,
        )
    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        {'lm loss': (reporting_loss[0], reporting_loss[1])},
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator)
    timers('batch-generator').stop()
    # Where to put on_step_begin() and end????
    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        s3_cache_path=args.s3_cache_path,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
