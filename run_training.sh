#!/bin/bash
# Number of GPUs per node
GPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# Paths for saving checkpoints and tokenizer files.
CHECKPOINT_PATH=/workspace/Megatron-LM/experiments/codeparrot-small
VOCAB_FILE=/workspace/Megatron-LM/vocab.json
MERGE_FILE=/workspace/Megatron-LM/merges.txt
DATA_PATH=/workspace/Megatron-LM/codeparrot_content_document

# Model and training hyperparameters.
# original train-iters: 150000,  lr-deccay-iters: 150000,  lr-warmup-iters: 2000, save-interval: 2000, eval-interval: 200
GPT_ARGS="--num-layers 12 \
--hidden-size 768 \
--num-attention-heads 12 \
--seq-length 1024 \
--max-position-embeddings 1024 \
--micro-batch-size 12 \
--global-batch-size 192 \
--lr 0.0005 \
--train-iters 250 \
--lr-decay-iters 250 \
--lr-decay-style cosine \
--lr-warmup-iters 20 \
--weight-decay 0.1 \
--adam-beta2 0.999 \
--fp16 \
--log-interval 10 \
--save-interval 100 \
--eval-interval 100 \
--eval-iters 10"

TENSORBOARD_ARGS="--tensorboard-dir /workspace/megatron-lm/experiments/tensorboard"

# Launch training with torch.distributed.launch
python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
        pretrain_gpt.py \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        $GPT_ARGS \
        --vocab-file /workspace/Megatron-LM/vocab.json \
        --merge-file /workspace/Megatron-LM/merges.txt \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        $TENSORBOARD_ARGS
