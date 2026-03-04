#!/bin/bash
export NCCL_DEBUG=WARN
LOG_FILE=""
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_TIMEOUT=1800000

torchrun --nproc_per_node=$RESOURCE_GPU \
         --nnodes=$WORLD_SIZE \
         --node_rank=$RANK \
         --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    scripts/train.py configs/drivelaw_video.yaml 2>&1 | tee $LOG_FILE
