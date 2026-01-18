export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${NAVSIM_MAPS_ROOT:-./NAVSIM/dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-./NAVSIM/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-./NAVSIM/navsim-main}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-./NAVSIM/dataset}"
TRAIN_TEST_SPLIT=navtrain
GPUS=${IDP_N_GPU:-8}
NNODES=${IDP_N_NODES:-1}
NODE_RANK=${IDP_N_RANK:-0}
MASTER_ADDR=${IDP_MASTER_ADDR:-localhost}

export PORT=29500

export LOCAL_RANK=$NODE_RANK
export NCCL_DEBUG=NONE
export NCCL_NET_PLUGIN=none
export NCCL_SOCKET_NTHREADS=8
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export UCX_NET_DEVICES=bond0
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=13
export NCCL_IB_GID_INDEX=3
export PYTHONPATH="$(pwd):${PYTHONPATH}"
export HYDRA_FULL_ERROR=1 

echo "CONFIG: $CONFIG"
echo "GPUS: $GPUS"
echo "PORT: $PORT"
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "MASTER_ADDR: $MASTER_ADDR"


DISTRIBUTED_ARGS="--nproc_per_node $GPUS --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $PORT "

torchrun $DISTRIBUTED_ARGS \
    navsim/agents/videodrive/run_training_videodrive_ema.py \
    agent=videodrive_agent \
    agent.config_file="navsim/agents/videodrive/configs/ltx_model/video_model_infer_navsim_ema.yaml" \
    experiment_name=training_video_agent \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path="${CACHE_PATH:-./cache/videodrive_agent_cache_dir_train}" \
    use_cache_without_dataset=False > train_drivelaw_act_ema.txt 2>&1