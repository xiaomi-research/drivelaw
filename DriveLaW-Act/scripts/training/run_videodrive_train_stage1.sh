export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/path/to/NAVSIM/dataset/maps"
export NAVSIM_EXP_ROOT="/path/to/NAVSIM/exp"
export NAVSIM_DEVKIT_ROOT="/path/to/NAVSIM/navsim-main"
export OPENSCENE_DATA_ROOT="/path/to/NAVSIM/dataset"
TRAIN_TEST_SPLIT=navtrain
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_TIMEOUT=1800000

torchrun --nproc_per_node=$RESOURCE_GPU \
         --nnodes=$WORLD_SIZE \
         --node_rank=$RANK \
         --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $NAVSIM_DEVKIT_ROOT/navsim/agents/videodrive/run_training_videodrive_stage1.py \
    agent=videodrive_agent \
    agent.config_file="/path/to/navsim/agents/videodrive/configs/ltx_model/video_model_infer_navsim_stage1.yaml" \
    experiment_name=training_video_agent \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path='/path/to/exp/videodrive_agent_cache_dir_final_front' \
    use_cache_without_dataset=False 


    