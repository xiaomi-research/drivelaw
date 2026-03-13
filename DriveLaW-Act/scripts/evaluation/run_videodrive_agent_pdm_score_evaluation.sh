set -x
TRAIN_TEST_SPLIT=navtest
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/path/to/dataset/maps"
export NAVSIM_EXP_ROOT="/path/to/DriveLaW-Act"
export NAVSIM_DEVKIT_ROOT="/path/to/DriveLaW-Act"
export OPENSCENE_DATA_ROOT="/path/to/dataset"
# Ensure navsim package is importable (fixes "Error locating target VideoDriveAgent")
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH}"
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_TIMEOUT=1800000

torchrun --nproc_per_node=$RESOURCE_GPU \
         --nnodes=$WORLD_SIZE \
         --node_rank=$RANK \
         --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_videodrive.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=videodrive_agent \
    agent.config_file="/path/to/DriveLaW-Act/navsim/agents/videodrive/configs/ltx_model/video_model_infer_navsim_eval.yaml" \
    cache_path="" \
    use_cache_without_dataset=False \
    experiment_name=videodrive_agent_eval 

