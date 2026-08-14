#!/usr/bin/env bash
set -euo pipefail

export MASTER_PORT=$(shuf -i 20000-65535 -n 1)

# ==============================================================================
# Local Inference Script: Cosmos3 (Nano / Super) — image2video mode
# Usage:
#   bash scripts_inference/run_cosmos3.sh <DATASET> <TEST_NAME> [CHECKPOINT] [N_PROC] [DATA_ROOT] [GPU_IDS] [N_ATTEMPTS] [KEEP_PROB]
#
# Examples:
#   # Cosmos3-Nano, 8 parallel processes (1 GPU each), keep_prob=0.8
#   bash scripts_inference/run_cosmos3.sh open_x_embodiment cosmos3_nano Cosmos3-Nano 8 ./data 0,1,2,3,4,5,6,7 1 0.8
#
#   # Cosmos3-Super, 1 process (uses 8 GPUs internally via torchrun), keep_prob=0.6
#   bash scripts_inference/run_cosmos3.sh open_x_embodiment cosmos3_super Cosmos3-Super 1 ./data 0,1,2,3,4,5,6,7 1 0.6
# ==============================================================================

DATASET=${1:-"debug"}
TEST_NAME=${2:-"cosmos3_nano"}
CHECKPOINT=${3:-"Cosmos3-Nano"}
N_PROC=${4:-1}
DATA_ROOT=${5:-"./data"}
GPU_IDS=${6:-"0"}
N_ATTEMPTS=${7:-1}
KEEP_PROB=${8:-1.0}

PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
THIRD_PARTY_ROOT=${THIRD_PARTY_ROOT:-"${PROJECT_ROOT}/third_party"}
COSMOS3_ROOT=${COSMOS3_ROOT:-"${THIRD_PARTY_ROOT}/cosmos-framework"}

# Determine GPUs per process based on checkpoint
if [[ "${CHECKPOINT}" == *"Super"* ]]; then
    # Super needs multiple GPUs per process
    COSMOS3_NUM_GPUS=${COSMOS3_NUM_GPUS:-8}
    PARALLELISM="throughput"
else
    # Nano: 1 GPU per process
    COSMOS3_NUM_GPUS=${COSMOS3_NUM_GPUS:-1}
    PARALLELISM="throughput"
fi

GT_JSON="${DATA_ROOT}/${DATASET}/summary.json"
PRED_ROOT="${DATA_ROOT}/${DATASET}/generated_data/${TEST_NAME}"

if [ ! -f "${GT_JSON}" ]; then
    echo "Error: summary JSON not found: ${GT_JSON}"
    exit 1
fi

if [ ! -d "${COSMOS3_ROOT}" ]; then
    echo "Error: cosmos-framework not found: ${COSMOS3_ROOT}"
    exit 1
fi

mkdir -p "${PRED_ROOT}"
mkdir -p logs

IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"

echo "------------------------------------------------"
echo "Starting Cosmos3 local inference"
echo "Dataset       : ${DATASET}"
echo "Test name     : ${TEST_NAME}"
echo "Checkpoint    : ${CHECKPOINT}"
echo "Task list     : ${GT_JSON}"
echo "Pred root     : ${PRED_ROOT}"
echo "Cosmos3 root  : ${COSMOS3_ROOT}"
echo "Processes     : ${N_PROC}"
echo "GPU ids       : ${GPU_IDS}"
echo "GPUs/process  : ${COSMOS3_NUM_GPUS}"
echo "Parallelism   : ${PARALLELISM}"
echo "Attempts      : ${N_ATTEMPTS}"
echo "Keep prob     : ${KEEP_PROB}"
echo "------------------------------------------------"

PIDS=()

for ((RANK=0; RANK<N_PROC; RANK++)); do
    if [[ "${CHECKPOINT}" == *"Super"* ]]; then
        # Super: all GPUs visible to the single process
        GPU="${GPU_IDS}"
    else
        # Nano: round-robin 1 GPU per process
        GPU=${GPU_ARRAY[$((RANK % ${#GPU_ARRAY[@]}))]}
    fi
    LOG_FILE="logs/${TEST_NAME}_rank${RANK}.log"

    (
        export CUDA_VISIBLE_DEVICES="${GPU}"

        "${PYTHON_BIN}" run_local.py \
            --model cosmos3 \
            --gt_root "${GT_JSON}" \
            --pred_root "${PRED_ROOT}" \
            --checkpoint_folder "unused" \
            --cosmos3_root "${COSMOS3_ROOT}" \
            --cosmos3_checkpoint "${CHECKPOINT}" \
            --cosmos3_num_gpus "${COSMOS3_NUM_GPUS}" \
            --cosmos3_parallelism "${PARALLELISM}" \
            --n_attempts "${N_ATTEMPTS}" \
            --sample_keep_prob "${KEEP_PROB}" \
            --rank "${RANK}" \
            --world_size "${N_PROC}" \
            --gpu 0
    ) 2>&1 | tee "${LOG_FILE}" &

    PIDS+=($!)
done

STATUS=0
for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
        STATUS=1
    fi
done

echo "------------------------------------------------"
echo "Cosmos3 inference finished. Output: ${PRED_ROOT}"
echo "------------------------------------------------"

exit "${STATUS}"
