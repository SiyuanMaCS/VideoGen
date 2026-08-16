#!/usr/bin/env bash
set -euo pipefail

export MASTER_PORT=$(shuf -i 20000-65535 -n 1)

# ==============================================================================
# Local Inference Script: GigaWorld
# Usage:
#   bash scripts_inference/run_gigaworld.sh <DATASET> <TEST_NAME> [MODEL_NAME] [N_PROC] [DATA_ROOT] [GPU_IDS] [N_ATTEMPTS]
#
# Example:
#   bash scripts_inference/run_gigaworld.sh open_x_embodiment gigaworld GigaWorld 1 ./data 0 1
# ==============================================================================

DATASET=${1:-"debug"}
TEST_NAME=${2:-"gigaworld"}
MODEL_NAME=${3:-"GigaWorld"}
N_PROC=${4:-1}
DATA_ROOT=${5:-"./data"}
GPU_IDS=${6:-"0"}
N_ATTEMPTS=${7:-1}

PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"${PROJECT_ROOT}/checkpoints"}
THIRD_PARTY_ROOT=${THIRD_PARTY_ROOT:-"${PROJECT_ROOT}/third_party"}
PYTHON_BIN=${PYTHON_BIN:-python}

GIGAWORLD_ROOT=${GIGAWORLD_ROOT:-"${THIRD_PARTY_ROOT}/GigaWorld"}
GIGAWORLD_CKPT=${GIGAWORLD_CKPT:-"${CHECKPOINT_ROOT}/GigaWorld/GigaWorld-0-Video-GR1-2b"}
GIGAWORLD_EXTRA_ARGS=${GIGAWORLD_EXTRA_ARGS:-""}

GT_JSON="${DATA_ROOT}/${DATASET}/summary.json"
PRED_ROOT="${DATA_ROOT}/${DATASET}/generated_data/${TEST_NAME}"

if [ ! -f "${GT_JSON}" ]; then
    echo "Error: summary JSON not found: ${GT_JSON}"
    exit 1
fi

if [ ! -d "${GIGAWORLD_ROOT}" ]; then
    echo "Error: GigaWorld repo not found: ${GIGAWORLD_ROOT}"
    exit 1
fi

if [ ! -d "${GIGAWORLD_CKPT}" ]; then
    echo "Error: GigaWorld checkpoint not found: ${GIGAWORLD_CKPT}"
    exit 1
fi

mkdir -p "${PRED_ROOT}"
mkdir -p logs

IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"

echo "------------------------------------------------"
echo "Starting GigaWorld local inference"
echo "Dataset      : ${DATASET}"
echo "Test name    : ${TEST_NAME}"
echo "Model name   : ${MODEL_NAME}"
echo "Task list    : ${GT_JSON}"
echo "Pred root    : ${PRED_ROOT}"
echo "GigaWorld root: ${GIGAWORLD_ROOT}"
echo "Checkpoint   : ${GIGAWORLD_CKPT}"
echo "Extra args   : ${GIGAWORLD_EXTRA_ARGS}"
echo "Processes    : ${N_PROC}"
echo "GPU ids      : ${GPU_IDS}"
echo "Attempts     : ${N_ATTEMPTS}"
echo "------------------------------------------------"

PIDS=()

for ((RANK=0; RANK<N_PROC; RANK++)); do
    GPU=${GPU_ARRAY[$((RANK % ${#GPU_ARRAY[@]}))]}
    LOG_FILE="logs/${TEST_NAME}_rank${RANK}.log"

    (
        export CUDA_VISIBLE_DEVICES="${GPU}"

        "${PYTHON_BIN}" run_local.py \
            --model gigaworld \
            --gt_root "${GT_JSON}" \
            --pred_root "${PRED_ROOT}" \
            --checkpoint_folder "${GIGAWORLD_CKPT}" \
            --gigaworld_root "${GIGAWORLD_ROOT}" \
            --gigaworld_extra_args "${GIGAWORLD_EXTRA_ARGS}" \
            --n_attempts "${N_ATTEMPTS}" \
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
echo "GigaWorld inference finished. Output: ${PRED_ROOT}"
echo "------------------------------------------------"

exit "${STATUS}"