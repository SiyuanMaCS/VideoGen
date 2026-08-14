#!/usr/bin/env bash
set -euo pipefail

export MASTER_PORT=$(shuf -i 20000-65535 -n 1)

# ==============================================================================
# Local Inference Script: WoW
# Usage:
#   bash scripts_inference/run_wow.sh <DATASET> <TEST_NAME> [N_PROC] [DATA_ROOT] [GPU_IDS] [N_ATTEMPTS] [NUM_FRAMES] [STEPS]
#
# Example:
#   bash scripts_inference/run_wow.sh open_x_embodiment wow 1 ./data 0 1 49 50
# ==============================================================================

DATASET=${1:-"debug"}
TEST_NAME=${2:-"wow"}
N_PROC=${3:-1}
DATA_ROOT=${4:-"./data"}
GPU_IDS=${5:-"0"}
N_ATTEMPTS=${6:-1}
NUM_FRAMES=${7:-49}
STEPS=${8:-50}

PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"${PROJECT_ROOT}/checkpoints"}
THIRD_PARTY_ROOT=${THIRD_PARTY_ROOT:-"${PROJECT_ROOT}/third_party"}
PYTHON_BIN=${PYTHON_BIN:-python}

WOW_ROOT=${WOW_ROOT:-"${THIRD_PARTY_ROOT}/WoW"}
WOW_CKPT=${WOW_CKPT:-"${CHECKPOINT_ROOT}/WoW/WoW-1-Wan-14B-2M"}
CUSTOM_CHECKPOINT=${CUSTOM_CHECKPOINT:-"WoW_video_dit.pt"}
PERSISTENT_PARAM_GB=${PERSISTENT_PARAM_GB:-70}

GT_JSON="${DATA_ROOT}/${DATASET}/summary.json"
PRED_ROOT="${DATA_ROOT}/${DATASET}/generated_data/${TEST_NAME}"

if [ ! -f "${GT_JSON}" ]; then
    echo "Error: summary JSON not found: ${GT_JSON}"
    exit 1
fi

if [ ! -d "${WOW_ROOT}" ]; then
    echo "Error: WoW repo not found: ${WOW_ROOT}"
    exit 1
fi

if [ ! -d "${WOW_CKPT}" ]; then
    echo "Error: WoW checkpoint not found: ${WOW_CKPT}"
    exit 1
fi

mkdir -p "${PRED_ROOT}"
mkdir -p logs

IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"

echo "------------------------------------------------"
echo "Starting WoW local inference"
echo "Dataset      : ${DATASET}"
echo "Test name    : ${TEST_NAME}"
echo "Task list    : ${GT_JSON}"
echo "Pred root    : ${PRED_ROOT}"
echo "WoW root     : ${WOW_ROOT}"
echo "WoW ckpt     : ${WOW_CKPT}"
echo "Processes    : ${N_PROC}"
echo "GPU ids      : ${GPU_IDS}"
echo "Attempts     : ${N_ATTEMPTS}"
echo "Num frames   : ${NUM_FRAMES}"
echo "Steps        : ${STEPS}"
echo "------------------------------------------------"

PIDS=()

for ((RANK=0; RANK<N_PROC; RANK++)); do
    GPU=${GPU_ARRAY[$((RANK % ${#GPU_ARRAY[@]}))]}
    LOG_FILE="logs/${TEST_NAME}_rank${RANK}.log"

    (
        export CUDA_VISIBLE_DEVICES="${GPU}"

        "${PYTHON_BIN}" run_local.py \
            --model wow \
            --gt_root "${GT_JSON}" \
            --pred_root "${PRED_ROOT}" \
            --checkpoint_folder "${WOW_CKPT}" \
            --wow_root "${WOW_ROOT}" \
            --custom_checkpoint "${CUSTOM_CHECKPOINT}" \
            --persistent_param_gb "${PERSISTENT_PARAM_GB}" \
            --steps "${STEPS}" \
            --num_frames "${NUM_FRAMES}" \
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
echo "WoW inference finished. Output: ${PRED_ROOT}"
echo "------------------------------------------------"

exit "${STATUS}"