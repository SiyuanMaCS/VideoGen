#!/usr/bin/env bash
set -euo pipefail

export MASTER_PORT=$(shuf -i 20000-65535 -n 1)

# ==============================================================================
# Local Inference Script: Cosmos-Predict2.5
# Usage:
#   bash scripts_inference/run_cosmos.sh <DATASET> <TEST_NAME> [COSMOS_MODEL] [N_PROC] [DATA_ROOT] [GPU_IDS] [N_ATTEMPTS] [NUM_FRAMES] [STEPS]
#
# Examples:
#   bash scripts_inference/run_cosmos.sh open_x_embodiment cosmos2.5_2b 2B/post-trained 1 ./data 0 1 49 50
#   bash scripts_inference/run_cosmos.sh open_x_embodiment cosmos2.5_14b 14B/post-trained 1 ./data 0 1 49 50
# ==============================================================================

DATASET=${1:-"debug"}
TEST_NAME=${2:-"cosmos2.5_2b"}
COSMOS_MODEL=${3:-"2B/post-trained"}
N_PROC=${4:-1}
DATA_ROOT=${5:-"./data"}
GPU_IDS=${6:-"0"}
N_ATTEMPTS=${7:-1}
NUM_FRAMES=${8:-49}
STEPS=${9:-50}

PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"${PROJECT_ROOT}/checkpoints"}
THIRD_PARTY_ROOT=${THIRD_PARTY_ROOT:-"${PROJECT_ROOT}/third_party"}
PYTHON_BIN=${PYTHON_BIN:-python}

COSMOS_ROOT=${COSMOS_ROOT:-"${THIRD_PARTY_ROOT}/cosmos-predict2.5"}
COSMOS_CKPT_ROOT=${COSMOS_CKPT_ROOT:-"${CHECKPOINT_ROOT}/Cosmos-Predict2.5"}

COSMOS_VARIANT="${COSMOS_MODEL%%/*}"
COSMOS_CKPT="${COSMOS_CKPT_ROOT}/${COSMOS_VARIANT}"
COSMOS_INFERENCE_TYPE=${COSMOS_INFERENCE_TYPE:-"image2world"}

GT_JSON="${DATA_ROOT}/${DATASET}/summary.json"
PRED_ROOT="${DATA_ROOT}/${DATASET}/generated_data/${TEST_NAME}"

if [ ! -f "${GT_JSON}" ]; then
    echo "Error: summary JSON not found: ${GT_JSON}"
    exit 1
fi

if [ ! -d "${COSMOS_ROOT}" ]; then
    echo "Error: Cosmos-Predict2.5 repo not found: ${COSMOS_ROOT}"
    exit 1
fi

if [ ! -d "${COSMOS_CKPT}" ]; then
    echo "Error: Cosmos checkpoint not found: ${COSMOS_CKPT}"
    exit 1
fi

mkdir -p "${PRED_ROOT}"
mkdir -p logs

IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"

echo "------------------------------------------------"
echo "Starting Cosmos-Predict2.5 local inference"
echo "Dataset      : ${DATASET}"
echo "Test name    : ${TEST_NAME}"
echo "Cosmos model : ${COSMOS_MODEL}"
echo "Task list    : ${GT_JSON}"
echo "Pred root    : ${PRED_ROOT}"
echo "Cosmos root  : ${COSMOS_ROOT}"
echo "Cosmos ckpt  : ${COSMOS_CKPT}"
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
            --model cosmos \
            --gt_root "${GT_JSON}" \
            --pred_root "${PRED_ROOT}" \
            --checkpoint_folder "${COSMOS_CKPT}" \
            --cosmos_root "${COSMOS_ROOT}" \
            --cosmos_model "${COSMOS_MODEL}" \
            --cosmos_inference_type "${COSMOS_INFERENCE_TYPE}" \
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
echo "Cosmos-Predict2.5 inference finished. Output: ${PRED_ROOT}"
echo "------------------------------------------------"

exit "${STATUS}"