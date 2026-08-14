#!/usr/bin/env bash
set -euo pipefail

export MASTER_PORT=$(shuf -i 20000-65535 -n 1)

# ==============================================================================
# Local Inference Script: Wan2.2 TI2V-5B
# Usage:
#   bash scripts_inference/run_wan22.sh <DATASET> <TEST_NAME> [WAN_TASK] [N_PROC] [DATA_ROOT] [GPU_IDS] [N_ATTEMPTS]
#
# Example:
#   bash scripts_inference/run_wan22.sh open_x_embodiment wan22_ti2v_5b ti2v-5B 1 ./data 0 1
# ==============================================================================

DATASET=${1:-"debug"}
TEST_NAME=${2:-"wan22_ti2v_5b"}
WAN_TASK=${3:-"ti2v-5B"}
N_PROC=${4:-1}
DATA_ROOT=${5:-"./data"}
GPU_IDS=${6:-"0"}
N_ATTEMPTS=${7:-1}

PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"${PROJECT_ROOT}/checkpoints"}
THIRD_PARTY_ROOT=${THIRD_PARTY_ROOT:-"${PROJECT_ROOT}/third_party"}
PYTHON_BIN=${PYTHON_BIN:-python}

WAN_ROOT=${WAN_ROOT:-"${THIRD_PARTY_ROOT}/Wan2.2"}
WAN_CKPT=${WAN_CKPT:-"${CHECKPOINT_ROOT}/Wan2.2/Wan2.2-TI2V-5B"}

WAN_SIZE=${WAN_SIZE:-"1280*704"}
WAN_FRAME_NUM=${WAN_FRAME_NUM:--1}
WAN_SAMPLE_STEPS=${WAN_SAMPLE_STEPS:--1}
WAN_SAMPLE_SHIFT=${WAN_SAMPLE_SHIFT:--1.0}
WAN_SAMPLE_GUIDE_SCALE=${WAN_SAMPLE_GUIDE_SCALE:--1.0}
WAN_SAMPLE_SOLVER=${WAN_SAMPLE_SOLVER:-""}
WAN_EXTRA_ARGS=${WAN_EXTRA_ARGS:-""}

GT_JSON="${DATA_ROOT}/${DATASET}/summary.json"
PRED_ROOT="${DATA_ROOT}/${DATASET}/generated_data/${TEST_NAME}"

if [ ! -f "${GT_JSON}" ]; then
    echo "Error: summary JSON not found: ${GT_JSON}"
    exit 1
fi

if [ ! -d "${WAN_ROOT}" ]; then
    echo "Error: Wan2.2 repo not found: ${WAN_ROOT}"
    exit 1
fi

if [ ! -d "${WAN_CKPT}" ]; then
    echo "Error: Wan2.2 checkpoint not found: ${WAN_CKPT}"
    exit 1
fi

mkdir -p "${PRED_ROOT}"
mkdir -p logs

IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"

echo "------------------------------------------------"
echo "Starting Wan2.2 TI2V-5B local inference"
echo "Dataset      : ${DATASET}"
echo "Test name    : ${TEST_NAME}"
echo "Wan task     : ${WAN_TASK}"
echo "Task list    : ${GT_JSON}"
echo "Pred root    : ${PRED_ROOT}"
echo "Wan root     : ${WAN_ROOT}"
echo "Wan ckpt     : ${WAN_CKPT}"
echo "Wan size     : ${WAN_SIZE}"
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
            --model wan22 \
            --gt_root "${GT_JSON}" \
            --pred_root "${PRED_ROOT}" \
            --checkpoint_folder "${WAN_CKPT}" \
            --wan_root "${WAN_ROOT}" \
            --wan_task "${WAN_TASK}" \
            --wan_size "${WAN_SIZE}" \
            --wan_frame_num "${WAN_FRAME_NUM}" \
            --wan_sample_steps "${WAN_SAMPLE_STEPS}" \
            --wan_sample_shift "${WAN_SAMPLE_SHIFT}" \
            --wan_sample_guide_scale "${WAN_SAMPLE_GUIDE_SCALE}" \
            --wan_sample_solver "${WAN_SAMPLE_SOLVER}" \
            --wan_extra_args "${WAN_EXTRA_ARGS}" \
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
echo "Wan2.2 TI2V-5B inference finished. Output: ${PRED_ROOT}"
echo "------------------------------------------------"

exit "${STATUS}"