#!/usr/bin/env bash
set -euo pipefail

export MASTER_PORT=$(shuf -i 20000-65535 -n 1)

# ==============================================================================
# Local Inference Script: CogVideo
# Usage:
#   bash scripts_inference/run_cogvideo.sh <DATASET> <TEST_NAME> [MODEL_NAME] [N_PROC] [DATA_ROOT] [GPU_IDS] [N_ATTEMPTS]
#
# Example:
#   bash scripts_inference/run_cogvideo.sh open_x_embodiment cogvideo CogVideoX-5b 1 ./data 0 1
# ==============================================================================

DATASET=${1:-"debug"}
TEST_NAME=${2:-"cogvideo"}
MODEL_NAME=${3:-"CogVideoX1.5-5B-I2V"}
N_PROC=${4:-1}
DATA_ROOT=${5:-"./data"}
GPU_IDS=${6:-"0"}
N_ATTEMPTS=${7:-1}

PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"${PROJECT_ROOT}/checkpoints"}
PYTHON_BIN=${PYTHON_BIN:-python}

COGVIDEO_CKPT=${COGVIDEO_CKPT:-"${CHECKPOINT_ROOT}/CogVideo/${MODEL_NAME}"}

COGVIDEO_STEPS=${COGVIDEO_STEPS:-50}
COGVIDEO_NUM_FRAMES=${COGVIDEO_NUM_FRAMES:-81}
COGVIDEO_GUIDANCE_SCALE=${COGVIDEO_GUIDANCE_SCALE:-6.0}
COGVIDEO_FPS=${COGVIDEO_FPS:-16}
COGVIDEO_WIDTH=${COGVIDEO_WIDTH:--1}
COGVIDEO_HEIGHT=${COGVIDEO_HEIGHT:--1}

GT_JSON="${DATA_ROOT}/${DATASET}/summary.json"
PRED_ROOT="${DATA_ROOT}/${DATASET}/generated_data/${TEST_NAME}"

if [ ! -f "${GT_JSON}" ]; then
    echo "Error: summary JSON not found: ${GT_JSON}"
    exit 1
fi

if [ ! -d "${COGVIDEO_CKPT}" ]; then
    echo "Error: CogVideo checkpoint not found: ${COGVIDEO_CKPT}"
    exit 1
fi

mkdir -p "${PRED_ROOT}"
mkdir -p logs

IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"

echo "------------------------------------------------"
echo "Starting CogVideo local inference"
echo "Dataset      : ${DATASET}"
echo "Test name    : ${TEST_NAME}"
echo "Model name   : ${MODEL_NAME}"
echo "Task list    : ${GT_JSON}"
echo "Pred root    : ${PRED_ROOT}"
echo "Checkpoint   : ${COGVIDEO_CKPT}"
echo "Processes    : ${N_PROC}"
echo "GPU ids      : ${GPU_IDS}"
echo "Attempts     : ${N_ATTEMPTS}"
echo "Frames       : ${COGVIDEO_NUM_FRAMES}"
echo "Steps        : ${COGVIDEO_STEPS}"
echo "------------------------------------------------"

PIDS=()

for ((RANK=0; RANK<N_PROC; RANK++)); do
    GPU=${GPU_ARRAY[$((RANK % ${#GPU_ARRAY[@]}))]}
    LOG_FILE="logs/${TEST_NAME}_rank${RANK}.log"

    (
        export CUDA_VISIBLE_DEVICES="${GPU}"

        "${PYTHON_BIN}" run_local.py \
            --model cogvideo \
            --gt_root "${GT_JSON}" \
            --pred_root "${PRED_ROOT}" \
            --checkpoint_folder "${COGVIDEO_CKPT}" \
            --cogvideo_steps "${COGVIDEO_STEPS}" \
            --cogvideo_num_frames "${COGVIDEO_NUM_FRAMES}" \
            --cogvideo_guidance_scale "${COGVIDEO_GUIDANCE_SCALE}" \
            --cogvideo_fps "${COGVIDEO_FPS}" \
            --cogvideo_width "${COGVIDEO_WIDTH}" \
            --cogvideo_height "${COGVIDEO_HEIGHT}" \
            --cogvideo_model_cpu_offload \
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
echo "CogVideo inference finished. Output: ${PRED_ROOT}"
echo "------------------------------------------------"

exit "${STATUS}"