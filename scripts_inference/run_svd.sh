#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Local Inference: Stable Video Diffusion (SVD-xt)  [image-only i2v]
# Usage:
#   bash scripts_inference/run_svd.sh <DATASET> <TEST_NAME> [N_PROC] [DATA_ROOT] [GPU_IDS] [N_ATTEMPTS]
# Example:
#   bash scripts_inference/run_svd.sh atomic_test svd_xt 1 ./data 0 1
#
# Checkpoint (diffusers SVD dir: unet/vae/image_encoder/scheduler + model_index.json):
#   checkpoints/SVD/stable-video-diffusion-img2vid-xt
#   (non-gated ModelScope mirror: AI-ModelScope/stable-video-diffusion-img2vid-xt)
# ==============================================================================

DATASET=${1:-"debug"}
TEST_NAME=${2:-"svd_xt"}
N_PROC=${3:-1}
DATA_ROOT=${4:-"./data"}
GPU_IDS=${5:-"0"}
N_ATTEMPTS=${6:-1}

PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"${PROJECT_ROOT}/checkpoints"}
PYTHON_BIN=${PYTHON_BIN:-python}
SVD_CKPT=${SVD_CKPT:-"${CHECKPOINT_ROOT}/SVD/stable-video-diffusion-img2vid-xt"}

GT_JSON="${DATA_ROOT}/${DATASET}/summary.json"
PRED_ROOT="${DATA_ROOT}/${DATASET}/generated_data/${TEST_NAME}"

if [ ! -f "${GT_JSON}" ]; then echo "Error: summary not found: ${GT_JSON}"; exit 1; fi
if [ ! -d "${SVD_CKPT}" ]; then echo "Error: SVD checkpoint not found: ${SVD_CKPT}"; exit 1; fi

mkdir -p "${PRED_ROOT}" logs
IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"

echo "------------------------------------------------"
echo "SVD-xt local inference (image-only; prompt ignored)"
echo "Dataset / Test : ${DATASET} / ${TEST_NAME}"
echo "SVD ckpt       : ${SVD_CKPT}"
echo "Pred root      : ${PRED_ROOT}"
echo "Processes/GPUs : ${N_PROC} / ${GPU_IDS}"
echo "------------------------------------------------"

PIDS=()
for ((RANK=0; RANK<N_PROC; RANK++)); do
    GPU=${GPU_ARRAY[$((RANK % ${#GPU_ARRAY[@]}))]}
    (
        export CUDA_VISIBLE_DEVICES="${GPU}"
        "${PYTHON_BIN}" run_local.py \
            --model svd \
            --gt_root "${GT_JSON}" \
            --pred_root "${PRED_ROOT}" \
            --checkpoint_folder "${SVD_CKPT}" \
            --n_attempts "${N_ATTEMPTS}" \
            --rank "${RANK}" \
            --world_size "${N_PROC}" \
            --gpu 0
    ) 2>&1 | tee "logs/${TEST_NAME}_rank${RANK}.log" &
    PIDS+=($!)
done

STATUS=0
for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then STATUS=1; fi
done
echo "SVD-xt inference finished. Output: ${PRED_ROOT}"
exit "${STATUS}"
