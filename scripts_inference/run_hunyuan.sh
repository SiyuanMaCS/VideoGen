#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Local Inference: HunyuanVideo-1.5 (480p i2v, step-distilled)
# Usage:
#   bash scripts_inference/run_hunyuan.sh <DATASET> <TEST_NAME> [N_PROC] [DATA_ROOT] [GPU_IDS] [N_ATTEMPTS]
# Example:
#   bash scripts_inference/run_hunyuan.sh atomic_test hunyuan15 1 ./data 0 1
#
# Prereqs on the GPU box:
#   third_party/HunyuanVideo-1.5           (git clone + pip install -r requirements + matching flash-attn wheel)
#   checkpoints/HunyuanVideo-1.5           (transformer/480p_i2v_step_distilled + vae + scheduler +
#                                           text_encoder/{llm,byt5-small,Glyph-SDXL-v2} + vision_encoder/siglip)
# ==============================================================================

DATASET=${1:-"debug"}
TEST_NAME=${2:-"hunyuan15"}
N_PROC=${3:-1}
DATA_ROOT=${4:-"./data"}
GPU_IDS=${5:-"0"}
N_ATTEMPTS=${6:-1}

PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"${PROJECT_ROOT}/checkpoints"}
THIRD_PARTY_ROOT=${THIRD_PARTY_ROOT:-"${PROJECT_ROOT}/third_party"}
PYTHON_BIN=${PYTHON_BIN:-python}
HUNYUAN_ROOT=${HUNYUAN_ROOT:-"${THIRD_PARTY_ROOT}/HunyuanVideo-1.5"}
HUNYUAN_CKPT=${HUNYUAN_CKPT:-"${CHECKPOINT_ROOT}/HunyuanVideo-1.5"}

GT_JSON="${DATA_ROOT}/${DATASET}/summary.json"
PRED_ROOT="${DATA_ROOT}/${DATASET}/generated_data/${TEST_NAME}"

if [ ! -f "${GT_JSON}" ]; then echo "Error: summary not found: ${GT_JSON}"; exit 1; fi
if [ ! -d "${HUNYUAN_ROOT}" ]; then echo "Error: HunyuanVideo-1.5 repo not found: ${HUNYUAN_ROOT}"; exit 1; fi
if [ ! -d "${HUNYUAN_CKPT}" ]; then echo "Error: Hunyuan checkpoint not found: ${HUNYUAN_CKPT}"; exit 1; fi

mkdir -p "${PRED_ROOT}" logs
IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"

echo "------------------------------------------------"
echo "HunyuanVideo-1.5 local inference (480p i2v step-distilled)"
echo "Dataset / Test : ${DATASET} / ${TEST_NAME}"
echo "Hunyuan repo   : ${HUNYUAN_ROOT}"
echo "Hunyuan ckpt   : ${HUNYUAN_CKPT}"
echo "Pred root      : ${PRED_ROOT}"
echo "------------------------------------------------"

PIDS=()
for ((RANK=0; RANK<N_PROC; RANK++)); do
    GPU=${GPU_ARRAY[$((RANK % ${#GPU_ARRAY[@]}))]}
    (
        export CUDA_VISIBLE_DEVICES="${GPU}"
        export PYTHONPATH="${HUNYUAN_ROOT}:${PYTHONPATH:-}"
        export MASTER_PORT=$((29500 + RANK))
        "${PYTHON_BIN}" run_local.py \
            --model hunyuan \
            --gt_root "${GT_JSON}" \
            --pred_root "${PRED_ROOT}" \
            --checkpoint_folder "${HUNYUAN_CKPT}" \
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
echo "HunyuanVideo-1.5 inference finished. Output: ${PRED_ROOT}"
exit "${STATUS}"
