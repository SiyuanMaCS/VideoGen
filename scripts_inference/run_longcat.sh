#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Local Inference: LongCat-Video (base i2v, 480p)
# Usage:
#   bash scripts_inference/run_longcat.sh <DATASET> <TEST_NAME> [N_PROC] [DATA_ROOT] [GPU_IDS] [N_ATTEMPTS]
# Example:
#   bash scripts_inference/run_longcat.sh atomic_test longcat 1 ./data 0 1
#
# Prereqs on the GPU box:
#   third_party/LongCat-Video     (git clone + pip install -r requirements + matching flash-attn wheel)
#   checkpoints/LongCat-Video     (tokenizer / text_encoder(UMT5) / vae / scheduler / dit)
#
# NOTE: base i2v is 50 steps × 13.6B ≈ ~10 min/video on one H100. Each rank runs an
# independent single-GPU process (world_size=1) with its own MASTER_PORT.
# ==============================================================================

DATASET=${1:-"debug"}
TEST_NAME=${2:-"longcat"}
N_PROC=${3:-1}
DATA_ROOT=${4:-"./data"}
GPU_IDS=${5:-"0"}
N_ATTEMPTS=${6:-1}

PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"${PROJECT_ROOT}/checkpoints"}
THIRD_PARTY_ROOT=${THIRD_PARTY_ROOT:-"${PROJECT_ROOT}/third_party"}
PYTHON_BIN=${PYTHON_BIN:-python}
LONGCAT_ROOT=${LONGCAT_ROOT:-"${THIRD_PARTY_ROOT}/LongCat-Video"}
LONGCAT_CKPT=${LONGCAT_CKPT:-"${CHECKPOINT_ROOT}/LongCat-Video"}

GT_JSON="${DATA_ROOT}/${DATASET}/summary.json"
PRED_ROOT="${DATA_ROOT}/${DATASET}/generated_data/${TEST_NAME}"

if [ ! -f "${GT_JSON}" ]; then echo "Error: summary not found: ${GT_JSON}"; exit 1; fi
if [ ! -d "${LONGCAT_ROOT}" ]; then echo "Error: LongCat-Video repo not found: ${LONGCAT_ROOT}"; exit 1; fi
if [ ! -d "${LONGCAT_CKPT}" ]; then echo "Error: LongCat checkpoint not found: ${LONGCAT_CKPT}"; exit 1; fi

mkdir -p "${PRED_ROOT}" logs
IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"

echo "------------------------------------------------"
echo "LongCat-Video local inference (base i2v, 480p)"
echo "Dataset / Test : ${DATASET} / ${TEST_NAME}"
echo "LongCat repo   : ${LONGCAT_ROOT}"
echo "LongCat ckpt   : ${LONGCAT_CKPT}"
echo "Pred root      : ${PRED_ROOT}"
echo "------------------------------------------------"

PIDS=()
for ((RANK=0; RANK<N_PROC; RANK++)); do
    GPU=${GPU_ARRAY[$((RANK % ${#GPU_ARRAY[@]}))]}
    (
        export CUDA_VISIBLE_DEVICES="${GPU}"
        export PYTHONPATH="${LONGCAT_ROOT}:${PYTHONPATH:-}"
        export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
        export MASTER_ADDR=localhost
        export MASTER_PORT=$((29500 + RANK))   # unique per shard to avoid collisions
        "${PYTHON_BIN}" run_local.py \
            --model longcat \
            --gt_root "${GT_JSON}" \
            --pred_root "${PRED_ROOT}" \
            --checkpoint_folder "${LONGCAT_CKPT}" \
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
echo "LongCat-Video inference finished. Output: ${PRED_ROOT}"
exit "${STATUS}"
