#!/usr/bin/env bash
set -euo pipefail

# Quick smoke test: run Cosmos3-Nano on 1 sample to verify the pipeline works.
# Usage:
#   bash scripts_inference/test_cosmos3.sh

export PYTHON_BIN=${PYTHON_BIN:-"/mnt/bn/embodied-lf/yyq/cosmos-framework/.venv/bin/python"}
export HF_HOME=${HF_HOME:-"/mnt/bn/embodied-lf/masiyuan/.cache/huggingface"}
export HF_TOKEN=${HF_TOKEN:-"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"}
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118

COSMOS3_ROOT="/mnt/bn/embodied-lf/yyq/cosmos-framework"
DATA_ROOT="./data"
DATASET="agibot_world"
TEST_NAME="cosmos3_test_smoke"
PRED_ROOT="${DATA_ROOT}/${DATASET}/generated_data/${TEST_NAME}"

mkdir -p "${PRED_ROOT}" logs

echo "=== Cosmos3-Nano smoke test ==="
echo "Python: ${PYTHON_BIN}"
echo "Cosmos3 root: ${COSMOS3_ROOT}"
echo "Dataset: ${DATASET}"
echo "Output: ${PRED_ROOT}"
echo ""

# Run on rank 0 only, world_size=1, keep_prob=1.0 (no sampling), first 1 sample
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" run_local.py \
    --model cosmos3 \
    --gt_root "${DATA_ROOT}/${DATASET}/summary.json" \
    --pred_root "${PRED_ROOT}" \
    --checkpoint_folder "unused" \
    --cosmos3_root "${COSMOS3_ROOT}" \
    --cosmos3_checkpoint "Cosmos3-Nano" \
    --cosmos3_num_gpus 1 \
    --cosmos3_parallelism "throughput" \
    --prompt_key "prompt_prefix" \
    --n_attempts 1 \
    --sample_keep_prob 1.0 \
    --rank 0 \
    --world_size 9999 \
    --gpu 0 \
    2>&1 | tee logs/cosmos3_smoke_test.log

echo ""
echo "=== Check output ==="
find "${PRED_ROOT}" -name "*.mp4" -ls
