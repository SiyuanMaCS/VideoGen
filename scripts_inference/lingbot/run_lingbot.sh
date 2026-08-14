#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?Usage: run_lingbot.sh DATASET RUN_NAME [PROMPT_KEY]}"
RUN_NAME="${2:?Usage: run_lingbot.sh DATASET RUN_NAME [PROMPT_KEY]}"
PROMPT_KEY="${3:-prompt}"

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
THIRD_PARTY_ROOT="${THIRD_PARTY_ROOT:-${PROJECT_ROOT}/third_party}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}"
LINGBOT_ROOT="${LINGBOT_ROOT:-${THIRD_PARTY_ROOT}/lingbot-video}"
LINGBOT_CKPT="${LINGBOT_CKPT:-${CHECKPOINT_ROOT}/LingBot-Video/lingbot-video-moe-30b-a3b-base}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_COUNT="${GPU_COUNT:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"

test -d "${LINGBOT_ROOT}" || { echo "Missing LingBot repo: ${LINGBOT_ROOT}"; exit 1; }
test -d "${LINGBOT_CKPT}" || { echo "Missing LingBot checkpoint: ${LINGBOT_CKPT}"; exit 1; }
test -f "${DATA_ROOT}/${DATASET}/summary.json" || { echo "Missing summary: ${DATA_ROOT}/${DATASET}/summary.json"; exit 1; }

export PYTHONPATH="${LINGBOT_ROOT}:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc-per-node="${GPU_COUNT}" \
  --master-port="${MASTER_PORT}" \
  "${PROJECT_ROOT}/scripts_inference/lingbot/run_lingbot_batch.py" \
  --project-root "${PROJECT_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --model-dir "${LINGBOT_CKPT}" \
  --datasets "${DATASET}" \
  --run-name "${RUN_NAME}" \
  --prompt-key "${PROMPT_KEY}"
