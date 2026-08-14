#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Batch: Cosmos3-Nano across all datasets × prompt variants (prefix + rewrite)
# Resume-safe: existing mp4s (including placeholders) are skipped.
#
# Usage:
#   PYTHON_BIN=/mnt/bn/embodied-lf/yyq/cosmos-framework/.venv/bin/python \
#   bash scripts_inference/run_cosmos3_all.sh
# ==============================================================================

export PYTHON_BIN=${PYTHON_BIN:-"/mnt/bn/embodied-lf/yyq/cosmos-framework/.venv/bin/python"}
export HF_HOME=${HF_HOME:-"/mnt/bn/embodied-lf/masiyuan/.cache/huggingface"}

CHECKPOINT="Cosmos3-Nano"
N_PROC=8
DATA_ROOT="./data"
GPU_IDS="0,1,2,3,4,5,6,7"
N_ATTEMPTS=1
KEEP_PROB=0.8

DATASETS=(
    agibot_world
    dreamdojo_hv
    droid
    egodex
    egodex_human
    egoscaler_human
    epickitchens_human
    gr1_inlab
    libero
    open_x_embodiment
    robotwin
)

VARIANTS=(
    "cosmos3_prefix:prompt_prefix"
    "cosmos3_rewrite:prompt_rewrite"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for variant_pair in "${VARIANTS[@]}"; do
    TEST_NAME="${variant_pair%%:*}"
    PROMPT_KEY="${variant_pair##*:}"

    for DATASET in "${DATASETS[@]}"; do
        echo ""
        echo "========================================================"
        echo " ${DATASET} / ${TEST_NAME} (prompt_key=${PROMPT_KEY})"
        echo "========================================================"

        bash "${SCRIPT_DIR}/run_cosmos3.sh" \
            "${DATASET}" \
            "${TEST_NAME}" \
            "${CHECKPOINT}" \
            "${N_PROC}" \
            "${DATA_ROOT}" \
            "${GPU_IDS}" \
            "${N_ATTEMPTS}" \
            "${KEEP_PROB}" \
            "${PROMPT_KEY}"

        echo "[DONE] ${DATASET} / ${TEST_NAME}"
    done
done

echo ""
echo "All datasets completed."
