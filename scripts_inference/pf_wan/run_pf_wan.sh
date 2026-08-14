#!/usr/bin/env bash
set -euo pipefail

NAME="${1:?Usage: run_pf_wan.sh NAME MANIFEST [GPU_COUNT]}"
MANIFEST="${2:?Usage: run_pf_wan.sh NAME MANIFEST [GPU_COUNT]}"
GPU_COUNT="${3:-4}"

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
THIRD_PARTY_ROOT="${THIRD_PARTY_ROOT:-${PROJECT_ROOT}/third_party}"
PF_WAN_ROOT="${PF_WAN_ROOT:-${THIRD_PARTY_ROOT}/PhysisForcing/pf_wan}"
PF_WAN_CKPT="${PF_WAN_CKPT:-${PROJECT_ROOT}/checkpoints/PhysisForcing/PF_Wan}"
PF_WAN_RAW_ROOT="${PF_WAN_RAW_ROOT:-${PROJECT_ROOT}/outputs/pf_wan_raw}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
MASTER_PORT="${MASTER_PORT:-29500}"
NUM_FRAMES="${NUM_FRAMES:-81}"
FPS="${FPS:-16}"
SEED="${SEED:-42}"

test -d "${PF_WAN_ROOT}" || { echo "Missing PF_Wan repo: ${PF_WAN_ROOT}"; exit 1; }
test -d "${PF_WAN_CKPT}" || { echo "Missing PF_Wan checkpoint: ${PF_WAN_CKPT}"; exit 1; }
test -f "${MANIFEST}" || { echo "Missing manifest: ${MANIFEST}"; exit 1; }
mkdir -p "${PF_WAN_RAW_ROOT}" "${PROJECT_ROOT}/logs"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT - 1)))
  export CUDA_VISIBLE_DEVICES
fi

export PYTHONPATH="${PF_WAN_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

cd "${PF_WAN_ROOT}"
"${TORCHRUN_BIN}" \
  --nproc-per-node="${GPU_COUNT}" \
  --nnodes=1 \
  --node-rank=0 \
  --rdzv-endpoint="127.0.0.1:${MASTER_PORT}" \
  tools/main.py \
  --config-file configs/generate/pf_wan_i2v.jsonc \
  output_dir "${PF_WAN_RAW_ROOT}" \
  proj_name "${NAME}" \
  inference.positive_prompt "${MANIFEST}" \
  inference.num_frames "${NUM_FRAMES}" \
  inference.fps "${FPS}" \
  inference.seed "${SEED}" \
  meta_model.backbone.weight "${PF_WAN_CKPT}/backbone.pth" \
  meta_model.text_encoder.weight "${PF_WAN_CKPT}/models_t5_umt5-xxl-enc-bf16.pth" \
  meta_model.vae.weight "${PF_WAN_CKPT}/Wan2.1_VAE.pth" \
  meta_model.tokenizer.override_list "['config.name','${PF_WAN_CKPT}/google/umt5-xxl']" \
  meta_model.ulysses_size "${GPU_COUNT}" \
  meta_model.hybrid_gpu_num "${GPU_COUNT}" \
  2>&1 | tee "${PROJECT_ROOT}/logs/pf_wan_${NAME}.log"
