#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/vepfs/fs_projects/linhaitao/ExplicitShortCut"
CKPT_DIR="${ROOT_DIR}/exp/imm-b2-cnd/checkpoints"

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "ERROR: 未找到目录: ${CKPT_DIR}"
  exit 1
fi

cd "${CKPT_DIR}"

if compgen -G "*.pt" > /dev/null; then
  RESUME_NUM=$(ls -1 *.pt 2>/dev/null \
    | sed -n 's/^\([0-9]\+\)\.pt$/\1/p' \
    | sort -n \
    | tail -1)
else
  echo "ERROR: ${CKPT_DIR} 下未找到 *.pt 文件"
  exit 1
fi

if [[ -z "${RESUME_NUM:-}" ]]; then
  echo "ERROR: 未能解析到有效的检查点编号 (*.pt)"
  exit 1
fi

echo ">>> 选择检查点: ${RESUME_NUM}.pt"

cd "${ROOT_DIR}"

torchrun --nproc_per_node=8 train_imm.py \
  --exp-name "imm-b2-cnd" \
  --output-dir "exp" \
  --data-dir "/internfs/linhaitao/esc/imagenet_vq/train/" \
  --model "SiT-B/2" \
  --resolution 256 \
  --batch-size 2048 \
  --allow-tf32 \
  --resume-step "${RESUME_NUM}" \
  --optimizer "adamw" \
  --adam-beta1 0.9 \
  --adam-beta2 0.999 \
  --adam-epsilon 1e-8 \
  --adam-weight-decay 0.0 \
  --time-sampler "uniform" \
  --mixed-precision "bf16" \
  --epochs 960 \
  --path-type "linear" \
  --no-debug
