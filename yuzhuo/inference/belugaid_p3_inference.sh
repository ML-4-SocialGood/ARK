#!/bin/bash
#SBATCH --job-name=beluga-p3
#SBATCH --time=9999:00:00
#SBATCH --open-mode=append
#SBATCH --output=/data/yil708/ARK/logs/slurm_beluga_p3_%j.out
#SBATCH --error=/data/yil708/ARK/logs/slurm_beluga_p3_%j.err
#SBATCH --gres=gpu:1

set -euo pipefail

cd /data/yil708/ARK
mkdir -p logs

source /data/yil708/miniconda3/etc/profile.d/conda.sh
conda activate ark

export PATH="/data/yil708/software/ollama/bin:$PATH"
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

SPECIES="BelugaID"
PROTOCOL="p3"
MODELS=(
  "qwen3-vl:30b"
)

ANNOTATION_FILES=(
  "annotations/BelugaID/p3/BelugaID_MIA_P3_N4_M2.json"
  # "annotations/BelugaID/p3/BelugaID_MIA_P3_N4_M3.json"
  # "annotations/BelugaID/p3/BelugaID_MIA_P3_N8_M2.json"
  # "annotations/BelugaID/p3/BelugaID_MIA_P3_N8_M3.json"
  # "annotations/BelugaID/p3/BelugaID_MIA_P3_N8_M4.json"
  # "annotations/BelugaID/p3/BelugaID_MIA_P3_N16_M2.json"
  # "annotations/BelugaID/p3/BelugaID_MIA_P3_N16_M3.json"
  # "annotations/BelugaID/p3/BelugaID_MIA_P3_N16_M4.json"
  # "annotations/BelugaID/p3/BelugaID_MIA_P3_N32_M2.json"
  # "annotations/BelugaID/p3/BelugaID_MIA_P3_N32_M3.json"
  # "annotations/BelugaID/p3/BelugaID_MIA_P3_N32_M4.json"
)

PORT_SEED="${SLURM_JOB_ID:-$$}"
export OLLAMA_PORT="${OLLAMA_PORT:-$((20000 + (PORT_SEED % 20000)))}"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:${OLLAMA_PORT}}"
export OLLAMA_MODELS=/data/yil708/software/ollama/models
OLLAMA_URL="http://${OLLAMA_HOST}"

for annotation_file in "${ANNOTATION_FILES[@]}"; do
  if [[ ! -f "${annotation_file}" ]]; then
    echo "Annotation file not found: ${annotation_file}"
    exit 1
  fi
done

echo "Using Ollama host: ${OLLAMA_HOST}"
echo "Species: ${SPECIES}"
echo "Protocol: ${PROTOCOL}"
echo "Annotation files: ${#ANNOTATION_FILES[@]}"

ollama serve > logs/ollama_job_${SLURM_JOB_ID:-local}.log 2>&1 &
OLLAMA_PID=$!
trap 'kill ${OLLAMA_PID} 2>/dev/null || true' EXIT

echo "Waiting for Ollama to start..."
for i in {1..60}; do
    if curl -s "${OLLAMA_URL}" > /dev/null; then echo "Ollama started on ${OLLAMA_HOST}!"; break; fi
    sleep 2
done

echo "Available models:"
ollama list

for model in "${MODELS[@]}"; do
  echo "========================================"
  echo "Starting model: ${model}"
  echo "========================================"

  for annotation_file in "${ANNOTATION_FILES[@]}"; do
    echo "Running ${SPECIES} / ${PROTOCOL} / ${annotation_file} / ${model}"

    python scripts_evaluate/run_inference.py \
      --species "${SPECIES}" \
      --protocol "${PROTOCOL}" \
      --annotation_file "${annotation_file}" \
      --model "${model}" \
      --host "${OLLAMA_URL}" \
      --resume
  done
done

# sbatch belugaid_p3_inference.sh
# OLLAMA_PORT=23456 sbatch belugaid_p3_inference.sh
