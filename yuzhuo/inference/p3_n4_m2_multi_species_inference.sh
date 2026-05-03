#!/bin/bash
#SBATCH --job-name=p3-n4-m2
#SBATCH --time=9999:00:00
#SBATCH --open-mode=append
#SBATCH --output=/data/yil708/ARK/logs/slurm_p3_n4_m2_%j.out
#SBATCH --error=/data/yil708/ARK/logs/slurm_p3_n4_m2_%j.err
#SBATCH --gres=gpu:1

set -euo pipefail

cd /data/yil708/ARK
mkdir -p logs

source /data/yil708/miniconda3/etc/profile.d/conda.sh
conda activate ark

export PATH="/data/yil708/software/ollama/bin:$PATH"
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

PROTOCOL="p3"
MODELS=(
  # "qwen3-vl:30b"
  # "gemma3:4b"
  "qwen3.5:0.8b"
)

ANNOTATION_FILES=(
  # "annotations/BelugaID/p3/BelugaID_MIA_P3_N4_M2.json"
  # "annotations/BirdIndividualID/p3/BirdIndividualID_MIA_P3_N4_M2.json"
  # "annotations/CTai/p3/CTai_MIA_P3_N4_M2.json"
  # "annotations/Giraffes/p3/Giraffes_MIA_P3_N4_M2.json"
  # "annotations/HumpbackWhaleID/p3/HumpbackWhaleID_MIA_P3_N4_M2.json"
  # "annotations/IPanda50/p3/IPanda50_MIA_P3_N4_M2.json"
  # "annotations/LeopardID2022/p3/LeopardID2022_MIA_P3_N4_M2.json"
  # "annotations/Lion/p3/Lion_MIA_P3_N4_M2.json"
  # "annotations/NDD20/p3/NDD20_MIA_P3_N4_M2.json"
  # "annotations/NyalaData/p3/NyalaData_MIA_P3_N4_M2.json"
  # "annotations/SealID/p3/SealID_MIA_P3_N4_M2.json"
  "annotations/WhaleSharkID/p3/WhaleSharkID_MIA_P3_N4_M2.json"
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
    species="$(basename "$(dirname "$(dirname "${annotation_file}")")")"
    echo "Running ${species} / ${PROTOCOL} / ${annotation_file} / ${model}"

    python scripts_evaluate/run_inference.py \
      --species "${species}" \
      --protocol "${PROTOCOL}" \
      --annotation_file "${annotation_file}" \
      --model "${model}" \
      --host "${OLLAMA_URL}" \
      --resume
  done
done

# sbatch p3_n4_m2_multi_species_inference.sh
# OLLAMA_PORT=23456 sbatch p3_n4_m2_multi_species_inference.sh
