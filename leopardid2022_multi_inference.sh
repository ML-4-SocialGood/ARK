#!/bin/bash
#SBATCH --job-name=leopard-multi
#SBATCH --time=9999:00:00
#SBATCH --open-mode=append
#SBATCH --output=/data/yil708/ARK/logs/slurm_leopard_multi_%j.out
#SBATCH --error=/data/yil708/ARK/logs/slurm_leopard_multi_%j.err
#SBATCH --gres=gpu:1

set -euo pipefail

cd /data/yil708/ARK
mkdir -p logs

source /data/yil708/miniconda3/etc/profile.d/conda.sh
conda activate ark

export PATH="/data/yil708/software/ollama/bin:$PATH"
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

SPECIES="LeopardID2022"
PROTOCOL_FILTER="${PROTOCOL_FILTER:-}"
MODELS=(
  "qwen3.5:2b"
  "qwen3-vl:2b"
  "qwen3-vl:4b"
)

PORT_SEED="${SLURM_JOB_ID:-$$}"
export OLLAMA_PORT="${OLLAMA_PORT:-$((20000 + (PORT_SEED % 20000)))}"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:${OLLAMA_PORT}}"
export OLLAMA_MODELS=/data/yil708/software/ollama/models
OLLAMA_URL="http://${OLLAMA_HOST}"

if [[ -n "${PROTOCOL_FILTER}" ]]; then
  ANNOTATION_ROOT="annotations/${SPECIES}/${PROTOCOL_FILTER}"
else
  ANNOTATION_ROOT="annotations/${SPECIES}"
fi

if [[ ! -d "${ANNOTATION_ROOT}" ]]; then
  echo "Annotation directory not found: ${ANNOTATION_ROOT}"
  exit 1
fi

mapfile -t ALL_ANNOTATION_FILES < <(find "${ANNOTATION_ROOT}" -type f -name "*.json" | sort)

ANNOTATION_FILES=()
for annotation_file in "${ALL_ANNOTATION_FILES[@]}"; do
  protocol="$(basename "$(dirname "${annotation_file}")")"

  # P1 only needs the N4 setting.
  if [[ "${protocol}" == "p1" && "$(basename "${annotation_file}")" != *_N4.json ]]; then
    continue
  fi

  ANNOTATION_FILES+=("${annotation_file}")
done

if [[ ${#ANNOTATION_FILES[@]} -eq 0 ]]; then
  echo "No annotation files found under ${ANNOTATION_ROOT}"
  exit 1
fi

echo "Using Ollama host: ${OLLAMA_HOST}"
echo "Species: ${SPECIES}"
echo "Protocol filter: ${PROTOCOL_FILTER:-<all>}"
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
    protocol="$(basename "$(dirname "${annotation_file}")")"
    echo "Running ${SPECIES} / ${protocol} / ${annotation_file} / ${model}"

    python scripts_evaluate/run_inference.py \
      --species "${SPECIES}" \
      --protocol "${protocol}" \
      --annotation_file "${annotation_file}" \
      --model "${model}" \
      --host "${OLLAMA_URL}" \
      --resume
  done
done

# PROTOCOL_FILTER=p1 sbatch leopardid2022_multi_inference.sh
# OLLAMA_PORT=23456 sbatch leopardid2022_multi_inference.sh
