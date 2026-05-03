#!/bin/bash
#SBATCH --job-name=p5-resolution
#SBATCH --time=9999:00:00
#SBATCH --open-mode=append
#SBATCH --output=/data/yil708/ARK/logs/slurm_p5_resolution_%j.out
#SBATCH --error=/data/yil708/ARK/logs/slurm_p5_resolution_%j.err
#SBATCH --gres=gpu:1

set -euo pipefail

cd /data/yil708/ARK
mkdir -p logs

source /data/yil708/miniconda3/etc/profile.d/conda.sh
conda activate ark

export PATH="/data/yil708/software/ollama/bin:$PATH"
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

PROTOCOL="p5"
SPECIES_FILTER="${SPECIES_FILTER:-}"
MODELS=(
  "qwen3-vl:30b"
)

if [[ -n "${SPECIES_FILTER}" ]]; then
  mapfile -t ANNOTATION_FILES < <(
    find "annotations/${SPECIES_FILTER}" -type f -path "*/p5/*_I2I_P5_resolution_*.json" | sort
  )
else
  mapfile -t ANNOTATION_FILES < <(
    find annotations -type f -path '*/p5/*_I2I_P5_resolution_*.json' | sort
  )
fi

if [[ ${#ANNOTATION_FILES[@]} -eq 0 ]]; then
  if [[ -n "${SPECIES_FILTER}" ]]; then
    echo "No P5 resolution annotation files found for species ${SPECIES_FILTER}."
  else
    echo "No P5 resolution annotation files found."
  fi
  exit 1
fi

PORT_SEED="${SLURM_JOB_ID:-$$}"
export OLLAMA_PORT="${OLLAMA_PORT:-$((20000 + (PORT_SEED % 20000)))}"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:${OLLAMA_PORT}}"
export OLLAMA_MODELS=/data/yil708/software/ollama/models
OLLAMA_URL="http://${OLLAMA_HOST}"

echo "Using Ollama host: ${OLLAMA_HOST}"
echo "Protocol: ${PROTOCOL}"
echo "Species filter: ${SPECIES_FILTER:-<all>}"
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

# sbatch p5_resolution_multi_species_inference.sh
# SPECIES_FILTER=Lion sbatch p5_resolution_multi_species_inference.sh
# OLLAMA_PORT=23456 sbatch p5_resolution_multi_species_inference.sh
