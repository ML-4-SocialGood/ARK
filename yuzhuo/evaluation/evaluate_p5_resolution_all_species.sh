#!/bin/bash

set -euo pipefail

cd /data/yil708/ARK

source /data/yil708/miniconda3/etc/profile.d/conda.sh
conda activate ark

MODEL_TAG="${MODEL_TAG:-qwen3-vl:30b}"
MODEL_SAFE="${MODEL_TAG//:/_}"
SPECIES_FILTER="${SPECIES_FILTER:-}"

if [[ -n "${SPECIES_FILTER}" ]]; then
  mapfile -t TARGET_DIRS < <(
    find "results/${SPECIES_FILTER}" -type d -path "*/p5/predictions/${MODEL_SAFE}/*_I2I_P5_resolution_*" | sort
  )
else
  mapfile -t TARGET_DIRS < <(
    find results -type d -path "*/p5/predictions/${MODEL_SAFE}/*_I2I_P5_resolution_*" | sort
  )
fi

if [[ ${#TARGET_DIRS[@]} -eq 0 ]]; then
  if [[ -n "${SPECIES_FILTER}" ]]; then
    echo "No P5 resolution result directories found for species ${SPECIES_FILTER} and model ${MODEL_TAG} (${MODEL_SAFE})."
  else
    echo "No P5 resolution result directories found for model ${MODEL_TAG} (${MODEL_SAFE})."
  fi
  exit 1
fi

echo "Model: ${MODEL_TAG}"
echo "Species filter: ${SPECIES_FILTER:-<all>}"
echo "Matched run directories: ${#TARGET_DIRS[@]}"

python - "${MODEL_SAFE}" "${TARGET_DIRS[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

from scripts_evaluate.evaluate import evaluate_model_directory

model_safe = sys.argv[1]
target_dirs = [Path(p) for p in sys.argv[2:]]

summary_rows = []

print("=" * 105)
print(f"{'Species':<20} | {'Run':<35} | {'Acc(Str)':<9} | {'Acc(Ans)':<9} | {'Acc(Exp)':<9} | {'Corr':<5} | {'Total':<5}")
print("-" * 105)

for target_dir in target_dirs:
    species = target_dir.parts[1]
    metrics = evaluate_model_directory(target_dir, "p5", model_name=model_safe)
    if metrics is None:
        continue

    with open(target_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    summary_rows.append(
        {
            "species": species,
            "model": metrics["model"],
            "run_name": metrics["run_name"],
            "acc_strict": metrics["acc_strict"],
            "acc_answered": metrics["acc_answered"],
            "acc_expected": metrics["acc_expected"],
            "correct": metrics["correct"],
            "total": metrics["total"],
            "answered": metrics["answered"],
            "missing_extraction": metrics["missing_extraction"],
            "target_dir": str(target_dir),
        }
    )

    print(
        f"{species:<20} | {metrics['run_name']:<35} | "
        f"{metrics['acc_strict']:<9.2%} | {metrics['acc_answered']:<9.2%} | {metrics['acc_expected']:<9.2%} | "
        f"{metrics['correct']:<5} | {metrics['total']:<5}"
    )

print("=" * 105)

summary_json = Path("results") / f"p5_resolution_summary_{model_safe}.json"
summary_csv = Path("results") / f"p5_resolution_summary_{model_safe}.csv"

with open(summary_json, "w") as f:
    json.dump(summary_rows, f, indent=4)

fieldnames = [
    "species",
    "model",
    "run_name",
    "acc_strict",
    "acc_answered",
    "acc_expected",
    "correct",
    "total",
    "answered",
    "missing_extraction",
    "target_dir",
]
with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"Saved summary JSON to {summary_json}")
print(f"Saved summary CSV  to {summary_csv}")
PY

# bash evaluate_p5_resolution_all_species.sh
# SPECIES_FILTER=Lion bash evaluate_p5_resolution_all_species.sh
# MODEL_TAG=qwen3-vl:30b bash evaluate_p5_resolution_all_species.sh
