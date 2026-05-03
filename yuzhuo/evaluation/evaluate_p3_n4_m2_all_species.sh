#!/bin/bash

set -euo pipefail

cd /data/yil708/ARK

source /data/yil708/miniconda3/etc/profile.d/conda.sh
conda activate ark

MODEL_TAG="${MODEL_TAG:-qwen3-vl:30b}"
MODEL_SAFE="${MODEL_TAG//:/_}"

mapfile -t TARGET_DIRS < <(
  find results -type d -path "*/p3/predictions/${MODEL_SAFE}/*_MIA_P3_N4_M2" | sort
)

if [[ ${#TARGET_DIRS[@]} -eq 0 ]]; then
  echo "No P3_N4_M2 result directories found for model ${MODEL_TAG} (${MODEL_SAFE})."
  exit 1
fi

echo "Model: ${MODEL_TAG}"
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

print("=" * 110)
print(f"{'Species':<20} | {'Run':<30} | {'Acc(Str)':<9} | {'Acc(Ans)':<9} | {'Acc(Exp)':<9} | {'Prec':<7} | {'Recall':<7} | {'F1':<7}")
print("-" * 110)

for target_dir in target_dirs:
    species = target_dir.parts[1]
    metrics = evaluate_model_directory(target_dir, "p3", model_name=model_safe)
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
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "correct": metrics["correct"],
            "total": metrics["total"],
            "answered": metrics["answered"],
            "missing_extraction": metrics["missing_extraction"],
            "target_dir": str(target_dir),
        }
    )

    print(
        f"{species:<20} | {metrics['run_name']:<30} | "
        f"{metrics['acc_strict']:<9.2%} | {metrics['acc_answered']:<9.2%} | {metrics['acc_expected']:<9.2%} | "
        f"{metrics['precision']:<7.2%} | {metrics['recall']:<7.2%} | {metrics['f1_score']:<7.2%}"
    )

print("=" * 110)

summary_json = Path("results") / f"p3_n4_m2_summary_{model_safe}.json"
summary_csv = Path("results") / f"p3_n4_m2_summary_{model_safe}.csv"

with open(summary_json, "w") as f:
    json.dump(summary_rows, f, indent=4)

fieldnames = [
    "species",
    "model",
    "run_name",
    "acc_strict",
    "acc_answered",
    "acc_expected",
    "precision",
    "recall",
    "f1_score",
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

# bash evaluate_p3_n4_m2_all_species.sh
# MODEL_TAG=qwen3-vl:30b bash evaluate_p3_n4_m2_all_species.sh
