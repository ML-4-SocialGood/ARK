# ARK Evaluation Toolkit Guide

This guide provides practical instructions for running inference and evaluation
with the ARK annotation files. It is intended as a companion to the main README.

## Environment

Install the Python dependencies:

```bash
conda create -n ark python=3.10 -y
conda activate ark
pip install requests tenacity tqdm pillow pandas matplotlib openpyxl
```

For local open-source MLLM inference, install Ollama and start the service:

```bash
ollama serve
ollama pull qwen3-vl:30b
```

If your server uses a proxy, make sure local Ollama traffic bypasses it:

```bash
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"
```

## Running Inference

Use `scripts_evaluate/run_inference.py` to process one annotation file:

```bash
python scripts_evaluate/run_inference.py \
  --species BelugaID \
  --protocol p1 \
  --annotation_file annotations/BelugaID/p1/BelugaID_I2I_P1_N4.json \
  --model qwen3-vl:30b \
  --host http://localhost:11434 \
  --resume
```

Important arguments:

| Argument | Description |
|---|---|
| `--species` | Species name used for output organization. |
| `--protocol` | Protocol identifier, such as `p1`, `p3`, or `p6`. |
| `--annotation_file` | JSON annotation file containing ARK MCQs. |
| `--model` | Ollama model tag. Colons are converted to underscores in output paths. |
| `--host` | Ollama API host. |
| `--resume` | Skip tasks already present in the prediction directory. |

Outputs are stored under:

```text
results/<species>/<protocol>/predictions/<model_name>/<annotation_basename>/
```

## Slurm Usage

For cluster jobs, `main.sh` can be adapted to start a private Ollama server per
job. The key pattern is to allocate a job-specific local port, start `ollama
serve`, and pass the resulting host to `run_inference.py`.

```bash
sbatch main.sh
```

Typical runtime artifacts:

```text
logs/slurm_ARK_<JOB_ID>.out
logs/slurm_ARK_<JOB_ID>.err
logs/ollama_job_<JOB_ID>.log
logs/<species>/<protocol>/inference_<model>_<run>.log
```

## Evaluating Results

After inference, evaluate one species/protocol pair:

```bash
python scripts_evaluate/evaluate.py \
  --species BelugaID \
  --protocol p1 \
  --model qwen3-vl:30b
```

If `--model` is omitted, the evaluator scans all available model directories for
that species and protocol.

Evaluation outputs:

```text
results/<species>/<protocol>/evaluation_summary.json
results/<species>/<protocol>/predictions/<model>/<run>/metrics.json
results/<species>/<protocol>/predictions/<model>/<run>/evaluation_details.csv
```

## Metrics

| Protocol | Metrics |
|---|---|
| P1, P2, P4, P5 | Accuracy |
| P3 | Precision, Recall, F1 |
| P6 | In-Gallery Accuracy, Out-Gallery Accuracy |
| P7 | Neutral Accuracy, Counterfactual Accuracy |

The evaluator also reports answered accuracy and expected accuracy to separate
formatting/parsing failures from substantive mistakes.

## Annotation Generation and Verification

Protocol-specific generation and verification scripts live under:

```text
scripts_annotate/p1/
scripts_annotate/p2/
scripts_annotate/p3/
scripts_annotate/p4/
scripts_annotate/p5/
scripts_annotate/p6/
scripts_annotate/p7/
```

Typical commands:

```bash
python scripts_annotate/p1/run_all.py
python scripts_annotate/p5/verify_dataset.py --dataset_name BelugaID --data_root .
```

The sampler uses identity-balanced query selection and dynamic distractor
sampling to reduce redundant negative combinations.
