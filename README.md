# ARK Re-ID Evaluation Toolkit

This toolkit provides scripts to run inference and evaluate Re-ID (Re-identification) tasks using Vision-Language Models (VLMs) hosted via Ollama.

## Evaluation Framework

The ARK evaluation framework is systematically designed across three main axes to comprehensively assess the capabilities of Vision-Language Models (VLMs) in Re-ID tasks:

### Axis 1: Visual Perception & Feature Alignment
**Positioning:** This serves as the foundational baseline. It establishes the pure visual matching capability of VLMs without complex logical constraints.
*   **Protocol 1: Pure Visual Re-Identification (P1)**
    *   *Core Capability:* Zero-shot/few-shot discrimination of fine-grained features.
    *   *Narrative Value:* Demonstrates that large models still have shortcomings in basic cross-instance matching, motivating the introduction of "reasoning" as a necessary solution.

### Axis 2: Information Integration & Logical Reasoning
**Positioning:** This represents the core contribution area. It demonstrates how VLMs transcend single-image pattern matching by utilizing context, multi-source evidence, and external constraints to perform complex identity associations.
*   **Protocol 2: Multi-View Identity Integration (P2)**
    *   *Core Capability:* Cross-image evidence supplementation and invariant feature extraction. Evaluates whether the model can piece together a complete profile of an individual from multiple query images.
*   **Protocol 3: Multi-Target Identity Association (P3)**
    *   *Core Capability:* Semantic induction. Breaks the "single-choice" paradigm to find all visual variants of the same underlying identity within a complex gallery.
*   **Protocol 4: Metadata-Constrained Logical Reasoning (P4)**
    *   *Core Capability:* Cross-modal logical verification. Utilizes ecological metadata (e.g., time, location, day/night) as hard constraints to eliminate visually highly similar but logically invalid "false positives."

### Axis 3: Robustness & Reliability in the Wild
**Positioning:** Specifically targets long-tail issues and uncontrollable risks in real-world scenarios like wildlife monitoring. This evaluates the model's readiness for actual deployment.
*   **Protocol 5: Corrupted Feature Completion (P5)**
    *   *Core Capability:* Robustness under extremely low signal-to-noise ratios. Tests the model's ability to maintain identity coherence relying on local topological structures despite occlusion, low resolution, or abrupt lighting changes.
*   **Protocol 6: Open-Set Hallucination Rejection (P6)**
    *   *Core Capability:* Epistemic uncertainty. Evaluates the ability to answer "None of the above" when the query individual is absent from the gallery, which is crucial for controlling false positive rates.
*   **Protocol 7: Resilience to Counterfactual Suggestion (P7)**
    *   *Core Capability:* Independent critical thinking. Tests whether the model falls prey to compliance hallucination when presented with misleading prompts (e.g., incorrect expert assumptions).

---

## Prerequisites

1.  **Ollama**: Ensure Ollama is installed.
    *   **HPC/Slurm Usage**: The provided `main.sh` script handles the Ollama server automatically (recommended for clusters).
    *   **Local Usage**: You will need to run `ollama serve` manually.

2.  **Python Dependencies**: Install the required Python packages.
    ```bash
    pip install requests tenacity tqdm pillow
    ```

## 1. Run Inference

The `run_inference.py` script connects to the Ollama instance, processes the tasks defined in the annotation file (including images), and saves the structured JSON results.

### Method A: HPC / Slurm (Recommended)

When running on a cluster, use `main.sh`. This script automatically starts a **private Ollama instance** for your job, ensuring proper GPU access and avoiding port conflicts.

1.  **Configure `main.sh`**:
    Edit the script to specify your model path (if needed) and inference arguments.
    ```bash
    # Inside main.sh:
    export OLLAMA_HOST=127.0.0.1:11435           # Custom port
    # export OLLAMA_MODELS=/path/to/models       # Uncomment if using custom model path

    # Check the python command line for your specific task
    python scripts_evaluate/run_inference.py ...
    ```

2.  **Submit Job**:
    ```bash
    sbatch main.sh
    ```

3.  **Logs**: Logs are centrally managed to keep the workspace clean.
    *   **Slurm Job output**: `logs/slurm_ARK_<JOB_ID>.out` / `.err`
    *   **Ollama background logs**: `logs/ollama_job_<JOB_ID>.log`
    *   **Inference script log**: `logs/<species>/<protocol>/inference_<model_name>.log`
    *   **Evaluation script log**: `logs/<species>/<protocol>/evaluation.log`

### Method B: Manual / Local Usage

```bash
python scripts_evaluate/run_inference.py \
    --species <SPECIES> \
    --protocol <PROTOCOL> \
    --annotation_file <PATH_TO_JSON> \
    --model <MODEL_NAME>
```

### Example

```bash
python scripts_evaluate/run_inference.py \
    --species BelugaID \
    --protocol p1 \
    --annotation_file annotations/BelugaID/p1/BelugaID_I2I_P1_N4.json \
    --model qwen3-vl:4b
```

**Arguments:**
*   `--species`: Target species name (e.g., `BelugaID`). Used for organizing output directories.
*   `--protocol`: Evaluation protocol identifier (e.g., `p1`).
*   `--annotation_file`: Path to the ground truth JSON file containing the MCQ tasks.
*   `--model`: The Ollama model tag to use (e.g., `qwen3-vl:4b`).
*   `--resume`: (Optional) Add this flag to skip tasks that have already been processed in the output folder.
*   `--host`: (Optional) Ollama API host (default: `http://localhost:11434`).

**Output Location:** `results/<species>/<protocol>/predictions/<model_name>/<annotation_basename>/` (Automatically grouped by annotation file to separate different runs or conditions, e.g., P5 corruption types).

---

## 2. Evaluate Results

The `evaluate.py` script parses the inference results, compares them against the ground truth, and calculates accuracy metrics (including Precision, Recall, and F1-Score for multi-target protocols like P3). It also generates a detailed failure analysis report.

### Usage

```bash
python scripts_evaluate/evaluate.py --species <SPECIES> --protocol <PROTOCOL>
```

### Example

```bash
python scripts_evaluate/evaluate.py --species BelugaID --protocol p1
```

**Arguments:**
*   `--species`: Target species name.
*   `--protocol`: Evaluation protocol identifier.
*   `--model`: (Optional) Specify a single model name to evaluate. If omitted, all models found in the predictions directory will be evaluated.

**Outputs:**
*   **Console**: Displays a summary table with multidimensional accuracy metrics:
    *   `Condition / Run`: Automatically detected from subdirectories, allowing easy comparison across different testing conditions (e.g., `grayscale_s1` vs `occlusion_s2`).
    *   `Acc(Str)`: Strict Accuracy (Unformatted/Null outputs are treated as incorrect).
    *   `Acc(Ans)`: Answered Accuracy (Accuracy calculated *only* on tasks the model successfully formatted/answered).
    *   `Acc(Exp)`: Expected Accuracy (Null outputs are given 0.25 fractional points, simulating 4-choice random guessing).
    *   Includes Correct/Total counts for single-choice tasks, and Precision, Recall, F1-Score for multi-target tasks (P3).
*   **Summary File**: `results/<species>/<protocol>/evaluation_summary.json`
*   **Model Metrics**: `results/<species>/<protocol>/predictions/<model_name>/<run_name>/metrics.json` (Individual run performance).
*   **Detailed Report**: `results/<species>/<protocol>/predictions/<model_name>/<run_name>/evaluation_details.csv` (Contains row-by-row comparisons for debugging).

---

## Appendix: Ollama Remote Server Deployment Guide (Custom Path)

This guide documents how to deploy Ollama on a remote server (e.g., research cluster) without root access, using a custom path for storage efficiency.

### 1. Installation & Path Configuration
To keep the `/home` directory clean and utilize data disks effectively:

```bash
# 1. Create and enter target directory
export INSTALL_DIR="/data/dzha866/software/ollama"
mkdir -p $INSTALL_DIR && cd $INSTALL_DIR

# 2. Download binary package (x86_64)
curl -L https://ollama.com/download/ollama-linux-amd64.tar.zst -o ollama-linux-amd64.tar.zst

# 3. Extract to current directory
# (Ensure zstd is installed or use unzstd if tar support is missing)
tar --zstd -xf ollama-linux-amd64.tar.zst -C .

# 4. Configure PATH (Add this to ~/.bashrc)
echo "export PATH=\"$INSTALL_DIR/bin:\$PATH\"" >> ~/.bashrc
source ~/.bashrc
```

### 2. Model Storage Configuration
Ollama defaults to storing models on the system disk. For large models (e.g., Qwen-32B), redirect storage to the data partition:

```bash
# Set model storage location
export OLLAMA_MODELS="/data/dzha866/software/ollama/models"
mkdir -p $OLLAMA_MODELS
```

### 3. Start Service in Background
Use `nohup` to keep the service running after SSH disconnects (useful for manual debugging or model downloading).

```bash
nohup ollama serve > $INSTALL_DIR/ollama.log 2>&1 &
```

### 4. Critical: Bypass Network Proxy
If the server uses a global proxy (e.g., Squid), accessing localhost might fail (503 error). Set `no_proxy` before running commands.

**Terminal:**
```bash
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"
```

**Or in Python Code:**
```python
import os
os.environ['no_proxy'] = 'localhost,127.0.0.1'
```

### 5. Common Commands
*   **Verify**: `ollama --version`
*   **Check GPU**: Look for "discovering available GPUs" in `ollama.log`.
*   **Download Model**: `ollama pull qwen3-vl:32b` (or `qwen2.5-vl:7b`)
*   **Check Running**: `ollama ps`
*   **Stop Service**: `pkill ollama`

### 6. Slurm Job Management Commands
*   **Monitor Jobs**: `watch -n 2 squeue -u <username>` (Auto-refreshes every 2 seconds)
*   **Check Status**: `squeue -u <username>` (Lists your current queued or running jobs)
*   **Cancel Job**: `scancel <JOB_ID>` (Stops a specific job using its ID)
*   **Cancel All Jobs**: `scancel -u <username>` (Stops all your running and pending jobs)
*   **View Logs**: `tail -f <logfile.log>` (Watches job output logs in real-time)
