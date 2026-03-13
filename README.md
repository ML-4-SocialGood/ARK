# ARK Re-ID Evaluation Toolkit

This toolkit provides scripts to run inference and evaluate Re-ID (Re-identification) tasks using Vision-Language Models (VLMs) hosted via Ollama.

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

3.  **Logs**:
    *   Job output: `ditest.log`
    *   Ollama logs: `ollama_job.log`

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
python scripts_eval/run_inference.py \
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

**Output Location:** `results/<species>/<protocol>/predictions/<model_name>/`

---

## 2. Evaluate Results

The `evaluate.py` script parses the inference results, compares them against the ground truth, and calculates accuracy metrics. It also generates a detailed failure analysis report.

### Usage

```bash
python scripts_eval/evaluate.py --species <SPECIES> --protocol <PROTOCOL>
```

### Example

```bash
python scripts_eval/evaluate.py --species BelugaID --protocol p1
```

**Arguments:**
*   `--species`: Target species name.
*   `--protocol`: Evaluation protocol identifier.
*   `--model`: (Optional) Specify a single model name to evaluate. If omitted, all models found in the predictions directory will be evaluated.

**Outputs:**
*   **Console**: Displays a summary table of Accuracy, Correct Counts, and Totals.
*   **Summary File**: `results/<species>/<protocol>/evaluation_summary.json`
*   **Detailed Report**: `results/<species>/<protocol>/predictions/<model_name>/evaluation_details.csv` (Contains row-by-row comparisons for debugging).

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
