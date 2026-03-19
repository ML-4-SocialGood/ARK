#!/bin/bash
#SBATCH --job-name=ARK
#SBATCH --time=9999:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/slurm_ARK_%j.out
#SBATCH --error=logs/slurm_ARK_%j.err
#SBATCH --gres=gpu:1

cd /data/dzha866/ARK

source /data/dzha866/miniconda3/etc/profile.d/conda.sh
conda activate ARK

# 1. 使用 Job ID 和 RANDOM 组合生成一个动态端口，避免与节点上现有的服务冲突
# 端口范围大致在 10000 到 29998 之间
OLLAMA_PORT=$(( 10000 + (${SLURM_JOB_ID:-0} % 10000) + RANDOM % 10000 ))
export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT
# 如果您的模型保存在非默认路径（例如不在 ~/.ollama/models），请取消注释并修改下行：
export OLLAMA_MODELS=/data/dzha866/software/ollama/models

# 2. 在后台启动 Ollama 服务，这样它能继承当前作业的 GPU 权限
ollama serve > logs/ollama_job_${SLURM_JOB_ID:-local}.log 2>&1 &
OLLAMA_PID=$!

# 3. 等待服务启动 (循环检查端口直到服务就绪)
echo "Waiting for Ollama to start..."
for i in {1..60}; do
    if curl -s http://127.0.0.1:$OLLAMA_PORT > /dev/null; then echo "Ollama started!"; break; fi
    sleep 2
done

# 调试信息：列出当前可用的模型，请检查您的模型是否在列表中
echo "Available models:"
ollama list

# 4. 运行推理脚本，并通过 --host 参数指向我们刚启动的端口
# 请确保 --model 参数与上面 ollama list 显示的名称完全一致（例如 qwen3-vl:32b）
python scripts_evaluate/run_inference.py --species BelugaID --protocol p1 --annotation_file annotations/BelugaID/p1/BelugaID_I2I_P1_N4.json --model gemma3:27b --host http://localhost:$OLLAMA_PORT

# 5. 作业结束后清理后台进程
kill $OLLAMA_PID