#!/bin/bash
# Medusa-1 Online Training for LLaMA 3.1 8B
# 严格按照Eagle3的方式，所有步骤完全硬编码
# 包含完整的数据准备流程

set -e

echo "======================================================"
echo "Medusa Training Pipeline for LLaMA 3.1 8B"
echo "======================================================"
echo "包含步骤:"
echo "  1. 下载模型和数据集"
echo "  2. 准备训练数据（合并ShareGPT + UltraChat）"
echo "  3. 用目标模型重新生成数据（可选，建议启用）"
echo "  4. 构建数据集缓存"
echo "  5. 训练Medusa模型"
echo "======================================================"
echo ""

# ==================== 步骤 1: 下载模型和数据集 ====================
echo "步骤 1/5: 下载模型和数据集"
echo "----------------------------------------------------"

hf download meta-llama/Llama-3.1-8B-Instruct
hf download Aeala/ShareGPT_Vicuna_unfiltered --repo-type dataset
hf download HuggingFaceH4/ultrachat_200k --repo-type dataset

echo "✅ 下载完成"
echo ""

# ==================== 步骤 2: 准备训练数据 ====================
echo "步骤 2/5: 准备训练数据"
echo "----------------------------------------------------"

python scripts/prepare_data.py --dataset sharegpt --output-path /tmp/dataset/
python scripts/prepare_data.py --dataset ultrachat --output-path /tmp/dataset/
cat /tmp/dataset/sharegpt.jsonl /tmp/dataset/ultrachat.jsonl > /tmp/dataset/sharegpt_ultrachat.jsonl

echo "✅ 数据准备完成"
echo ""

# ==================== 步骤 3: 用目标模型重新生成数据（可选但推荐）====================
echo "步骤 3/5: 用目标模型重新生成数据（可选）"
echo "----------------------------------------------------"
echo "⚠️  建议启用此步骤以获得更好的对齐效果"
echo "如需跳过，请注释掉以下部分"
echo ""

# 启动SGLang服务器（多卡并行）
for i in {0..1}; do
    CUDA_VISIBLE_DEVICES=$i python3 -m sglang.launch_server \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --cuda-graph-bs 1 2 4 8 16 32 64 128 256 512 \
        --dtype bfloat16 --mem-frac=0.7 --port $((30000 + i)) &
done

# 等待服务器启动
sleep 30

# 生成数据
python scripts/generate_data_by_target.py \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --raw-data-file /tmp/dataset/sharegpt_ultrachat.jsonl \
    --output-dir /tmp/generated-dataset/llama-3.1-8b-instruct \
    --max-concurrency 512 \
    --num-per-shard 50000 \
    --server-address-port 127.0.0.1:30000 127.0.0.1:30001

# 合并生成的数据分片
python scripts/merge_shards.py \
    --input-dir /tmp/generated-dataset/llama-3.1-8b-instruct \
    --output-dir /tmp/dataset \
    --eval-size 10000

# 关闭服务器
pkill -f "sglang.launch_server"

echo "✅ 数据重新生成完成"
echo ""

# ==================== 步骤 4: 构建数据集缓存 ====================
echo "步骤 4/5: 构建数据集缓存"
echo "----------------------------------------------------"

python scripts/build_eagle3_dataset_cache.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config configs/medusa/llama3-8B-medusa.json \
    --train-data-path /tmp/dataset/train.jsonl \
    --eval-data-path /tmp/dataset/eval.jsonl \
    --cache-dir /tmp/cache \
    --chat-template llama3 \
    --max-length 2048 \
    --build-dataset-num-proc 64 \
    --view-train-data 1 2

echo "✅ 缓存构建完成"
echo ""

# ==================== 步骤 5: 训练Medusa模型 ====================
echo "步骤 5/5: 训练Medusa模型"
echo "----------------------------------------------------"
echo "⚠️  所有参数与Eagle3严格对齐，保证公平对比"
echo ""
echo "对比 Eagle3 (run_llama3_eagle3_sgl_online.sh):"
echo "  Learning Rate:   5e-5  ← 来自 sgl_online.sh:58"
echo "  Batch Size:      1     ← 来自 sgl_online.sh:57"
echo "  Epochs:          1     ← 您的实验配置"
echo "  Warmup Ratio:    0.015 ← 来自 sgl_online.sh:65"
echo "  Max Grad Norm:   0.5   ← 来自 sgl_online.sh:66"
echo "  Max Length:      2048  ← 来自 sgl_online.sh:8"
echo "  Chat Template:   llama3← 来自 sgl_online.sh:9"
echo ""
echo "Medusa特有参数（算法差异，允许不同）:"
echo "  Num Heads:       4     ← Medusa论文Table 2推荐3-5"
echo "  Draft Layers:    0     ← Medusa无backbone"
echo "======================================================"
echo ""

torchrun \
    --standalone \
    --nproc_per_node $(nvidia-smi --list-gpus | wc -l) \
    scripts/train_medusa_online.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config configs/medusa/llama3-8B-medusa.json \
    --train-data-path /tmp/dataset/train.jsonl \
    --eval-data-path /tmp/dataset/eval.jsonl \
    --output-dir /tmp/outputs/llama3-8b-medusa \
    --cache-dir /tmp/cache \
    --num-epochs 1 \
    --batch-size 1 \
    --learning-rate 5e-5 \
    --max-length 2048 \
    --chat-template llama3 \
    --warmup-ratio 0.015 \
    --max-grad-norm 0.5 \
    --save-interval 5000 \
    --log-interval 50 \
    --seed 42 \
    --tp-size $(nvidia-smi --list-gpus | wc -l)

echo ""
echo "======================================================"
echo "✅ 训练完成！"
echo "======================================================"
echo "模型保存至: /tmp/outputs/llama3-8b-medusa/final"
echo ""
echo "SGLang推理命令:"
echo "  python -m sglang.launch_server \\"
echo "    --model-path meta-llama/Llama-3.1-8B-Instruct \\"
echo "    --speculative-draft-model-path /tmp/outputs/llama3-8b-medusa/final \\"
echo "    --speculative-algorithm medusa \\"
echo "    --speculative-num-steps 4"
echo "======================================================"
