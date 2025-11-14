#!/bin/bash
# Medusa-1 Online Training for LLaMA 3.1 8B
# 严格按照Eagle3的方式，所有参数硬编码

set -e

# ==================== 训练参数对比 ====================
echo "======================================================"
echo "Medusa Training for LLaMA 3.1 8B"
echo "======================================================"
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

# ==================== 训练命令（完全硬编码）====================
# 词表映射会在训练脚本内部自动生成（与Eagle3一致）
# 不需要在shell脚本中预先生成

torchrun \
    --standalone \
    --nproc_per_node $(nvidia-smi --list-gpus | wc -l) \
    scripts/train_medusa_online.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config configs/medusa/llama3-8B-medusa.json \
    --train-data-path /tmp/dataset/sharegpt_ultrachat.jsonl \
    --output-dir /tmp/outputs/llama3-8b-medusa \
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
echo "✅ Training completed!"
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
