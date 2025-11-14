#!/bin/bash
# Medusa-1 Online Training for Qwen2.5 7B
# 参数设置与Eagle3对齐，保证公平对比

set -e

# ==================== 配置路径 ====================
ROOT_DIR=$(pwd)
TARGET_MODEL="Qwen/Qwen2.5-7B-Instruct"
DRAFT_CONFIG="${ROOT_DIR}/configs/medusa/qwen2.5-7B-medusa.json"
TRAIN_DATA="${ROOT_DIR}/cache/dataset/sharegpt.jsonl"
OUTPUT_DIR="${ROOT_DIR}/outputs/qwen25-7b-medusa"
VOCAB_MAPPING="${ROOT_DIR}/cache/vocab_mapping_qwen25.pt"

# ==================== 检查环境 ====================
echo "======================================================"
echo "Medusa-1 Training for Qwen2.5 7B"
echo "======================================================"
echo "Target Model: ${TARGET_MODEL}"
echo "Draft Config: ${DRAFT_CONFIG}"
echo "Training Data: ${TRAIN_DATA}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "======================================================"

# 检查配置文件
if [ ! -f "${DRAFT_CONFIG}" ]; then
    echo "❌ Error: Draft config not found at ${DRAFT_CONFIG}"
    echo "Please ensure configs/medusa/qwen2.5-7B-medusa.json exists"
    exit 1
fi

# 检查训练数据
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "❌ Error: Training data not found at ${TRAIN_DATA}"
    echo ""
    echo "Please prepare training data first"
    exit 1
fi

# ==================== 生成词表映射 ====================
if [ ! -f "${VOCAB_MAPPING}" ]; then
    echo "📝 Generating vocabulary mapping..."
    echo "This maps target vocab (152064) to draft vocab (16000)"

    python ${ROOT_DIR}/scripts/generate_vocab_mapping.py \
        --target-model-path ${TARGET_MODEL} \
        --draft-vocab-size 16000 \
        --output-path ${VOCAB_MAPPING}

    echo "✅ Vocabulary mapping saved to ${VOCAB_MAPPING}"
else
    echo "✅ Vocabulary mapping already exists: ${VOCAB_MAPPING}"
fi

# ==================== 训练参数说明 ====================
echo ""
echo "======================================================"
echo "Training Hyperparameters (aligned with Eagle3):"
echo "======================================================"
echo "Learning Rate:      1e-4  (same as Eagle3)"
echo "Batch Size:         1 per device (same as Eagle3)"
echo "Gradient Accum:     4 steps (effective batch=4)"
echo "Epochs:             10 (same as Eagle3)"
echo "Warmup Ratio:       0.015 (same as Eagle3)"
echo "Max Grad Norm:      0.5 (same as Eagle3)"
echo "Max Length:         2048 tokens"
echo ""
echo "Medusa-specific:"
echo "Num Heads:          4"
echo "Num Layers/Head:    1 (ResBlock layers)"
echo "Draft Backbone:     None (num_hidden_layers=0)"
echo ""
echo "Qwen-specific:"
echo "Hidden Size:        3584 (vs LLaMA 4096)"
echo "Draft Vocab:        16000 (vs LLaMA 32000)"
echo "Context Length:     32768 (vs LLaMA 8192)"
echo "======================================================"
echo ""

# ==================== 训练 ====================
echo "🚀 Starting Medusa-1 training..."
echo "Training with $(nvidia-smi --list-gpus | wc -l) GPUs"
echo ""

torchrun \
    --standalone \
    --nproc_per_node $(nvidia-smi --list-gpus | wc -l) \
    ${ROOT_DIR}/scripts/train_medusa_online.py \
    --target-model-path ${TARGET_MODEL} \
    --draft-model-config ${DRAFT_CONFIG} \
    --train-data-path ${TRAIN_DATA} \
    --output-dir ${OUTPUT_DIR} \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --num-heads 4 \
    --warmup-ratio 0.015 \
    --max-grad-norm 0.5 \
    --save-interval 5000 \
    --log-interval 50 \
    --chat-template qwen \
    --seed 42

# ==================== 训练完成 ====================
echo ""
echo "======================================================"
echo "✅ Training completed successfully!"
echo "======================================================"
echo "Model saved to: ${OUTPUT_DIR}/final"
echo ""
echo "To evaluate or use in SGLang:"
echo "  python -m sglang.launch_server \\"
echo "    --model-path ${TARGET_MODEL} \\"
echo "    --speculative-draft-model-path ${OUTPUT_DIR}/final \\"
echo "    --speculative-algorithm medusa \\"
echo "    --speculative-num-steps 4"
echo "======================================================"
