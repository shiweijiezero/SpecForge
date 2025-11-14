#!/bin/bash
# Medusa-1 Online Training for Qwen2.5 7B
# 参数设置与Eagle3对齐，保证公平对比

set -e

# ==================== 配置路径 ====================
ROOT_DIR=$(pwd)
TARGET_MODEL="Qwen/Qwen2.5-7B-Instruct"
DRAFT_CONFIG="${ROOT_DIR}/configs/medusa/qwen2.5-7B-medusa.json"

# ⚠️ 重要：使用与Eagle3完全相同的训练数据！
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
echo "Medusa Training for Qwen2.5 7B"
echo "======================================================"
echo "⚠️  Parameters aligned with Eagle3 for fair comparison"
echo ""
echo "Training Data:     ${TRAIN_DATA}"
if [ -f "${TRAIN_DATA}" ]; then
    echo "Data Size:         $(wc -l < ${TRAIN_DATA}) samples"
fi
echo ""
echo "Hyperparameters:"
echo "  Learning Rate:   5e-5  (Eagle3: 5e-5)"
echo "  Batch Size:      1 per device"
echo "  Epochs:          1  ← Aligned with Eagle3 baseline"
echo "  Warmup Ratio:    0.015"
echo "  Max Grad Norm:   0.5"
echo "  Max Length:      2048"
echo ""
echo "Medusa Config:"
echo "  Num Heads:       4"
echo "  Draft Layers:    0 (no backbone)"
echo ""
echo "Qwen-specific:"
echo "  Hidden Size:     3584 (vs LLaMA 4096)"
echo "  Draft Vocab:     16000 (vs LLaMA 32000)"
echo "======================================================"
echo ""

# ==================== 训练 ====================
echo "🚀 Starting Medusa-1 training..."
echo "Training with $(nvidia-smi --list-gpus | wc -l) GPUs"
echo ""

# ==================== 训练命令 ====================
# 参数说明：
# --num-epochs 1:      与Eagle3基线对齐
# --learning-rate 5e-5: 与Eagle3对齐
# --batch-size 1:      与Eagle3对齐
# --num-heads 4:       Medusa配置

torchrun \
    --standalone \
    --nproc_per_node $(nvidia-smi --list-gpus | wc -l) \
    ${ROOT_DIR}/scripts/train_medusa_online.py \
    --target-model-path ${TARGET_MODEL} \
    --draft-model-config ${DRAFT_CONFIG} \
    --train-data-path ${TRAIN_DATA} \
    --output-dir ${OUTPUT_DIR} \
    --num-epochs 1 \
    --batch-size 1 \
    --learning-rate 5e-5 \
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
