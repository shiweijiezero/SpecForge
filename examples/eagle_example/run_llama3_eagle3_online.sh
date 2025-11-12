#!/bin/bash

# ============================================================================
# Llama3.1-8B-Instruct Eagle3 在线训练完整流程
# 参考: CLAUDE.md 中的 "Llama3 Eagle3 在线训练复现指南"
# ============================================================================

echo "============================================"
echo "Llama3.1-8B-Instruct Eagle3 在线训练流程"
echo "============================================"
echo "模型路径: meta-llama/Llama-3.1-8B-Instruct"
echo "数据集路径: ./cache/train_dataset/llama-3.1-8b-instruct"
echo "输出目录: ./model/Llama-3.1-8B-Instruct/dev_outputs"
echo "GPU 数量: 4"
echo "============================================"

# ---------- 步骤 1: 下载模型和数据集 ----------
echo ""
echo "步骤 1: 下载模型和数据集..."
hf download meta-llama/Llama-3.1-8B-Instruct
hf download Aeala/ShareGPT_Vicuna_unfiltered --repo-type dataset
hf download HuggingFaceH4/ultrachat_200k --repo-type dataset

# ---------- 步骤 2: 准备训练数据 ----------
echo ""
echo "步骤 2: 准备训练数据..."
mkdir -p ./cache/dataset
python scripts/prepare_data.py --dataset ultrachat --output-path ./cache/dataset
python scripts/prepare_data.py --dataset sharegpt --output-path ./cache/dataset
cat ./cache/dataset/sharegpt.jsonl ./cache/dataset/ultrachat.jsonl > ./cache/dataset/train_dataset.jsonl

# ---------- 步骤 3: 使用目标模型生成数据（推荐） ----------
echo ""
echo "步骤 3: 使用目标模型生成数据..."

# 启动 SGLang 服务器（使用 4 张卡）
echo "启动 SGLang 服务器..."
for i in {0..3}; do
    CUDA_VISIBLE_DEVICES=$i python3 -m sglang.launch_server \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --cuda-graph-bs 1 2 4 8 16 32 64 128 256 512 \
        --dtype bfloat16 --mem-frac=0.7 --port $((40000 + i)) &
done

# 等待服务器启动
echo "等待 SGLang 服务器启动（60秒）..."
sleep 60

# 生成数据
echo "开始生成数据..."
python scripts/generate_data_by_target.py \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --raw-data-file ./cache/dataset/train_dataset.jsonl \
    --output-dir ./cache/generated-dataset/llama-3.1-8b-instruct \
    --max-concurrency 256 \
    --num-per-shard 50000 \
    --server-address-port 127.0.0.1:40000 127.0.0.1:40001 127.0.0.1:40002 127.0.0.1:40003

# 合并生成的数据分片
echo "合并生成的数据分片..."
mkdir -p ./cache/train_dataset/llama-3.1-8b-instruct
python scripts/merge_shards.py \
    --input-dir ./cache/generated-dataset/llama-3.1-8b-instruct \
    --output-dir ./cache/train_dataset/llama-3.1-8b-instruct \
    --train-size 50000 \
    --eval-size 5000

# 关闭 SGLang 服务器
echo "关闭 SGLang 服务器..."
pkill -f "sglang.launch_server"
sleep 10

# ---------- 步骤 4: 构建数据集缓存 ----------
echo ""
echo "步骤 4: 构建数据集缓存..."
mkdir -p ./cache/preprocess_data_cache_dir/llama-3.1-8b-instruct
python scripts/build_eagle3_dataset_cache.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config ./configs/llama3-8B-eagle3.json \
    --train-data-path ./cache/train_dataset/llama-3.1-8b-instruct/train.jsonl \
    --eval-data-path ./cache/train_dataset/llama-3.1-8b-instruct/eval.jsonl \
    --cache-dir ./cache/preprocess_data_cache_dir/llama-3.1-8b-instruct \
    --chat-template llama3 \
    --max-length 4096 \
    --build-dataset-num-proc 64 \
    --view-train-data 1 2

# ---------- 步骤 5: 开始训练 ----------
echo ""
echo "步骤 5: 开始 Eagle3 在线训练..."
mkdir -p ./model/Llama-3.1-8B-Instruct/dev_outputs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --standalone \
    --nproc_per_node 4 \
    scripts/train_eagle3_online.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config ./configs/llama3-8B-eagle3.json \
    --train-data-path ./cache/train_dataset/llama-3.1-8b-instruct/train.jsonl \
    --eval-data-path ./cache/train_dataset/llama-3.1-8b-instruct/eval.jsonl \
    --tp-size 4 \
    --output-dir ./model/Llama-3.1-8B-Instruct/dev_outputs \
    --num-epochs 1 \
    --batch-size 1 \
    --learning-rate 5e-5 \
    --attention-backend flex_attention \
    --max-length 4096 \
    --chat-template llama3 \
    --cache-dir ./cache/preprocess_data_cache_dir/llama-3.1-8b-instruct \
    --dist-timeout 10 \
    --wandb-project llama3-8b \
    --wandb-name eagle3-online \
    --report-to wandb \
    --target-model-backend sglang

# ---------- 步骤 6: 基准测试（可选） ----------
echo ""
echo "步骤 6: 运行基准测试..."
echo "如果需要进行基准测试，请取消注释以下部分："
echo ""

# # 定义推理配置列表
config_list=(
 "4,0,0,0"      # baseline: 无投机解码，这个配置只用一次就行了
 "4,3,1,4"      # 配置1
 "4,7,10,60"    # 配置2
)
#
config_list=(
 "4,3,1,4"      # 配置1 batch=4, steps=3, topk=1, draft_tokens=4
 "4,7,10,60"    # 配置2 batch=4, steps=7, topk=10, draft_tokens=60
)
# # 运行 benchmark（使用硬编码路径避免环境变量问题）
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 CUDA_VISIBLE_DEVICES=4 python benchmarks/bench_model_speedup.py \
 --model-path meta-llama/Llama-3.1-8B-Instruct \
 --speculative-draft-model-path ./model/Llama-3.1-8B-Instruct/dev_outputs/epoch_0_step_10000 \
 --port 30000 \
 --enable-multi-turn-conversation \
 --trust-remote-code \
 --mem-fraction-static 0.7 \
 --tp-size 1 \
 --config-list "${config_list[@]}" \
 --benchmark-list mtbench:80 gsm8k:200 humaneval:164 math500:200 ceval:200 cmmlu:200 \
 --temperature 0.0 0.7 1.0 1.2 \
 --output results/llama3_results.jsonl

echo ""
echo "============================================"
echo "训练完成！"
echo "模型输出目录: ./model/Llama-3.1-8B-Instruct/dev_outputs"
echo "============================================"
