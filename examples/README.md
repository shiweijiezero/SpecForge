## 使用 Flex Attention 训练

Flex attention 节省 10 倍内存，同时也使训练更快。它目前处于实验阶段。要启用 flex attention，你需要向训练脚本传递 `--attention-backend flex_attention`。为了允许共享已编译的内核，你需要将 `TORCHINDUCTOR_CACHE_DIR` 设置为缓存目录。

> <b> 注意：确保你安装了 torch 2.8.0！</b>

示例训练脚本：
```bash
TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels \
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config $ROOT_DIR/configs/llama3-8B-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt.jsonl \
    --output-dir $ROOT_DIR/outputs/llama3-8b-eagle3 \
    --num-epochs 1 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template llama3 \
    --cache-dir $ROOT_DIR/cache
    --attention-backend flex_attention
```
