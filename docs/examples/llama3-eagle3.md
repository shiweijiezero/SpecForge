# Llama3 的 Eagle3

## 简介

本文档提供了一个分步指南,用于复现 EAGLE3 论文中描述的训练过程,使用脚本 `examples/run_llama3_eagle3_sgl_online.sh`。我们将逐步讲解该脚本并解释每个关键步骤。

## 工作流程

### 步骤 1. 准备环境

我们建议使用虚拟环境以确保所有依赖项都能正确安装。如果您想使用 `python>=3.12`,请设置 `export SETUPTOOLS_USE_DISTUTILS=local`。

```shell
uv venv --python 3.11
source .venv/bin/activate
cd PATH-TO-SpecForge
uv pip install -r requirements.txt
uv pip install -v .
```

完成这些步骤后,您可以通过运行以下命令来检查安装是否成功。如果安装成功,您应该不会看到任何错误。

```shell
python -c "import specforge"
```

### 步骤 2. 准备模型和数据集

接下来,我们可以开始准备模型和数据集。首先,使用以下命令下载模型和数据集。

```shell
hf download meta-llama/Llama-3.1-8B-Instruct
hf download Aeala/ShareGPT_Vicuna_unfiltered --repo-type dataset
hf download HuggingFaceH4/ultrachat_200k --repo-type dataset

python scripts/prepare_data.py --dataset ultrachat --output_path /YOUR/PATH/Llama-3.1-8B-Instruct/dataset
python scripts/prepare_data.py --dataset sharegpt --output_path /YOUR/PATH/Llama-3.1-8B-Instruct/dataset
```

然后,启动 SGLang 服务器并运行 `generate_data_by_target.py` 以在不同数据集上从基础模型生成响应。请确保更新 `generate_data_by_target.py` 中的 `SYSTEM_PROMPT` 值以满足您的需求。

```shell
for i in {1..4}; do
    CUDA_VISIBLE_DEVICES=$i python3 -m sglang.launch_server \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --cuda-graph-bs 1 2 4 8 16 32 64 128 256 512 \
        --dtype bfloat16 --mem-frac=0.8 --port $((30000 + i)) &
done

python scripts/generate_data_by_target.py \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --raw-data-file /YOUR/PATH/Llama-3.1-8B-Instruct/dataset/sharegpt.jsonl \
    --output-dir /YOUR/PATH/Llama-3.1-8B-Instruct/generated-dataset/sharegpt-llama-3.1-8b-instruct \
    --max-concurrency 512 \
    --num-per-shard 50000 \
    --server-address-port 127.0.0.1:30001 127.0.0.1:30002 127.0.0.1:30003 127.0.0.1:30004

python scripts/generate_data_by_target.py \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --raw-data-file /YOUR/PATH/Llama-3.1-8B-Instruct/dataset/ultrachat.jsonl \
    --output-dir /YOUR/PATH/Llama-3.1-8B-Instruct/generated-dataset/ultrachat-llama-3.1-8b-instruct \
    --max-concurrency 512 \
    --num-per-shard 50000 \
    --server-address-port 127.0.0.1:30001 127.0.0.1:30002 127.0.0.1:30003 127.0.0.1:30004
```

完成这些步骤后,您可以查看 `error.jsonl` 中的错误条目。其中大多数可能是 `request timeout`。然后您可以决定是否要重新生成这些样本。在我的情况下,我选择不重新生成,所以在上传到 Hugging Face 之前直接删除了 error.jsonl。使用以下命令:

```shell
hf repo create zhuyksir/Ultrachat-Sharegpt-Llama3.1-8B --type dataset
hf upload /YOUR/PATH/Llama-3.1-8B-Instruct/generated-dataset/ultrachat-llama-3.1-8b-instruct --commit-message "generated dataset by Llama3.1-8B"
```

```python
from datasets import load_dataset
ds = load_dataset("zhuyksir/Ultrachat-Sharegpt-Llama3.1-8B", split="train")
ds.to_json("merged.jsonl", orient="records", lines=True)
ds = ds.train_test_split(test_size=0.05)
train_ds = ds["train"]
test_ds = ds["test"]
```

或者,对于 `meta-llama/Llama-3.1-8B-Instruct`,您可以使用我们生成的数据集:[zhuyksir/Ultrachat-Sharegpt-Llama3.1-8B](https://huggingface.co/datasets/zhuyksir/Ultrachat-Sharegpt-Llama3.1-8B)。

每一行应该具有以下结构:
```json
{
    "id": XXX,
    "conversations":[
        {"role": "system", "content": XXX},
        {"role": "user", "content": XXX},
        {"role": "assistant", "content": XXX},
        ...
    ]
}
```

其次,我们需要为训练预先构建缓存。

- 在训练过程中,文本必须被编码为输入 ID。这些编码步骤可以在训练开始前执行。生成的缓存文件将保存在 `$CACHE_DIR` 下。
- 该脚本还会选择具有 top-k 大小的词汇表。
- 使用选项 `--view train-data`,您可以通过索引检查数据集(例如,下面示例中的索引 1 或索引 2)。这有助于验证损失掩码是否正确生成:
    - 绿色文本表示 `loss_mask == 1` 的标记。
    - 红色文本表示 `loss_mask == 0` 的标记(通常是用户输入和系统提示)。由于目标是仅在目标模型的输出上训练草稿模型,因此必须屏蔽用户文本。换句话说,只有目标模型生成的标记才应该对损失有贡献。

- 您可能会看到此警告。`WARNING: No assistant response spans found in the conversation text.` 当在数据生成期间,错误导致样本仅包含用户输入而没有任何助手响应时,就会发生这种情况。您可以安全地忽略此警告——此类样本的损失掩码完全设置为零。

```shell
python scripts/build_eagle3_dataset_cache.py \
    --target-model-path $MODEL_PATH \
    --draft-model-config ./configs/llama3-8B-eagle3.json \
    --train-data-path $DATASET_PATH/sharegpt_ultrachat_train.jsonl \
    --eval-data-path $DATASET_PATH/sharegpt_ultrachat_test.jsonl \
    --cache-dir $CACHE_DIR \
    --chat-template $CHAT_TEMPLATE \
    --max-length $MAX_LENGTH \
    --view-train-data 1 2
```

### 步骤 3. 开始训练

使用以下脚本进行训练。

- 设置 `total-steps=800000, learning-rate=5e-5` 以与 [EAGLE 官方仓库配置](https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/ds_config.json)保持一致。可以随意更改这些设置来进行您自己的实验。`total-steps` 和 `warmup-ratio` 决定学习率的增长曲线。

```shell
export NUM_GPUS=4
export OUTPUT_DIR=/YOUR/PATH/Llama-3.1-8B-Instruct/dev_outputs/
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    scripts/train_eagle3_sgl_online.py \
    --target-model-path $MODEL_PATH \
    --model-path $MODEL_PATH \
    --draft-model-config ./configs/llama3-8B-eagle3.json \
    --train-data-path $DATASET_PATH/sharegpt_ultrachat_train.jsonl \
    --eval-data-path $DATASET_PATH/sharegpt_ultrachat_test.jsonl \
    --tp-size $NUM_GPUS \
    --output-dir $OUTPUT_DIR \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 5e-5 \
    --draft-attention-backend flex_attention \
    --max-length $MAX_LENGTH \
    --chat-template $CHAT_TEMPLATE \
    --cache-dir $CACHE_DIR \
    --mem-frac=0.4 \
    --total-steps=800000 \
    --dist-timeout=10 \
    --wandb-project llama3-8b-eagle3 \
    --wandb-name sgl-online \
    --report-to wandb
```

### 步骤 4. 基准测试

对于 `Llama3.1-8B`,我们在所有训练数据中添加了系统提示,遵循官方仓库中使用的方法。因此,在进行基准测试时,我们也应该包含此系统提示以获得完整的接受长度。请取消注释相应的行并添加系统提示。

配置中的四个数字代表:`batch_size, num_steps, topk, num_verify_tokens`。您可以调整配置列表中的值来尝试不同的测试用例。

我已经在 [zhuyksir/EAGLE3-Llama-3.1-8B-Instruct](https://huggingface.co/zhuyksir/EAGLE3-Llama-3.1-8B-Instruct) 上传了我训练的 eagle 模型。欢迎您下载并检查其接受长度。

```shell
config_list=(
    "4,3,1,4"
    "4,7,10,60"
)
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 bench_model_speedup.py \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --speculative-draft-model-path /YOUR/PATH/Llama-3.1-8B-Instruct/dev_outputs/epoch_0 \
    --port 20001 \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --tp-size 4 \
    --config-list "${config_list[@]}" \
    --benchmark-list mtbench:80 gsm8k:200 humaneval:200 math500:200 \
    --output output.jsonl
```
