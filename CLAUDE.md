# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

SpecForge 是一个用于训练投机解码模型（speculative decoding models，具体为 Eagle3 草稿模型）的框架，训练出的模型可直接兼容 SGLang 推理框架。项目支持在线和离线两种训练模式，并提供 FSDP 和张量并行能力。

**核心概念：**
- **草稿模型（Draft Model）**：正在训练的轻量级投机模型（Eagle3）
- **目标模型（Target Model）**：冻结的大型模型，用于生成训练所需的隐藏状态
- **TTT（Test-Time Training）**：一种训练技术，在训练过程中将草稿模型展开多次迭代
- **辅助隐藏状态（Aux Hidden States）**：从目标模型的 3 个层（低层、中层、高层）提取的隐藏状态，拼接后投影用于草稿模型输入
- **在线训练（Online Training）**：训练时实时生成隐藏状态（更高 GPU 内存占用，更低磁盘占用）
- **离线训练（Offline Training）**：隐藏状态预先生成并缓存到磁盘（更低 GPU 内存占用，ultrachat+sharegpt 约需 12TB 磁盘空间）

## 安装与设置

```bash
# 从源码安装（推荐）
pip install -v .

# 安装 pre-commit hooks
pre-commit install
```

**依赖项**：需要 torch>=2.8.0、transformers>=4.57.1、sglang[all]>=0.5.4

## 常用命令

### 运行测试

```bash
# 运行所有测试
python -m unittest discover -s ./tests -p "test_*.py" -v

# 运行特定测试文件
python -m unittest tests/test_draft_modeling/test_llama3.py -v

# 运行单个测试用例
python -m unittest tests.test_draft_modeling.test_llama3.TestLlama3DraftModel.test_forward -v
```

### 代码检查与格式化

```bash
# 对所有文件运行 pre-commit
pre-commit run --all-files

# 使用 black 格式化
black .

# 使用 isort 排序导入
isort .
```

### 数据准备

```bash
# 准备在线训练的示例数据集（ultrachat/sharegpt）
python scripts/prepare_data.py --dataset ultrachat
python scripts/prepare_data.py --dataset sharegpt

# 为离线训练生成隐藏状态（完整数据集需要约 12TB 磁盘空间）
torchrun --nproc_per_node=8 \
    scripts/prepare_hidden_states.py \
    --model-path <target-model-path> \
    --enable-aux-hidden-states \
    --data-path <jsonl-file-path> \
    --chat-template llama3 \
    --max-length 2048 \
    --tp-size 8 \
    --batch-size 4 \
    --mem-frac=0.75

# 使用目标模型重新生成数据集（推荐，以获得更好的对齐效果）
# regenerate_train_data.py: 仅替换最后一轮 assistant 回复，同步批量处理，适合快速改写
python scripts/regenerate_train_data.py \
    --model <target-model-path> \
    --input-file-path <jsonl-file-path> \
    --output-file-path <regenerated-jsonl-file-path> \
    --batch-size 128 \
    --tp-size 8

# generate_data_by_target.py: 重建所有轮次回复，异步高并发，支持推理模型（如 GPT-OSS）
python scripts/generate_data_by_target.py \
    --model-name <target-model-path> \
    --raw-data-file <jsonl-file-path> \
    --output-dir <output-dir> \
    --max-concurrency 512 \
    --server-address-port 127.0.0.1:30000 \
    --is-reasoning-model
```

### 训练

```bash
# 在线训练（Llama3 示例）
bash ./examples/run_llama3_eagle3_online.sh

# 离线训练（Llama3 示例）
bash ./examples/run_llama3_eagle3_offline.sh

# 自定义在线训练
torchrun --standalone --nproc_per_node 8 \
    scripts/train_eagle3_online.py \
    --target-model-path <target-model> \
    --draft-model-config configs/llama3-8B-eagle3.json \
    --train-data-path <train.jsonl> \
    --output-dir <output-dir> \
    --num-epochs 2 \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template llama3 \
    --target-model-backend sglang

# 自定义离线训练
torchrun --standalone --nproc_per_node 8 \
    scripts/train_eagle3_offline.py \
    --target-model-path <target-model> \
    --draft-model-config configs/llama3-8B-eagle3.json \
    --train-data-path <train.jsonl> \
    --train-hidden-states-path <hidden-states-dir> \
    --output-dir <output-dir> \
    --num-epochs 10 \
    --draft-global-batch-size 16 \
    --draft-micro-batch-size 1 \
    --learning-rate 5e-5
```

## 代码架构

### 模块结构

```
specforge/
├── core/                    # 核心 Eagle3 模型实现
│   ├── eagle3.py           # OnlineEagle3Model、OfflineEagle3Model 及 TTT 逻辑
│   └── loss.py             # 训练用的 LogSoftmaxLoss
├── modeling/               # 模型架构
│   ├── draft/             # 草稿模型实现
│   │   ├── base.py        # Eagle3DraftModel 抽象基类
│   │   ├── llama3_eagle.py # Llama3 特定的草稿模型
│   │   └── flex_attention.py # FlexAttention 后端支持
│   └── target/            # 目标模型后端
│       ├── eagle3_target_model.py # 抽象目标模型接口
│       ├── custom_backend/        # 基于 HuggingFace 的自定义后端
│       └── sglang_backend/        # 基于 SGLang 的后端（最快）
├── data/                   # 数据处理流水线
│   ├── preprocessing.py   # 数据集预处理和分词
│   ├── parse.py           # 对话解析逻辑
│   └── template.py        # 聊天模板处理器
├── distributed.py         # DP/TP/FSDP 设置工具
├── optimizer.py           # BF16Optimizer 包装器
├── lr_scheduler.py        # 学习率调度器
└── layers/                # 自定义层（如支持 TP 的 linear）
```

### 目标模型后端

SpecForge 支持多种目标模型后端：

1. **SGLang 后端** (`--target-model-backend sglang`)：推理最快，用于主流模型
2. **自定义后端** (`--target-model-backend custom`)：基于 HuggingFace，用于 SGLang 不支持的模型
3. **自动检测**：如果未指定，会根据模型类型自动选择

自定义后端定义在 `specforge/modeling/target/custom_backend/` 中（如 `llama.py`、`qwen2.py`、`phi3.py`）。

### Eagle3 训练流程

**在线模式**（`OnlineEagle3Model` 位于 `specforge/core/eagle3.py`）：
1. 目标模型实时从 3 个辅助层生成隐藏状态
2. 隐藏状态被拼接并通过 `draft_model.project_hidden_states()` 投影
3. TTT 展开：循环 `length` 次迭代（默认 7 次）：
   - 通过 `draft_model.embed_input_ids()` 嵌入输入 tokens
   - 与投影后的隐藏状态拼接
   - 前向传播通过草稿模型各层
   - 通过 `draft_model.compute_logits()` 计算 logits
   - 计算与偏移目标的损失
4. 聚合所有 TTT 迭代的损失

**离线模式**（`OfflineEagle3Model` 位于 `specforge/core/eagle3.py`）：
- 与在线模式类似，但隐藏状态从磁盘预加载而非实时生成

**关键文件：**
- `specforge/core/eagle3.py`：TTT 训练循环实现
- `specforge/modeling/draft/base.py`：抽象草稿模型接口
- `specforge/modeling/target/eagle3_target_model.py`：抽象目标模型接口

### 数据格式

**对话格式**（默认）：
```json
{
    "id": "xxxx",
    "conversations": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
```

**预格式化文本格式**（配合 `--is-preformatted` 使用）：
```json
{
    "id": "xxxx",
    "text": "<|im_start|>system\n...<|im_end|>\n..."
}
```

### 支持的模型

`configs/` 中预配置的草稿模型配置：
- Llama 3/3.1/3.3/4 系列
- Qwen 2.5/3 系列（稠密和 MoE）
- Phi 4/4-mini
- DeepSeek-V2-Lite
- GPT-OSS 变体
- QwQ-32B

## 开发注意事项

### 添加新的模型架构

1. **创建草稿模型配置**：在 `configs/` 中添加 JSON 配置，指定架构参数
2. **实现草稿模型**（如需要）：在 `specforge/modeling/draft/` 中继承 `Eagle3DraftModel`
3. **实现目标后端**（如需要）：在 `specforge/modeling/target/custom_backend/` 中继承 `Eagle3TargetModel`
4. **注册自动加载**：如果使用自定义架构，更新 `specforge/modeling/auto.py`

### 注意力后端

- `sdpa`（在线训练默认）：Scaled Dot-Product Attention
- `flex_attention`（离线训练默认）：FlexAttention，离线训练性能更好

使用 `--draft-attention-backend` 或 `--attention-backend` 参数指定。

### 实验追踪

通过 `--report-to` 参数支持 Wandb、TensorBoard、SwanLab：
```bash
--report-to wandb --wandb-project <project> --wandb-name <run-name>
```

### 分布式训练

- **数据并行（DP）**：通过 `specforge.distributed.get_dp_device_mesh()` 自动配置
- **张量并行（TP）**：目标模型可通过 `--tp-size` 参数使用 TP
- **FSDP**：草稿模型在离线训练中使用 FSDP（参见 `train_eagle3_offline.py`）

### 重要训练参数

- `--ttt-length`：TTT 展开迭代次数（默认：7）
- `--max-length`：最大序列长度（默认：2048）
- `--draft-global-batch-size`（离线）：所有 GPU 的总批次大小
- `--draft-micro-batch-size`（离线）：每个 GPU 的微批次大小，用于梯度累积
- `--batch-size`（在线）：每个 GPU 的批次大小
- `--chat-template`：处理对话的模板名称（llama3、qwen 等）
- `--is-preformatted`：用于预格式化文本数据的标志

# Llama3 Eagle3 在线训练复现指南
本文档提供了一个分步指南,用于复现 EAGLE3 论文中描述的训练过程,使用脚本 `examples/run_llama3_eagle3_sgl_online.sh`。我们将逐步讲解该脚本并解释每个关键步骤。

## 工作流程
### 步骤 1. 准备环境
如果您想使用 `python>=3.12`,请设置 `export SETUPTOOLS_USE_DISTUTILS=local`。
```shell
uv venv --python 3.11
source .venv/bin/activate
cd PATH-TO-SpecForge
uv pip install -r requirements.txt --prerelease=allow
uv pip install -v .
```
```shell
# 如果用conda
conda create -n SpecForge python=3.11 -y
conda activate SpecForge
conda install pip uv -y
conda install nvidia::cuda-toolkit
uv pip install -r requirements.txt --prerelease=allow
uv pip install -v .

# 如果flashinfer报错，
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

完成这些步骤后,您可以通过运行以下命令来检查安装是否成功。如果安装成功,您应该不会看到任何错误。

```shell
python -c "import specforge"
```
注意sglang版本需要==0.5.4。

### 步骤 2. 准备模型和数据集
接下来,我们可以开始准备模型和数据集。首先,使用以下命令下载模型和数据集。
```shell
hf download meta-llama/Llama-3.1-8B-Instruct
hf download Aeala/ShareGPT_Vicuna_unfiltered --repo-type dataset
hf download HuggingFaceH4/ultrachat_200k --repo-type dataset

python scripts/prepare_data.py --dataset ultrachat --output_path ./cache/dataset
python scripts/prepare_data.py --dataset sharegpt --output_path ./cache/dataset
cat ./cache/dataset/sharegpt.jsonl ./cache/dataset/ultrachat.jsonl > ./cache/dataset/train_dataset.jsonl
```

然后,启动 SGLang 服务器并运行 `generate_data_by_target.py` 以在不同数据集上从基础模型生成响应。请确保更新 `generate_data_by_target.py` 中的 `SYSTEM_PROMPT` 值以满足您的需求。

```shell
# 开启两个卡
for i in {0..1}; do
    CUDA_VISIBLE_DEVICES=$i python3 -m sglang.launch_server \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --cuda-graph-bs 1 2 4 8 16 32 64 128 256 512 \
        --dtype bfloat16 --mem-frac=0.7 --port $((30000 + i)) &
done

(
如果启动报错，可以尝试：
export CUDA_HOME=/usr/local/cuda-12.6 # or your version, sometimes is /usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
)

python scripts/generate_data_by_target.py \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --raw-data-file ./cache/dataset/train_dataset.jsonl \
    --output-dir ./cache/generated-dataset/llama-3.1-8b-instruct \
    --max-concurrency 512 \
    --num-per-shard 50000 \
    --server-address-port 127.0.0.1:30000 127.0.0.1:30001

#
python scripts/generate_data_by_target.py \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --raw-data-file /YOUR/PATH/Llama-3.1-8B-Instruct/dataset/ultrachat.jsonl \
    --output-dir /YOUR/PATH/Llama-3.1-8B-Instruct/generated-dataset/ultrachat-llama-3.1-8b-instruct \
    --max-concurrency 512 \
    --num-per-shard 50000 \
    --server-address-port 127.0.0.1:30001 127.0.0.1:30002 127.0.0.1:30003 127.0.0.1:30004
```

这样生成的是每50000条数据一个文件的多个分片。接下来,我们需要将这些分片合并为训练集和测试集。
```bash
(SpecForge) wshiah@zxcpu3:~/code/weijie/SpecForge$ ll cache/generated-dataset/llama-3.1-8b-instruct/ -h
total 2.6G
drwxrwxr-x 2 wshiah wshiah   16 Nov 10 20:42 ./
drwxrwxr-x 3 wshiah wshiah    3 Nov 10 04:11 ../
-rw-rw-r-- 1 wshiah wshiah 242M Nov 10 08:11 error_0-50000.jsonl
-rw-rw-r
-- 1 wshiah wshiah 251M Nov 10 12:25 error_100000-150000.jsonl
-rw-rw-r-- 1 wshiah wshiah 229M Nov 10 15:13 error_150000-200000.jsonl
-rw-rw-r-- 1 wshiah wshiah 241M Nov 10 17:23 error_200000-250000.jsonl
-rw-rw-r-- 1 wshiah wshiah 247M Nov 10 19:20 error_250000-300000.jsonl
-rw-rw-r-- 1 wshiah wshiah 137M Nov 10 20:42 error_300000-328540.jsonl
-rw-rw-r-- 1 wshiah wshiah 258M Nov 10 10:18 error_50000-100000.jsonl
-rw-rw-r-- 1 wshiah wshiah 126M Nov 10 08:11 shard_0-50000.jsonl
-rw-rw-r-- 1 wshiah wshiah 145M Nov 10 12:25 shard_100000-150000.jsonl
-rw-rw-r-- 1 wshiah wshiah 182M Nov 10 15:13 shard_150000-200000.jsonl
-rw-rw-r-- 1 wshiah wshiah 165M Nov 10 17:23 shard_200000-250000.jsonl
-rw-rw-r-- 1 wshiah wshiah 155M Nov 10 19:20 shard_250000-300000.jsonl
-rw-rw-r-- 1 wshiah wshiah  95M Nov 10 20:42 shard_300000-328540.jsonl
-rw-rw-r-- 1 wshiah wshiah 128M Nov 10 10:18 shard_50000-100000.jsonl

```
可以用一个简单的python脚本来完成这个任务，
```bash
# 合并并随机选择10000条作为测试集，其余全部作为训练集
python scripts/merge_shards.py --input-dir ./cache/generated-dataset/llama-3.1-8b-instruct --eval-size 10000

# 指定输出目录
python scripts/merge_shards.py \
    --input-dir ./cache/generated-dataset/llama-3.1-8b-instruct \
    --output-dir ./cache/train_dataset/llama-3.1-8b-instruct \
    --train-size 50000 \
    --eval-size 5000
```


完成这些步骤后,您可以查看 `error.jsonl` 中的错误条目。其中大多数可能是 `request timeout`。然后您可以决定是否要重新生成这些样本。

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

# # 这里draft config词表大小调整一下，跟目标模型一致
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

```

这样会得到
```bash
(SpecForge) wshiah@zxcpu3:~/code/weijie/SpecForge$ ll ./cache/preprocess_data_cache_dir/llama-3.1-8b-instruct/
total 83
drwxrwxr-x 4 wshiah wshiah   4 Nov 10 22:09 ./
drwxrwxr-x 3 wshiah wshiah   3 Nov 10 22:06 ../
drwxrwxr-x 2 wshiah wshiah 130 Nov 10 22:09 processed_dataset/
drwxrwxr-x 2 wshiah wshiah   3 Nov 10 22:12 vocab_mapping/
```

### 步骤 3. 开始训练

使用以下脚本进行训练。

- 设置 `total-steps=800000, learning-rate=5e-5` 以与 [EAGLE 官方仓库配置](https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/ds_config.json)保持一致。可以随意更改这些设置来进行您自己的实验。`total-steps` 和 `warmup-ratio` 决定学习率的增长曲线。

```shell
# 设置环境变量
export NUM_GPUS=4
export MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
export OUTPUT_DIR=./model/Llama-3.1-8B-Instruct/dev_outputs/
export DATASET_PATH=./cache/train_dataset/llama-3.1-8b-instruct
export CACHE_DIR=./cache/preprocess_data_cache_dir/llama-3.1-8b-instruct
export MAX_LENGTH=4096
export CHAT_TEMPLATE=llama3

# 开始训练 (使用 train_eagle3_online.py 脚本，支持 SGLang 后端)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    scripts/train_eagle3_online.py \
    --target-model-path $MODEL_PATH \
    --draft-model-config ./configs/llama3-8B-eagle3.json \
    --train-data-path $DATASET_PATH/train.jsonl \
    --eval-data-path $DATASET_PATH/eval.jsonl \
    --tp-size $NUM_GPUS \
    --output-dir $OUTPUT_DIR \
    --num-epochs 1 \
    --batch-size 1 \
    --learning-rate 5e-5 \
    --attention-backend flex_attention \
    --max-length $MAX_LENGTH \
    --chat-template $CHAT_TEMPLATE \
    --cache-dir $CACHE_DIR \
    --dist-timeout 10 \
    --wandb-project llama3-8b \
    --wandb-name eagle3-online-sglang \
    --report-to wandb \
    --target-model-backend sglang
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
    --speculative-draft-model-path model/Llama-3.1-8B-Instruct/dev_outputs/epoch_0_step_10000 \
    --port 20001 \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --tp-size 4 \
    --config-list "${config_list[@]}" \
    --benchmark-list mtbench:80 gsm8k:200 humaneval:200 math500:200 \
    --output output_Llama-3.1-8B-Instruct.jsonl


# 定义推理配置列表
config_list=(
    "1,0,0,0"      # baseline: 无投机解码
    "4,3,1,4"      # 配置1: batch=4, steps=3, topk=1, draft_tokens=4
    "4,5,1,6"      # 配置2: batch=4, steps=5, topk=1, draft_tokens=6
    "4,7,10,60"    # 配置3: batch=4, steps=7, topk=10, draft_tokens=60
)
# 运行 benchmark（一次性测试所有 6 个数据集）
CUDA_VISIBLE_DEVICES=2 uv run benchmarks/bench_model_speedup.py \
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
      --output results_all_temps.jsonl
```
