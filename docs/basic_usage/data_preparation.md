## 📝 数据准备

在本节中，我们将介绍如何为在线和离线训练准备数据集。如 [概览](#-overview) 部分所述，在线训练只需要原始数据集，而离线训练需要从目标模型生成的隐藏状态。在下面的部分中，我们将介绍如何准备原始数据集和隐藏状态。

### 🔄 重新生成训练数据集

许多公开数据集并非由你的目标模型生成，这可能导致草稿模型的输出与目标模型的行为不一致——降低接受率和推理效率。为解决这个问题，我们**推荐使用目标模型重新生成数据集**，这能更好地将草稿模型与目标模型的输出分布对齐，提高接受长度和整体性能。

运行以下命令重新生成你的数据集：

```bash
python3 \
    scripts/regenerate_data.py \
    --model <target-model-path> \
    --input-file-path <jsonl-file-path> \
    --output-file-path <regenerated-jsonl-file-path> \
    --batch-size 128 \
    --tp-size 8 \
    --num-samples 1000 \
    --port 30000 \
    --temperature 0 \
    --mem-fraction-static 0.85 \
    --auto-launch-server
```

### ☁️ 准备在线训练数据集

我们提供了一个脚本来准备一些示例数据集，包括 ultrachat (200k) 和 sharegpt (120k)，用于演示目的。你可以通过运行以下命令轻松处理数据集。jsonl 文件默认将被放置在项目路径的 `cache/dataset/<dataset_name>` 目录中。这些数据集将被处理成 `jsonl` 文件，这就是在线训练所需的原始数据集！

```bash
# ultrachat
python scripts/prepare_data.py --dataset ultrachat

# sharegpt
python scripts/prepare_data.py --dataset sharegpt
```

### 💾 准备离线训练数据集

与在线数据相比，离线数据需要额外的隐藏状态生成步骤。因此，在深入本节之前，请确保你已按照 [准备在线训练数据集](#-prepare-online-training-dataset) 部分的说明准备好 `jsonl` 文件。一旦你有了 `jsonl` 文件，就可以开始生成隐藏状态。

你可以运行以下命令来获取隐藏状态。

```bash
torchrun --nproc_per_node=8 \
    scripts/prepare_hidden_states.py \
    --model-path <target-model-path> \
    --enable-aux-hidden-states \
    --data-path <jsonl-file-path> \
    --chat-template llama3 \
    --max-length 2048 \
    --tp-size 8 \
    --batch-size 4 \
    --mem-frac=0.75 \
    --num-samples 1000
```
> ⚠️ 此提取过程可能需要 2 小时和约 5TB 磁盘空间

你需要指定以下参数：
- `--model-path`：这是 huggingface 仓库名称或目标模型的路径。
- `--data-path`：这是前一个 `prepare_data.py` 脚本的实际输出路径。
- `--chat-template`：这是用于此模型的聊天模板。
- `--num-samples`：这指定用于隐藏状态生成的数据样本数量。默认情况下，它将使用 `data-path` 中的所有数据。


### 🤩 准备你自己的数据集

除了提供的 ShareGPT/Ultrachat 数据集外，你还可以准备自己的数据集。我们支持两种格式：

#### 选项 1：对话格式

你应该以 jsonl 格式准备数据集，模式应如下所示：

```json
{
    "id": "xxxx",
    "conversations": [
        {
            "role": "user | assistant",
            "content": "The message content"
        }
    ],
}
```

#### 选项 2：预格式化文本格式

如果你已经有使用特定聊天模板格式化的对话，可以直接使用预格式化文本：

```json
{
    "id": "xxxx",
    "text": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there!<|im_end|>\n"
}
```

当你有在目标模型训练期间使用的预格式化提示，并且有来自目标模型的原始生成时，此格式很有用。

要使用预格式化数据集，在训练命令中添加 `--is-preformatted` 标志。请注意，仍然需要 `--chat-template` 参数，并且应该与预格式化文本中使用的模板匹配，因为它用于识别用户/助手 tokens 以确定助手范围并生成相应的损失掩码。

```bash
torchrun --standalone --nproc_per_node 8 \
    scripts/train_eagle3_online.py \
    --is-preformatted \
    --chat-template qwen \
    --train-data-path ./your_preformatted_dataset.jsonl \
    # ... 其他参数
```

一旦你准备好 `jsonl` 文件，就可以直接进行在线训练或为离线训练生成隐藏状态。

如果你有多个数据集，可以将它们合并到一个 jsonl 文件中。例如，你可以这样做：

```bash
cat dataset1.jsonl dataset2.jsonl > merged_dataset.jsonl
```
