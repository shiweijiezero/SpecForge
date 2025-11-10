## 🚀 训练

### 🏎️ 在线训练

我们提供了简单的启动脚本来训练 Llama 3 和 4、Qwen3 模型的 Eagle3 模型。你可以运行以下命令开始训练。

```bash
# 确保你已经准备好 sharegpt 数据
# 训练 llama3-8B-instruct
bash ./examples/run_llama3_eagle3_online.sh

# 训练 llama4-scout
bash ./examples/run_llama4_eagle3_online.sh

# 训练 Qwen3-30B-A3B
# 也支持 Qwen3-235B-A22B 在线训练；
bash ./examples/run_qwen3_moe_eagle3_online.sh

# 训练 Qwen3-8B
bash ./examples/run_qwen3_dense_eagle3_online.sh

# 训练 Qwq-32B
bash ./examples/run_qwq_eagle3_online.sh
```

### 💨 离线训练

我们提供了一个简单的启动脚本，以离线方式为 Llama-3.1-8B-Instruct 模型训练 Eagle3 模型。你可以运行以下命令开始训练。几乎所有内容都与在线训练步骤相同，除了你不需要配置任何关于目标模型的内容。相反，你需要将 `--train-hidden-states-path` 传递给文件。

```bash
# 确保你已经准备好 sharegpt 数据
bash ./examples/run_llama3_eagle3_offline.sh
```

### 📈 实验追踪

本项目支持将训练进度记录到 Wandb、TensorBoard 和 SwanLab。你可以通过在 shell 脚本的命令行中添加 --report-to 参数来启用追踪。
