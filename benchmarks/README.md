# 投机解码基准测试

## 设置

你可以使用以下命令创建一个新环境并安装 SGLang：

```bash
# 创建虚拟环境
uv venv sglang -p 3.11
source sglang/bin/activate

# 安装 sglang
uv pip install "sglang[all]>=0.4.9.post2"
```

你可以使用以下命令通过 SGLang 服务你训练的模型，将 `<target-model-path>` 和 `<draft-model-path>` 替换为目标模型和草稿模型的实际路径。

```bash
python3 -m sglang.launch_server \
    --model <target-model-path>  \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path <draft-model-path> \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.75 \
    --cuda-graph-max-bs 2 \
    --tp 1 \
    --context-length 8192 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16
```

## 运行基准测试

你首先需要启动 SGLang 服务器：

```bash
python3 -m sglang.launch_server \
    --model <target-model-path>  \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path <draft-model-path> \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.75 \
    --cuda-graph-max-bs 2 \
    --tp 8 \
    --context-length 8192 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16
```

然后你可以运行基准测试：

```bash
# GSM8K
python run_gsm8k.py

# MATH-500
python run_math500.py

# MTBench
python run_mtbench.py

# HumanEval
python run_humaneval.py
```
