# Medusa配置文件说明

## 参数来源和设置依据

### 1. Head数量 (medusa_num_heads)

**论文依据**：Medusa论文 (arXiv:2401.10774)

- **论文推荐**：5个heads（Appendix B）
- **实践中常用**：3-4个heads
- **我们的选择**：**4个heads**

**原因**：
1. 论文Table 2显示：3头 vs 4头 vs 5头的加速比差异不大（~0.1x）
2. 4头在性能和计算间取得平衡
3. 与参考目录中Medusa-main的默认配置一致

**对比Eagle3**：Eagle3使用1个head + TTT递归

---

### 2. 训练参数对比

**核心原则**：为了**公平对比**，Medusa应使用与Eagle3**完全相同**的训练设置。

| 参数 | Eagle3 (sgl_online) | 本配置 (Medusa) | 说明 |
|------|---------------------|-----------------|------|
| **Learning Rate** | 5e-5 | **5e-5** | 来自run_llama3_eagle3_sgl_online.sh:58 |
| **Batch Size (per device)** | 1 | **1** | 来自sgl_online.sh:57 |
| **Warmup Ratio** | 0.015 | **0.015** | 来自sgl_online.sh:65 |
| **Max Grad Norm** | 0.5 | **0.5** | 标准值 |
| **Draft Layers** | 1 | **0** | Medusa架构差异（无backbone） |
| **Max Length** | 2048 | **2048** | 序列长度 |

**数据和Epochs设置 - 重要！**

Eagle3脚本使用了大规模数据：
- 数据：ShareGPT + Ultrachat 200k（>20万样本）
- Total Steps：800,000步
- Num Epochs：10

**但实际实验中应该使用相同的数据和epochs！**

如果您的Eagle3实验使用：
- 数据量：50k样本
- Epochs：1

那么Medusa也应该使用：
- **相同的50k样本**
- **相同的1 epoch**

**公平对比清单**：
- ✅ 相同的数据集（同一个.jsonl文件）
- ✅ 相同的数据量（50k或200k等）
- ✅ 相同的训练轮数（1 epoch或10 epochs）
- ✅ 相同的学习率（5e-5）
- ✅ 相同的batch size（1）
- ✅ 相同的warmup ratio（0.015）
- ✅ 相同的max length（2048）

**重要说明**：
- Eagle3论文脚本使用10 epochs + 200k+数据，但**您的实验设置可能不同**
- 配置文件中的epochs等参数仅为示例，**实际使用时应根据您的Eagle3基线调整**
- Medusa论文建议LR=1e-3（只训练heads），但为公平对比使用5e-5

---

### 3. 词表设置 (draft_vocab_size)

| 模型 | Target Vocab | Draft Vocab | 映射文件 |
|------|--------------|-------------|----------|
| **LLaMA-3.1-8B** | 128256 | 32000 | 需要t2d/d2t |
| **Qwen2.5-7B** | 152064 | 16000 | 需要t2d/d2t |

**依据**：与Eagle3保持一致

- LLaMA: `configs/llama3-8B-eagle3.json`中`draft_vocab_size=32000`
- Qwen: `configs/qwen2.5-7b-eagle3.json`中`draft_vocab_size=16000`

---

### 4. 隐藏层数 (num_hidden_layers)

**Medusa vs Eagle3**：

- **Eagle3**: `num_hidden_layers=1` （有draft backbone）
- **Medusa**: `num_hidden_layers=0` （无draft backbone，只有heads）

**架构对比**：
```
Eagle3: Input → Draft Transformer (1 layer) → LM Head
Medusa: Input → ResBlock → LM Head (直接在base model输出上)
```

---

### 5. Head层数 (medusa_num_layers)

**论文依据**：每个Medusa head包含1个ResBlock + 1个Linear层

- `medusa_num_layers=1`: 标准配置（论文默认）
- `medusa_num_layers=2`: 可以尝试，但参数量增加

**我们的选择**：`medusa_num_layers=1`（与论文一致）

---

## 配置文件列表

- `llama3-8B-medusa.json`: LLaMA 3.1 8B配置
- `qwen2.5-7B-medusa.json`: Qwen 2.5 7B配置

---

## 公平性对比保证

### 关键原则：控制变量法

为了确保Medusa和Eagle3的对比结果真实反映**算法本身的差异**，必须严格控制所有其他变量：

### 必须相同的参数（控制变量）

| 类别 | 参数 | 值 | 验证方法 |
|------|------|-----|---------|
| **数据** | 训练数据 | 完全相同的.jsonl文件 | 检查文件MD5 |
| **数据** | 数据量 | 相同样本数（如50k） | `wc -l` |
| **训练** | Epochs | 相同轮数（如1） | 检查训练日志 |
| **训练** | Learning Rate | 5e-5 | 脚本参数 |
| **训练** | Batch Size | 1 per device | 脚本参数 |
| **训练** | Warmup Ratio | 0.015 | 脚本参数 |
| **训练** | Max Grad Norm | 0.5 | 脚本参数 |
| **训练** | Max Length | 2048 | 脚本参数 |
| **训练** | Chat Template | 相同（llama3/qwen） | 脚本参数 |
| **词表** | Draft Vocab Size | 相同映射策略 | 配置文件 |

### 允许不同的参数（算法差异）

| 参数 | Eagle3 | Medusa | 原因 |
|------|--------|---------|------|
| Draft Layers | 1 | 0 | 架构差异 |
| Num Heads | 1 | 4 | 算法设计 |
| 训练方法 | TTT递归 | 单次forward | 算法核心 |
| 参数量 | ~135M | ~52M | 架构导致 |

### 验证Checklist

在进行对比实验前，请确认：

- [ ] Medusa和Eagle3使用**完全相同**的训练数据文件
- [ ] 数据量相同（通过`wc -l`验证）
- [ ] Epochs相同（如您使用1 epoch，两者都用1）
- [ ] 学习率相同（5e-5）
- [ ] Batch size相同（1）
- [ ] Max length相同（2048）
- [ ] Warmup ratio相同（0.015）
- [ ] 训练完成后，两个模型的总训练steps接近

**警告**：如果以上任何参数不同，对比结果将不可靠！

---

## 参考文献

1. Medusa论文：[arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
2. Medusa GitHub：https://github.com/FasterDecoding/Medusa
3. Eagle3配置：`configs/llama3-8B-eagle3.json`, `configs/qwen2.5-7b-eagle3.json`
4. AWS Medusa实现：https://aws.amazon.com/blogs/machine-learning/achieve-2x-speed-up-in-llm-inference-with-medusa-1-on-amazon-sagemaker-ai/
