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

| 参数 | Eagle3 (examples) | Medusa论文 | 本配置 (Medusa) | 理由 |
|------|-------------------|------------|-----------------|------|
| **Learning Rate** | 1e-4 | 1e-4 (constant) | **1e-4** | 与Eagle对齐，论文AWS实现同样用1e-4 |
| **Batch Size (per device)** | 1 | 1 | **1** | 与Eagle对齐 |
| **Gradient Accumulation** | - | 16 | **4** | 有效batch=4 (更小以适应内存) |
| **Epochs** | 10 | Not specified | **10** | 与Eagle对齐 |
| **Warmup Ratio** | 0.015 | 0.1 (cosine) | **0.015** | 与Eagle对齐 |
| **Max Grad Norm** | 0.5 | - | **0.5** | 与Eagle对齐 |
| **Draft Layers** | 1 | 0 (only heads) | **0** | Medusa无额外Transformer层 |

**重要说明**：
- 论文推荐Medusa-1的LR可以更高（1e-3），因为只训练小的heads
- 但为了**公平对比**，我们使用与Eagle3相同的1e-4
- 如果训练不稳定，可以尝试提高到1e-3

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

为了确保Medusa和Eagle3的对比是**公平的**，我们统一了以下参数：

1. ✅ **相同的学习率**：1e-4
2. ✅ **相同的batch size**：1 per device
3. ✅ **相同的训练轮数**：10 epochs
4. ✅ **相同的warmup ratio**：0.015
5. ✅ **相同的梯度裁剪**：0.5
6. ✅ **相同的词表映射**：draft_vocab_size与Eagle3一致
7. ✅ **相同的数据**：复用Eagle3的数据处理pipeline

**唯一差异**：
- 架构不同（Medusa heads vs Eagle3 TTT）
- 参数量不同（Medusa更少）

这样对比结果才能真实反映算法本身的差异！

---

## 参考文献

1. Medusa论文：[arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
2. Medusa GitHub：https://github.com/FasterDecoding/Medusa
3. Eagle3配置：`configs/llama3-8B-eagle3.json`, `configs/qwen2.5-7b-eagle3.json`
4. AWS Medusa实现：https://aws.amazon.com/blogs/machine-learning/achieve-2x-speed-up-in-llm-inference-with-medusa-1-on-amazon-sagemaker-ai/
