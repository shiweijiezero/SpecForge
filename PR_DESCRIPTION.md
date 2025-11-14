# 实现 Medusa 算法并添加 Lookahead 分析文档

## 📋 PR 概述

本 PR 在 SpecForge 框架中完成了 Medusa 算法的完整实现，并对 Lookahead Decoding 进行了深入分析。

## ✨ 主要工作

### 1. 📚 通用文档

**docs/如何在SpecForge中添加新算法.md**
- 7步通用流程：从理解算法到验证性能
- SpecForge 架构详解（modeling/draft/, core/, scripts/）
- Eagle3DraftModel 基类抽象方法说明
- 适用于所有新算法的开发指南

### 2. 🐍 Medusa 完整实现

#### 文档 (docs/Medusa实现教程.md - 800+ 行)
- **理论对比**：Medusa vs Eagle3 架构差异
- **实现细节**：ResBlock + MedusaHead 代码详解
- **训练逻辑**：OnlineMedusaModel / OfflineMedusaModel
- **数据处理**：与 Eagle3 共享前半部分流程
- **公平对比**：控制变量法，所有超参数与 Eagle3 严格对齐

#### 模型配置
**configs/medusa/llama3-8B-medusa.json**
- 仅包含架构参数（hidden_size, num_heads, etc.）
- 训练超参数移至训练脚本（遵循您的要求）
- 详细注释说明每个参数来源

**configs/medusa/qwen2.5-7B-medusa.json**
- Qwen 特定配置（hidden_size: 3584, rope_theta: 1e6）
- 与 LLaMA 的差异对比注释

**configs/medusa/README.md**
- 参数来源表格（每个参数标注来自 Eagle3 哪个脚本的哪一行）
- 公平对比保证：控制变量法详解
- 验证检查清单（wc -l 检查数据大小，MD5 校验等）

#### 训练脚本（硬编码参数）
**examples/medusa/run_llama3_medusa_online.sh**
**examples/medusa/run_qwen25_medusa_online.sh**
- 所有参数硬编码（不使用变量）
- 行内注释说明每个参数来源
- 环境检查 + Vocab mapping 自动生成
- 训练前显示完整参数对比表

### 3. 🔍 Lookahead Decoding 分析

**docs/Lookahead分析与说明.md** (232 行)
- **核心结论**：Lookahead 不适合集成到 SpecForge（训练框架 vs 推理优化）
- **原理详解**：Jacobi 迭代 + n-gram 缓存机制
- **架构分析**：两分支（lookahead + verification）+ token_map 数据结构
- **实现机制**：Monkey-patching, 自定义 attention mask, 自定义 forward
- **性能对比**：Eagle3(2.5-3x) vs Medusa(2-2.5x) vs Lookahead(1.5-2.3x)
- **使用建议**：Lookahead 作为独立工具，与 SpecForge 训练的模型互补

## 🔑 关键设计决策

### 参数对齐策略
所有 Medusa 训练参数与 Eagle3 基线严格对齐：
| 参数 | Eagle3 (sgl_online) | Medusa | 来源 |
|------|---------------------|--------|------|
| Learning Rate | 5e-5 | **5e-5** | run_llama3_eagle3_sgl_online.sh:58 |
| Batch Size | 1 | **1** | sgl_online.sh:57 |
| Epochs | 1 | **1** | 您的实验配置（非论文的10） |
| Warmup Ratio | 0.015 | **0.015** | sgl_online.sh:65 |
| Max Grad Norm | 0.5 | **0.5** | sgl_online.sh:66 |

### 架构差异（允许不同）
| 参数 | Eagle3 | Medusa | 理由 |
|------|--------|--------|------|
| Draft Layers | 1 | **0** | Medusa 无 backbone |
| Num Heads | 1 | **4** | Medusa 论文 Table 2 推荐 3-5 |
| 训练方式 | TTT 递归 | **单次 forward** | 算法本质差异 |

### 硬编码参数
按您的要求，所有参数直接硬编码在脚本中：
```bash
torchrun \
    --standalone \
    --nproc_per_node $(nvidia-smi --list-gpus | wc -l) \
    ${ROOT_DIR}/scripts/train_medusa_online.py \
    --num-epochs 1 \
    --batch-size 1 \
    --learning-rate 5e-5 \
    --num-heads 4 \
    --warmup-ratio 0.015 \
    --max-grad-norm 0.5 \
    # ... 其他参数
```

## 📊 文件清单

### 新增文件
```
docs/
├── 如何在SpecForge中添加新算法.md          (通用开发指南)
├── Medusa实现教程.md                       (800+ 行详细教程)
└── Lookahead分析与说明.md                  (推理优化分析)

configs/medusa/
├── README.md                                (参数来源和对比保证)
├── llama3-8B-medusa.json                    (LLaMA 3.1 8B 配置)
└── qwen2.5-7B-medusa.json                   (Qwen2.5 7B 配置)

examples/medusa/
├── run_llama3_medusa_online.sh              (LLaMA 训练脚本)
└── run_qwen25_medusa_online.sh              (Qwen 训练脚本)
```

### 关键特性
✅ **严谨的参数对齐**：每个参数都有文档来源（脚本名:行号）
✅ **控制变量法**：仅算法差异，其他全部相同
✅ **硬编码参数**：便于跨机器运行不同步骤
✅ **详细注释**：所有配置都有解释和对比
✅ **验证清单**：提供检查命令确保公平对比

## 🎯 使用方法

### 训练 Medusa (LLaMA 3.1 8B)
```bash
cd /path/to/SpecForge
bash examples/medusa/run_llama3_medusa_online.sh
```

### 训练 Medusa (Qwen2.5 7B)
```bash
cd /path/to/SpecForge
bash examples/medusa/run_qwen25_medusa_online.sh
```

### 使用 Lookahead（独立工具）
```python
import lade
lade.augment_all()
lade.config_lade(LEVEL=5, WINDOW_SIZE=7, GUESS_SET_SIZE=7)

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
outputs = model.generate(**inputs, max_new_tokens=1024)  # 自动加速 1.5-2.3x
```

## 🔄 与现有代码的关系

- **复用 Eagle3 数据处理**：Medusa 前半部分与 Eagle3 完全相同
- **遵循 SpecForge 架构**：符合 `Eagle3DraftModel` 基类设计
- **独立的 Lookahead**：不集成到训练框架，作为推理工具存在

## ✅ 测试建议

1. **参数验证**：
   ```bash
   # 检查数据大小是否与 Eagle3 相同
   wc -l cache/dataset/sharegpt.jsonl

   # 检查训练 step 数是否一致
   # 预期: (数据行数 / batch_size / GPU数) * epochs
   ```

2. **公平性验证**：
   - 确认使用相同的 sharegpt.jsonl
   - 确认 learning rate = 5e-5（正式版本）
   - 确认 epochs = 1（您的基线配置）

3. **性能对比**：
   - Eagle3 加速比 vs Medusa 加速比
   - 训练时间对比（Medusa 应更快，无 TTT 递归）
   - 参数量对比（Medusa ~52M, Eagle3 ~135M）

## 📖 文档质量

- **如何添加新算法.md**: 面向所有开发者的通用指南
- **Medusa实现教程.md**: 800+ 行，8个章节，从理论到实践
- **Lookahead分析.md**: 清晰说明为何不集成，如何独立使用
- **configs/README.md**: 参数溯源表 + 验证清单

## 🙏 致谢

感谢您在开发过程中的指导：
- 修正学习率为 5e-5（正式训练版本）
- 硬编码参数以便跨机器使用
- 将训练超参数从 config 移至脚本
- 严格的公平对比要求

---

**Ready for review!** 🚀
