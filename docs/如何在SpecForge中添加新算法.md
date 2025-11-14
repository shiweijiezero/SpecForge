# 如何在SpecForge中添加新的推测解码算法

## 概述

本文档详细说明如何在SpecForge框架中添加新的推测解码算法。我们以**Eagle3**为参考实现，帮助您理解SpecForge的设计模式，并指导您将新算法（如Medusa）集成到框架中。

**适用场景**：
- 添加全新的推测解码算法（如Medusa、Lookahead、EAGLE变体等）
- 为现有算法添加新的backbone支持（如支持Qwen、Mistral等）
- 扩展训练模式（如添加新的在线/离线训练策略）

**前置要求**：
- 熟悉PyTorch和Transformers库
- 理解推测解码的基本原理
- 了解您要实现的算法的论文和原始实现

## SpecForge架构全局视图

SpecForge采用模块化设计，分为以下核心组件：

```
SpecForge/
├── specforge/
│   ├── modeling/              # 模型定义层
│   │   ├── draft/            # Draft模型实现（算法特定）
│   │   │   ├── base.py       # 抽象基类
│   │   │   └── llama3_eagle.py  # Eagle3 for Llama示例
│   │   ├── target/           # Target模型包装器
│   │   │   ├── eagle3_target_model.py
│   │   │   ├── custom_backend/   # 自定义backend
│   │   │   └── sglang_backend/   # SGLang backend
│   │   ├── auto.py           # 自动加载类
│   │   └── utils.py
│   ├── core/                 # 训练逻辑层（算法特定）
│   │   ├── eagle3.py         # Eagle3训练逻辑
│   │   └── loss.py           # 损失函数
│   ├── data/                 # 数据处理（通用）
│   ├── distributed.py        # 分布式支持（通用）
│   ├── optimizer.py          # 优化器（通用）
│   └── lr_scheduler.py       # 学习率调度（通用）
├── scripts/                  # 训练脚本（算法特定）
│   ├── train_eagle3_online.py
│   ├── train_eagle3_offline.py
│   └── build_eagle3_dataset_cache.py
├── configs/                  # 配置文件（模型特定）
│   ├── llama3-8B-eagle3.json
│   └── ...
└── examples/                 # 使用示例
    └── README.md
```

**组件职责划分**：
- **Draft Model**: 定义模型架构（特定于算法+backbone组合）
- **Core Module**: 实现训练循环和推理逻辑（特定于算法）
- **Training Script**: 组装组件并执行训练（特定于算法）
- **Data/Distributed/Optimizer**: 通用基础设施（跨算法复用）

## 第一步：理解算法特性

在开始实现之前，您需要明确算法的以下特性：

### 1.1 算法类型分类

| 特性 | Eagle3 | Medusa | 您的算法 |
|------|--------|---------|----------|
| **预测方式** | 自回归（递归） | 并行预测 | ? |
| **是否需要Target Hidden States** | 是（TTT） | 否 | ? |
| **Head数量** | 1 | 多个（3-4） | ? |
| **训练目标** | 模仿target分布 | 预测future tokens | ? |
| **是否训练Base Model** | 否（Frozen） | Medusa-1:否<br>Medusa-2:是 | ? |

### 1.2 架构需求检查清单

请回答以下问题：

- [ ] **输入依赖**：Draft model需要哪些输入？
  - input_ids
  - target model的hidden states（哪几层？）
  - attention mask
  - position ids
  - past_key_values

- [ ] **模型结构**：Draft model的架构是什么？
  - Transformer layers数量
  - Hidden size
  - 特殊模块（如ResBlock、MLP head等）

- [ ] **训练方式**：如何训练？
  - Online（实时从target提取）还是Offline（预计算缓存）？
  - 损失函数是什么？
  - 是否需要特殊的mask或attention pattern？

- [ ] **词表映射**：是否需要词表映射？
  - Draft vocab size vs Target vocab size
  - 是否需要`t2d`和`d2t` mapping？

### 1.3 Eagle3算法特性（参考示例）

以Eagle3为例：
- **预测方式**：自回归递归，使用Test-Time Training (TTT)
- **输入**：input_ids + target hidden states（3层：layer 1, mid layer, layer -4）
- **架构**：
  - Project层：将3层hidden states concat后投影到单一hidden size
  - Backbone：轻量级Transformer（通常1层）
  - LM Head：从target model复制
- **训练**：
  - TTT循环：`length=7`轮递归训练
  - 每轮：predict → compute loss → 更新input
  - 损失：LogSoftmax loss on target distribution
- **词表**：Draft vocab (32K) → Target vocab (128K)通过t2d/d2t映射

---

## 第二步：实现Draft Model

### 2.1 继承`Eagle3DraftModel`基类

Draft model必须继承`specforge.modeling.draft.base.Eagle3DraftModel`并实现抽象方法。

**关键抽象方法**：

```python
from specforge.modeling.draft.base import Eagle3DraftModel

class YourDraftModel(Eagle3DraftModel):
    """
    您的Draft Model实现
    """

    @abstractmethod
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        将input_ids转换为embeddings

        Args:
            input_ids: (batch, seq_len)
        Returns:
            embeddings: (batch, seq_len, hidden_size)
        """
        pass

    @abstractmethod
    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        投影hidden states到目标维度

        对于Eagle3: 将3层hidden states concat (batch, seq, 3*hidden)
                    投影到 (batch, seq, hidden)
        对于Medusa: 可能不需要此步骤（直接使用base model输出）

        Args:
            hidden_states: 从target model提取的hidden states
        Returns:
            projected_hidden_states: 投影后的hidden states
        """
        pass

    @abstractmethod
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        从hidden states计算logits

        Args:
            hidden_states: (batch, seq_len, hidden_size)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        pass

    @abstractmethod
    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Draft model的主干网络

        Args:
            input_embeds: Token embeddings
            hidden_states: 从project_hidden_states得到的hidden states
            cache_hidden: KV cache（如果使用）
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Past KV cache
            use_cache: 是否使用cache

        Returns:
            output_hidden_states: 输出hidden states
        """
        pass
```

### 2.2 配置类定义

定义您的模型配置（通常继承自Transformers的Config类）：

```python
from transformers import LlamaConfig  # 或其他backbone config

class YourAlgorithmConfig(LlamaConfig):
    model_type = "llama_your_algorithm"  # 唯一标识符

    def __init__(
        self,
        # 继承base model参数
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=1,  # Draft model层数（通常很少）
        # 添加算法特定参数
        draft_vocab_size=32000,  # Draft model词表大小
        num_heads=3,  # 如果有多个head（Medusa）
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            **kwargs
        )
        self.draft_vocab_size = draft_vocab_size
        self.num_heads = num_heads
```

### 2.3 完整实现示例（简化版Medusa Head）

```python
# specforge/modeling/draft/llama3_medusa.py
import torch
import torch.nn as nn
from transformers import LlamaConfig, PreTrainedModel

from specforge.modeling.draft.base import Eagle3DraftModel

class MedusaHead(nn.Module):
    """
    Medusa预测头：ResBlock + Linear
    """
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.resblock = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        # 零初始化权重（从identity开始）
        nn.init.zeros_(self.resblock[0].weight)

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        # 残差连接
        x = x + self.resblock(x)
        return self.lm_head(x)

class LlamaForCausalLMMedusa(Eagle3DraftModel):
    """
    Medusa for Llama
    """
    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        # Embedding（从target model加载）
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # 多个Medusa heads
        self.num_heads = getattr(config, 'num_heads', 3)
        self.medusa_heads = nn.ModuleList([
            MedusaHead(config.hidden_size, config.vocab_size)
            for _ in range(self.num_heads)
        ])

        # 词表映射（如果需要）
        if hasattr(config, 'draft_vocab_size') and config.draft_vocab_size != config.vocab_size:
            self.register_buffer("t2d", torch.zeros(config.vocab_size, dtype=torch.long))
            self.register_buffer("d2t", torch.zeros(config.draft_vocab_size, dtype=torch.long))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Medusa不需要投影，直接使用base model的输出
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        对于Medusa，返回多个head的logits

        Returns:
            logits: List of (batch, seq_len, vocab_size), length=num_heads
        """
        logits_list = []
        for head in self.medusa_heads:
            logits_list.append(head(hidden_states))

        # Stack成 (num_heads, batch, seq_len, vocab_size)
        return torch.stack(logits_list, dim=0)

    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Medusa不需要额外的backbone，直接返回hidden_states
        （所有计算在base model中完成）
        """
        return hidden_states
```

### 2.4 注册到Auto类

修改`specforge/modeling/auto.py`，添加您的模型映射：

```python
# specforge/modeling/auto.py
from .draft.llama3_medusa import LlamaForCausalLMMedusa

class AutoEagle3DraftModel(AutoModelForCausalLMBase):
    _model_mapping = {
        LlamaConfig: LlamaForCausalLMEagle3,  # 原有
        # 添加新算法
        # MedusaConfig: LlamaForCausalLMMedusa,  # 如果使用独立config
    }

class AutoDraftModelConfig:
    _config_mapping = {
        "LlamaForCausalLMEagle3": LlamaConfig,
        "LlamaForCausalLMMedusa": LlamaConfig,  # 添加新算法
    }
```

---

## 第三步：实现Core训练逻辑

### 3.1 创建训练模块

在`specforge/core/`中创建算法特定的训练逻辑。

**Eagle3示例**（`specforge/core/eagle3.py`）：

```python
class OnlineEagle3Model(nn.Module):
    """
    Eagle3在线训练：实时从target model提取hidden states
    """
    def __init__(self, draft_model, length=7, attention_backend="sdpa"):
        super().__init__()
        self.draft_model = draft_model
        self.length = length  # TTT循环次数
        self.attention_backend = attention_backend

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        hidden_states: torch.Tensor,  # 从target model来
        past_key_values: Optional[Tuple] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Step 1: 投影hidden states
        hidden_states = self.draft_model.project_hidden_states(hidden_states)

        # Step 2: TTT循环
        losses = []
        for idx in range(self.length):
            # 2.1 Embed input
            inputs_embeds = self.draft_model.embed_input_ids(input_ids)

            # 2.2 Forward backbone
            hidden_states_out = self.draft_model.backbone(
                input_embeds=inputs_embeds,
                hidden_states=hidden_states,
                cache_hidden=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=True,
            )

            # 2.3 Compute logits
            logits = self.draft_model.compute_logits(hidden_states_out)

            # 2.4 Calculate loss
            loss = compute_loss(logits, target, loss_mask)
            losses.append(loss)

            # 2.5 Update for next iteration
            hidden_states = hidden_states_out
            input_ids = update_input_ids(input_ids, logits)  # 根据预测更新

        return losses
```

### 3.2 Medusa训练逻辑（对比示例）

```python
# specforge/core/medusa.py
class OnlineMedusaModel(nn.Module):
    """
    Medusa在线训练：多头并行预测
    """
    def __init__(self, base_model, draft_model, num_heads=3):
        super().__init__()
        self.base_model = base_model  # Frozen base model
        self.draft_model = draft_model  # Medusa heads
        self.num_heads = num_heads

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,  # Future tokens
        **kwargs,
    ):
        # Step 1: 获取base model的hidden states
        with torch.no_grad():
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden_states = base_outputs.hidden_states[-1]  # 最后一层

        # Step 2: 所有Medusa heads并行预测
        # logits: (num_heads, batch, seq_len, vocab_size)
        logits = self.draft_model.compute_logits(hidden_states)

        # Step 3: 计算loss（每个head预测对应的future token）
        losses = []
        for head_idx in range(self.num_heads):
            # head_idx预测第(head_idx+1)个future token
            target = labels[:, head_idx+1:]  # 向后偏移
            head_logits = logits[head_idx, :, :-head_idx-1, :]  # 对齐

            loss = F.cross_entropy(
                head_logits.reshape(-1, head_logits.size(-1)),
                target.reshape(-1),
                ignore_index=-100
            )
            losses.append(loss)

        return sum(losses) / len(losses)
```

### 3.3 关键差异总结

| 特性 | Eagle3 | Medusa |
|------|--------|---------|
| **训练循环** | 递归TTT（多轮） | 单次forward |
| **Target提取** | 3层hidden states | 仅最后一层 |
| **损失计算** | 每轮一个loss | 每个head一个loss |
| **Input更新** | 每轮更新input_ids | 无更新 |

---

## 第四步：编写训练脚本

### 4.1 训练脚本结构

参考`scripts/train_eagle3_online.py`的结构：

```python
# scripts/train_your_algorithm_online.py
import argparse
import torch
from torch.utils.data import DataLoader

from specforge import AutoEagle3DraftModel, AutoDraftModelConfig
from specforge.core.your_algorithm import OnlineYourAlgorithmModel
from specforge.modeling.target import get_eagle3_target_model
from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
from specforge.optimizer import BF16Optimizer
from specforge.lr_scheduler import get_scheduler

def parse_args():
    parser = argparse.ArgumentParser()
    # Model args
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--draft-model-config", type=str, required=True)
    # Training args
    parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--output-dir", type=str, required=True)
    # 算法特定参数
    parser.add_argument("--num-heads", type=int, default=3)  # Medusa
    parser.add_argument("--ttt-length", type=int, default=7)  # Eagle3
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. 加载配置
    draft_config = AutoDraftModelConfig.from_file(args.draft_model_config)

    # 2. 初始化模型
    target_model = get_eagle3_target_model(
        model_path=args.target_model_path,
        backend="custom"  # 或 "sglang"
    )

    draft_model = AutoEagle3DraftModel.from_config(draft_config)
    draft_model.load_embedding(args.target_model_path)
    draft_model.freeze_embedding()

    # 3. 组装训练模型
    training_model = OnlineYourAlgorithmModel(
        target_model=target_model,
        draft_model=draft_model,
        num_heads=args.num_heads,  # 算法特定参数
    )

    # 4. 准备数据
    dataset = build_eagle3_dataset(
        data_path=args.train_data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # 5. 优化器和scheduler
    optimizer = BF16Optimizer(
        training_model.parameters(),
        lr=args.learning_rate,
    )
    scheduler = get_scheduler(optimizer, ...)

    # 6. 训练循环
    training_model.train()
    for epoch in range(args.num_epochs):
        for batch in dataloader:
            # Forward
            losses = training_model(**batch)
            loss = sum(losses) / len(losses)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    # 7. 保存模型
    draft_model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
```

### 4.2 数据处理复用

SpecForge的数据处理模块是通用的，可以直接复用：

```python
from specforge.data import (
    build_eagle3_dataset,         # 构建dataset
    generate_vocab_mapping_file,  # 生成词表映射
    prepare_dp_dataloaders,       # 分布式dataloader
)

# 如果算法需要不同的数据格式，可以继承并修改
class YourAlgorithmDataset(Eagle3Dataset):
    def __getitem__(self, idx):
        # 自定义数据处理逻辑
        ...
```

---

## 第五步：配置文件

### 5.1 Draft Model配置

在`configs/`目录下创建JSON配置文件：

```json
// configs/llama3-8B-your-algorithm.json
{
  "architectures": [
    "LlamaForCausalLMYourAlgorithm"
  ],
  "bos_token_id": 128000,
  "eos_token_id": 128001,
  "hidden_act": "silu",
  "hidden_size": 4096,         // Base model hidden size
  "intermediate_size": 14336,
  "max_position_embeddings": 2048,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "num_hidden_layers": 1,      // Draft model层数（Medusa: 0, Eagle3: 1）
  "pad_token_id": 0,
  "rms_norm_eps": 1e-05,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "use_cache": true,
  "vocab_size": 128256,        // Target model词表
  "draft_vocab_size": 32000,   // Draft model词表（如果不同）

  // 算法特定参数
  "num_heads": 3,              // Medusa heads数量
  "medusa_num_layers": 1       // 每个head的层数
}
```

### 5.2 参数设置建议

基于原始论文和实践经验：

| 算法 | Draft Layers | Hidden Size | Heads | LR | Batch Size | Epochs |
|------|--------------|-------------|-------|-----|------------|--------|
| Eagle3 | 1 | Same as base | 1 | 1e-4 | 4 | 1-2 |
| Medusa-1 | 0 (only heads) | Same as base | 3-4 | 1e-3 | 8 | 2 |
| Medusa-2 | 全模型 | Same as base | 3-4 | 1e-5 | 4 | 2 |

**重要注意事项**：
- Medusa-1的学习率比Eagle3高（1e-3 vs 1e-4），因为只训练小的heads
- Medusa-2训练全模型，学习率需要更小（1e-5）
- Draft layers=0表示没有额外的transformer层，只有heads

---

## 第六步：创建使用示例

### 6.1 Example脚本结构

在`examples/`下创建您的示例：

```bash
examples/
├── README.md                              # 更新总README
├── prepare_hidden_states.sh              # 通用（复用）
└── run_llama3_your_algorithm_online.sh   # 您的算法
```

### 6.2 训练脚本示例

```bash
#!/bin/bash
# examples/run_llama3_medusa_online.sh

# 设置路径
ROOT_DIR=$(pwd)
TARGET_MODEL="meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR="${ROOT_DIR}/outputs/llama3-8b-medusa"

# 生成词表映射（如果需要）
python scripts/generate_vocab_mapping.py \
    --target-model-path ${TARGET_MODEL} \
    --draft-vocab-size 32000 \
    --output-path ${ROOT_DIR}/cache/vocab_mapping.pt

# 训练
torchrun \
    --standalone \
    --nproc_per_node 4 \
    scripts/train_medusa_online.py \
    --target-model-path ${TARGET_MODEL} \
    --draft-model-config configs/llama3-8B-medusa.json \
    --train-data-path ${ROOT_DIR}/data/sharegpt.jsonl \
    --output-dir ${OUTPUT_DIR} \
    --num-epochs 2 \
    --batch-size 8 \
    --learning-rate 1e-3 \
    --max-length 2048 \
    --num-heads 3 \
    --cache-dir ${ROOT_DIR}/cache
```

---

## 第七步：测试与验证

### 7.1 单元测试

创建测试文件`tests/test_your_algorithm.py`：

```python
import torch
import pytest
from specforge import AutoEagle3DraftModel, AutoDraftModelConfig

def test_draft_model_forward():
    config = AutoDraftModelConfig.from_file("configs/llama3-8B-your-algorithm.json")
    model = AutoEagle3DraftModel.from_config(config)

    # 测试forward
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    hidden_states = torch.randn(2, 10, config.hidden_size)

    outputs = model.backbone(
        input_embeds=model.embed_input_ids(input_ids),
        hidden_states=hidden_states,
        cache_hidden=None,
        attention_mask=None,
        position_ids=None,
    )

    assert outputs.shape == (2, 10, config.hidden_size)

def test_compute_logits():
    config = AutoDraftModelConfig.from_file("configs/llama3-8B-your-algorithm.json")
    model = AutoEagle3DraftModel.from_config(config)

    hidden_states = torch.randn(2, 10, config.hidden_size)
    logits = model.compute_logits(hidden_states)

    # Medusa: (num_heads, batch, seq, vocab)
    # Eagle3: (batch, seq, vocab)
    assert logits.dim() in [3, 4]
```

### 7.2 端到端训练测试

```bash
# 小规模测试
bash examples/run_llama3_your_algorithm_online.sh \
    --num-epochs 1 \
    --batch-size 2 \
    --max-length 512 \
    --train-data-path tests/data/tiny_dataset.jsonl
```

### 7.3 验证检查清单

- [ ] 模型可以正常初始化
- [ ] Forward pass没有shape错误
- [ ] Loss可以正常计算并backward
- [ ] 保存和加载checkpoint正常
- [ ] 分布式训练正常（如果使用）
- [ ] 与参考实现的输出对齐（如果有）

---

## 常见问题与解决方案

### Q1: 我的算法不需要target hidden states怎么办？

**A**: 如Medusa只需要base model的最后一层输出，可以：
1. 在`Core Module`的forward中，用`torch.no_grad()`调用frozen base model
2. `project_hidden_states()`直接返回输入（不做投影）
3. `backbone()`返回hidden_states（无额外计算）

```python
def forward(self, input_ids, **kwargs):
    # 只需要最后一层
    with torch.no_grad():
        base_outputs = self.base_model(input_ids, output_hidden_states=True)
    hidden_states = base_outputs.hidden_states[-1]

    # 直接用于heads
    logits = self.draft_model.compute_logits(hidden_states)
    ...
```

### Q2: 如何实现Medusa的树形验证？

**A**: 树形验证通常在**推理阶段**，不在训练阶段。SpecForge专注于训练，推理逻辑应该在SGLang中实现。但您可以在`specforge/inference/`中添加验证逻辑供测试使用。

### Q3: 我的算法需要特殊的attention mask怎么办？

**A**: 在`backbone()`方法中自定义attention mask生成逻辑：

```python
def backbone(self, ..., attention_mask, ...):
    # 自定义mask
    if self.special_attention:
        attention_mask = self.create_special_mask(...)

    # 传入transformer layers
    ...
```

或在`Core Module`的forward中修改mask后传入。

### Q4: 词表大小不匹配怎么处理？

**A**: 使用词表映射：
1. 生成映射文件（`scripts/generate_vocab_mapping.py`）
2. 在Draft Model中注册`t2d`和`d2t` buffers
3. 在forward前后进行映射：

```python
# Target → Draft
draft_ids = self.t2d[target_ids]

# Draft → Target
target_ids = self.d2t[draft_ids]
```

### Q5: 如何调试训练不稳定？

**A**: 常见解决方案：
- **梯度裁剪**：`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)`
- **降低学习率**：从`1e-5`开始尝试
- **增加warmup**：`warmup_ratio=0.1`
- **检查loss scale**：确保loss数值在合理范围
- **冻结base model**：先训练heads再训练全模型

---

## 完整实现Checklist

在完成实现后，请确认以下所有项：

### 代码实现
- [ ] Draft Model继承自`Eagle3DraftModel`并实现所有抽象方法
- [ ] 配置类正确定义并包含所有必要参数
- [ ] Core训练逻辑正确实现（Online/Offline）
- [ ] 注册到`auto.py`的映射表
- [ ] 导出到`__init__.py`

### 训练脚本
- [ ] 训练脚本可以正常运行
- [ ] 参数解析完整
- [ ] 支持分布式训练（如果需要）
- [ ] 支持checkpoint保存和恢复

### 配置和文档
- [ ] 配置文件格式正确
- [ ] 参数设置合理（基于论文）
- [ ] Example脚本可以运行
- [ ] README文档更新

### 测试
- [ ] 单元测试通过
- [ ] 端到端训练测试通过
- [ ] 与参考实现对比（如果有）

---

## 参考资料

### SpecForge源码参考
- **Eagle3 Draft Model**: `specforge/modeling/draft/llama3_eagle.py` (1107行)
- **Eagle3 Core**: `specforge/core/eagle3.py` (608行)
- **Base Class**: `specforge/modeling/draft/base.py` (194行)
- **Training Script**: `scripts/train_eagle3_online.py`

### 论文参考
- **EAGLE**: "EAGLE: Lossless Acceleration of LLM Decoding by Feature Extrapolation"
- **EAGLE-2**: "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"
- **Medusa**: "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" (arXiv:2401.10774)

### 社区资源
- [SpecForge Documentation](https://docs.sglang.ai/SpecForge/)
- [EAGLE GitHub](https://github.com/SafeAILab/EAGLE)
- [Medusa GitHub](https://github.com/FasterDecoding/Medusa)
- [SGLang GitHub](https://github.com/sgl-project/sglang)

---

## 下一步：Medusa实现示例

建议您按照以下顺序实现Medusa：

1. **阅读参考代码**：`参考目录/Medusa-main/代码架构详解.md`
2. **创建Draft Model**：`specforge/modeling/draft/llama3_medusa.py`
3. **创建Core逻辑**：`specforge/core/medusa.py`
4. **编写训练脚本**：`scripts/train_medusa_online.py`
5. **添加配置文件**：`configs/llama3-8B-medusa.json`
6. **创建Example**：`examples/run_llama3_medusa_online.sh`

下一个文档《Medusa在SpecForge中的实现教程》将提供具体的实现步骤。

---

**文档版本**: v1.0
**最后更新**: 2025-11-14
**作者**: SpecForge Team
**反馈**: 如有问题请提交GitHub Issue
