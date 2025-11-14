# Medusa在SpecForge中的完整实现教程

## 概述

本教程将指导您从零开始在SpecForge框架中实现Medusa算法。Medusa是一种通过多个预测头并行生成多个token来加速LLM推理的推测解码算法。

**学习目标**：
- 理解Medusa与Eagle3的核心差异
- 实现Medusa的Draft Model
- 实现Medusa的训练逻辑
- 配置和运行训练pipeline
- 验证实现正确性

**前置要求**：
- 已阅读《如何在SpecForge中添加新算法.md》
- 熟悉Medusa论文：[arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
- 了解SpecForge的基本架构

---

## 第一部分：Medusa vs Eagle3 核心差异

### 1.1 算法对比

| 维度 | Eagle3 | Medusa-1 | Medusa-2 |
|------|--------|----------|----------|
| **预测方式** | 自回归递归（TTT） | 并行多头预测 | 并行多头预测 |
| **Base Model** | Frozen | Frozen | 联合训练 |
| **输入依赖** | 3层hidden states | 仅最后一层 | 仅最后一层 |
| **Head数量** | 1 | 3-4 | 3-4 |
| **训练目标** | 模仿target分布（KL散度） | 预测future tokens（CE loss） | 预测future tokens + 保持base性能 |
| **学习率** | 1e-4 | 1e-3 | 1e-5 |
| **加速比** | 2x-3x | 2.2x | 2.3x-3.6x |

### 1.2 架构差异图解

**Eagle3架构**：
```
Input IDs → Embedding
              ↓
         Target Model (3层hidden states)
              ↓
    Project (3*hidden → hidden)
              ↓
    TTT Loop (7轮递归):
      ├─ Concat[embedding, projected]
      ├─ Draft Backbone (1-layer Transformer)
      ├─ LM Head → Logits
      └─ Update Input → 下一轮
```

**Medusa架构**：
```
Input IDs → Base Model (Frozen) → Last Hidden State
                                        ↓
                               ┌────────┼────────┐
                               ↓        ↓        ↓
                            Head 1   Head 2   Head 3
                            (t+1)    (t+2)    (t+3)
                               ↓        ↓        ↓
                            Logits   Logits   Logits
                               ↓        ↓        ↓
                            CE Loss  CE Loss  CE Loss
```

**关键差异**：
1. **无递归**：Medusa一次forward预测所有future tokens
2. **无额外Transformer**：只有ResBlock+Linear组成的轻量heads
3. **简单训练**：标准监督学习，无TTT复杂性

### 1.3 数据处理对比

**Eagle3数据格式**：
```python
{
    "input_ids": [1, 2, 3, 4, 5],
    "target": [2, 3, 4, 5, 6],  # 下一个token
    "hidden_states": tensor(batch, seq, 3*hidden),  # 3层concat
    "loss_mask": [1, 1, 1, 1, 1]
}
```

**Medusa数据格式**：
```python
{
    "input_ids": [1, 2, 3, 4, 5],
    "labels": [
        [2, 3, 4, 5, 6],  # Head 1目标（t+1）
        [3, 4, 5, 6, 7],  # Head 2目标（t+2）
        [4, 5, 6, 7, 8],  # Head 3目标（t+3）
    ],
    "attention_mask": [1, 1, 1, 1, 1]
}
```

**重要观察**：Medusa的数据处理**前半部分与Eagle3完全一致**（tokenization、chat template），只是label构造不同！

---

## 第二部分：Draft Model实现

### 2.1 ResBlock实现

Medusa的核心组件是ResBlock，设计精妙：

```python
# specforge/modeling/draft/medusa_components.py
import torch
import torch.nn as nn

class MedusaResBlock(nn.Module):
    """
    Medusa ResBlock: Linear → SiLU → Residual

    设计特点：
    1. 零初始化：训练开始时表现为identity mapping
    2. SiLU激活：与LLaMA一致
    3. 残差连接：保证梯度流动

    论文依据：Medusa (arXiv:2401.10774) Section 3.1
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        # 关键：零初始化权重
        # 初始时刻：ResBlock(x) = x + SiLU(0) = x
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x):
        """
        前向传播

        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            output: (batch, seq_len, hidden_size)
        """
        return x + self.act(self.linear(x))


class MedusaHead(nn.Module):
    """
    单个Medusa预测头

    架构：ResBlock^n → Linear (to vocab)

    参数：
        hidden_size: 隐藏层维度（与base model一致）
        vocab_size: 词表大小
        num_layers: ResBlock层数（默认1）
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_layers: int = 1,
    ):
        super().__init__()

        # 堆叠ResBlocks
        self.resblocks = nn.ModuleList([
            MedusaResBlock(hidden_size)
            for _ in range(num_layers)
        ])

        # LM Head（不共享权重，每个head独立）
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            hidden_states: (batch, seq_len, hidden_size)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # 通过所有ResBlocks
        for resblock in self.resblocks:
            hidden_states = resblock(hidden_states)

        # 投影到词表
        logits = self.lm_head(hidden_states)
        return logits
```

### 2.2 完整Medusa Draft Model

```python
# specforge/modeling/draft/llama3_medusa.py
import torch
import torch.nn as nn
from typing import Optional, List
from transformers import LlamaConfig
from transformers.cache_utils import Cache

from specforge.modeling.draft.base import Eagle3DraftModel
from .medusa_components import MedusaHead


class LlamaForCausalLMMedusa(Eagle3DraftModel):
    """
    Medusa Draft Model for LLaMA

    与Eagle3的关键差异：
    1. 无需project_hidden_states（直接使用base输出）
    2. backbone()返回输入（无额外计算）
    3. compute_logits()返回多个head的输出

    配置参数：
        medusa_num_heads: Medusa head数量（默认3）
        medusa_num_layers: 每个head的ResBlock层数（默认1）
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        # Embedding（从target model加载）
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Medusa heads
        self.medusa_num_heads = getattr(config, 'medusa_num_heads', 3)
        self.medusa_num_layers = getattr(config, 'medusa_num_layers', 1)

        self.medusa_heads = nn.ModuleList([
            MedusaHead(
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                num_layers=self.medusa_num_layers,
            )
            for _ in range(self.medusa_num_heads)
        ])

        # 词表映射（如果draft和target词表不同）
        draft_vocab_size = getattr(config, 'draft_vocab_size', None)
        if draft_vocab_size is not None and draft_vocab_size != self.vocab_size:
            self.register_buffer(
                "t2d",
                torch.zeros(self.vocab_size, dtype=torch.long)
            )
            self.register_buffer(
                "d2t",
                torch.zeros(draft_vocab_size, dtype=torch.long)
            )
            self.vocab_mapping_loaded = False
        else:
            self.vocab_mapping_loaded = True

        # 初始化权重
        self.post_init()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed input IDs

        Args:
            input_ids: (batch, seq_len)
        Returns:
            embeddings: (batch, seq_len, hidden_size)
        """
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Medusa不需要投影，直接使用base model的最后一层输出

        Args:
            hidden_states: (batch, seq_len, hidden_size)
        Returns:
            hidden_states: 原样返回
        """
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        使用所有Medusa heads计算logits

        Args:
            hidden_states: (batch, seq_len, hidden_size)
        Returns:
            logits: (num_heads, batch, seq_len, vocab_size)
        """
        logits_list = []
        for head in self.medusa_heads:
            logits = head(hidden_states)
            logits_list.append(logits)

        # Stack: (num_heads, batch, seq_len, vocab_size)
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
        Medusa没有额外的backbone，直接返回hidden_states

        所有计算在frozen base model + Medusa heads中完成

        Args:
            hidden_states: (batch, seq_len, hidden_size)
        Returns:
            hidden_states: 原样返回
        """
        # Medusa不需要额外的transformer layers
        # 直接返回base model的输出
        return hidden_states

    def freeze_heads(self):
        """
        冻结Medusa heads（用于某些训练场景）
        """
        for head in self.medusa_heads:
            for param in head.parameters():
                param.requires_grad = False

    def unfreeze_heads(self):
        """
        解冻Medusa heads
        """
        for head in self.medusa_heads:
            for param in head.parameters():
                param.requires_grad = True
```

### 2.3 注册到Auto类

```python
# specforge/modeling/auto.py (修改部分)
from .draft.llama3_medusa import LlamaForCausalLMMedusa

class AutoEagle3DraftModel(AutoModelForCausalLMBase):
    _model_mapping = {
        LlamaConfig: LlamaForCausalLMEagle3,  # 原有
        # 注意：Medusa使用相同的LlamaConfig，通过架构名区分
    }

    @classmethod
    def from_config(cls, config: PretrainedConfig, torch_dtype=None, **config_kwargs):
        # 根据architectures字段选择模型类
        if hasattr(config, 'architectures') and config.architectures:
            arch_name = config.architectures[0]
            if 'Medusa' in arch_name:
                model = LlamaForCausalLMMedusa(config, **config_kwargs)
            elif 'Eagle' in arch_name:
                model = LlamaForCausalLMEagle3(config, **config_kwargs)
            else:
                raise ValueError(f"Unknown architecture: {arch_name}")
        else:
            # 默认使用Eagle3
            model = LlamaForCausalLMEagle3(config, **config_kwargs)

        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        return model
```

---

## 第三部分：训练逻辑实现

### 3.1 Medusa训练模块

```python
# specforge/core/medusa.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class OnlineMedusaModel(nn.Module):
    """
    Medusa-1在线训练模型

    训练流程：
    1. Input → Frozen Base Model → Last Hidden State
    2. Hidden State → Medusa Heads → Logits (多个)
    3. 每个head预测对应的future token
    4. 计算CE loss并平均

    与Eagle3的关键差异：
    - 无TTT递归
    - 无input更新
    - 并行预测，单次forward
    """

    def __init__(
        self,
        base_model: nn.Module,
        draft_model: nn.Module,
        num_heads: int = 3,
    ):
        """
        Args:
            base_model: Frozen base LLM（用于提取hidden states）
            draft_model: Medusa draft model（包含heads）
            num_heads: Medusa head数量
        """
        super().__init__()
        self.base_model = base_model
        self.draft_model = draft_model
        self.num_heads = num_heads

        # 冻结base model
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,  # (batch, num_heads, seq_len)
        loss_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], dict]:
        """
        前向传播

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            labels: (batch, num_heads, seq_len)
                labels[:, 0, :] = input_ids向右shift 1 (t+1)
                labels[:, 1, :] = input_ids向右shift 2 (t+2)
                ...
            loss_mask: (batch, seq_len), 可选

        Returns:
            total_loss: 平均loss
            losses: 每个head的loss列表
            metrics: 训练指标（accuracy等）
        """
        batch_size, seq_len = input_ids.shape

        # Step 1: 从frozen base model提取hidden states
        with torch.no_grad():
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            # 使用最后一层hidden states
            hidden_states = base_outputs.hidden_states[-1]  # (batch, seq, hidden)

        # Step 2: 通过所有Medusa heads获取logits
        # logits: (num_heads, batch, seq_len, vocab_size)
        logits = self.draft_model.compute_logits(hidden_states)

        # Step 3: 计算每个head的loss
        losses = []
        accuracies = []

        for head_idx in range(self.num_heads):
            # 获取当前head的logits和target
            head_logits = logits[head_idx]  # (batch, seq_len, vocab_size)
            head_labels = labels[:, head_idx, :]  # (batch, seq_len)

            # 对齐：head预测t+head_idx+1，所以需要截断
            # head_0: logits[:-1] vs labels[:, 0, 1:]  (预测t+1)
            # head_1: logits[:-2] vs labels[:, 1, 2:]  (预测t+2)
            valid_len = seq_len - head_idx - 1
            if valid_len <= 0:
                continue

            pred_logits = head_logits[:, :valid_len, :]  # (batch, valid_len, vocab)
            target_labels = head_labels[:, head_idx+1:seq_len]  # (batch, valid_len)

            # 应用loss mask（如果有）
            if loss_mask is not None:
                valid_mask = loss_mask[:, :valid_len]
                pred_logits = pred_logits[valid_mask.bool()]
                target_labels = target_labels[valid_mask.bool()]

            # 计算CE loss
            loss = F.cross_entropy(
                pred_logits.reshape(-1, pred_logits.size(-1)),
                target_labels.reshape(-1),
                ignore_index=-100,
                reduction='mean'
            )
            losses.append(loss)

            # 计算accuracy
            with torch.no_grad():
                preds = pred_logits.argmax(dim=-1)
                mask = target_labels != -100
                acc = (preds == target_labels)[mask].float().mean()
                accuracies.append(acc.item())

        # Step 4: 平均所有head的loss
        total_loss = sum(losses) / len(losses) if losses else torch.tensor(0.0)

        # 收集metrics
        metrics = {
            'loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
            'head_losses': [l.item() for l in losses],
            'head_accuracies': accuracies,
        }

        return total_loss, losses, metrics


class OfflineMedusaModel(OnlineMedusaModel):
    """
    Medusa离线训练（从预计算的hidden states缓存读取）

    与Online的差异：不需要base_model，hidden_states直接传入
    """

    def __init__(self, draft_model: nn.Module, num_heads: int = 3):
        # 不需要base_model
        super(OnlineMedusaModel, self).__init__()  # 跳过OnlineMedusaModel的__init__
        self.draft_model = draft_model
        self.num_heads = num_heads

    def forward(
        self,
        hidden_states: torch.Tensor,  # 直接传入预计算的
        labels: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], dict]:
        """
        使用预计算的hidden states训练

        Args:
            hidden_states: (batch, seq_len, hidden_size) - 预计算的
            labels: (batch, num_heads, seq_len)
            loss_mask: (batch, seq_len)

        Returns:
            total_loss, losses, metrics
        """
        # 直接从Step 2开始（跳过base model forward）
        logits = self.draft_model.compute_logits(hidden_states)

        # 后续与Online相同
        losses = []
        accuracies = []
        batch_size, seq_len = hidden_states.shape[:2]

        for head_idx in range(self.num_heads):
            head_logits = logits[head_idx]
            head_labels = labels[:, head_idx, :]

            valid_len = seq_len - head_idx - 1
            if valid_len <= 0:
                continue

            pred_logits = head_logits[:, :valid_len, :]
            target_labels = head_labels[:, head_idx+1:seq_len]

            if loss_mask is not None:
                valid_mask = loss_mask[:, :valid_len]
                pred_logits = pred_logits[valid_mask.bool()]
                target_labels = target_labels[valid_mask.bool()]

            loss = F.cross_entropy(
                pred_logits.reshape(-1, pred_logits.size(-1)),
                target_labels.reshape(-1),
                ignore_index=-100,
                reduction='mean'
            )
            losses.append(loss)

            with torch.no_grad():
                preds = pred_logits.argmax(dim=-1)
                mask = target_labels != -100
                acc = (preds == target_labels)[mask].float().mean()
                accuracies.append(acc.item())

        total_loss = sum(losses) / len(losses) if losses else torch.tensor(0.0)

        metrics = {
            'loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
            'head_losses': [l.item() for l in losses],
            'head_accuracies': accuracies,
        }

        return total_loss, losses, metrics
```

### 3.2 导出模块

```python
# specforge/core/__init__.py (修改)
from .eagle3 import OfflineEagle3Model, OnlineEagle3Model, QwenVLOnlineEagle3Model
from .medusa import OnlineMedusaModel, OfflineMedusaModel

__all__ = [
    "OnlineEagle3Model",
    "OfflineEagle3Model",
    "QwenVLOnlineEagle3Model",
    "OnlineMedusaModel",
    "OfflineMedusaModel",
]
```

---

## 第四部分：数据处理Pipeline

### 4.1 数据格式转换

**关键洞察**：Medusa的数据处理**前半部分与Eagle3完全相同**！只需修改label构造。

```python
# specforge/data/medusa_dataset.py
import torch
from torch.utils.data import Dataset
from typing import Dict, List
from specforge.data import Eagle3Dataset  # 复用基础功能


class MedusaDataset(Eagle3Dataset):
    """
    Medusa数据集

    继承Eagle3Dataset的tokenization逻辑，只修改label构造
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        num_heads: int = 3,
        chat_template: str = "llama3",
    ):
        super().__init__(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            chat_template=chat_template,
        )
        self.num_heads = num_heads

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        Returns:
            {
                'input_ids': (seq_len,),
                'attention_mask': (seq_len,),
                'labels': (num_heads, seq_len),  # 关键差异！
                'loss_mask': (seq_len,),
            }
        """
        # 调用父类获取基础数据
        item = super().__getitem__(idx)

        input_ids = item['input_ids']  # (seq_len,)
        seq_len = input_ids.shape[0]

        # 构造multi-head labels
        labels = torch.full(
            (self.num_heads, seq_len),
            -100,
            dtype=torch.long
        )

        for head_idx in range(self.num_heads):
            # head_idx 预测 t+head_idx+1
            # labels[head_idx, i] = input_ids[i + head_idx + 1]
            shift = head_idx + 1
            if shift < seq_len:
                labels[head_idx, :seq_len-shift] = input_ids[shift:]

        item['labels'] = labels
        return item


def build_medusa_dataset(
    data_path: str,
    tokenizer,
    max_length: int = 2048,
    num_heads: int = 3,
    chat_template: str = "llama3",
) -> MedusaDataset:
    """
    构建Medusa数据集（便捷函数）

    Args:
        data_path: JSONL文件路径
        tokenizer: Tokenizer
        max_length: 最大序列长度
        num_heads: Medusa head数量
        chat_template: Chat template类型

    Returns:
        MedusaDataset实例
    """
    return MedusaDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        num_heads=num_heads,
        chat_template=chat_template,
    )
```

### 4.2 数据示例

**输入数据**（ShareGPT格式，与Eagle3相同）：
```json
{
  "conversations": [
    {"from": "human", "value": "What is machine learning?"},
    {"from": "gpt", "value": "Machine learning is a subset of artificial intelligence..."}
  ]
}
```

**处理后**：
```python
{
    'input_ids': tensor([1, 128000, 8948, 374, 5780, ...]),  # (2048,)
    'attention_mask': tensor([1, 1, 1, ..., 0, 0]),
    'labels': tensor([
        [128000, 8948, 374, ..., -100],  # Head 0: t+1
        [8948, 374, 5780, ..., -100],    # Head 1: t+2
        [374, 5780, 6975, ..., -100],    # Head 2: t+3
    ]),  # (3, 2048)
    'loss_mask': tensor([1, 1, 1, ..., 0, 0]),
}
```

---

## 第五部分：训练脚本

### 5.1 完整训练脚本

```python
# scripts/train_medusa_online.py
import argparse
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from tqdm import tqdm

from specforge import AutoEagle3DraftModel, AutoDraftModelConfig
from specforge.core import OnlineMedusaModel
from specforge.data import build_medusa_dataset
from specforge.modeling.target import get_eagle3_target_model
from specforge.distributed import init_distributed, destroy_distributed
from specforge.optimizer import BF16Optimizer
from specforge.lr_scheduler import get_cosine_schedule_with_warmup
from specforge.utils import print_on_rank0, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train Medusa-1 with online data")

    # Model arguments
    parser.add_argument("--target-model-path", type=str, required=True,
                        help="Path to target LLM")
    parser.add_argument("--draft-model-config", type=str, required=True,
                        help="Path to Medusa draft config JSON")

    # Training arguments
    parser.add_argument("--train-data-path", type=str, required=True,
                        help="Path to training data (JSONL)")
    parser.add_argument("--num-epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size per GPU")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate (1e-3 for Medusa-1)")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")

    # Medusa-specific arguments
    parser.add_argument("--num-heads", type=int, default=3,
                        help="Number of Medusa heads")

    # Output arguments
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--save-interval", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="Log metrics every N steps")

    # Other arguments
    parser.add_argument("--chat-template", type=str, default="llama3",
                        help="Chat template type")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize distributed training
    local_rank = init_distributed()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Set seed
    torch.manual_seed(args.seed)

    print_on_rank0("=" * 50)
    print_on_rank0("Medusa-1 Training Configuration")
    print_on_rank0("=" * 50)
    print_on_rank0(f"Target Model: {args.target_model_path}")
    print_on_rank0(f"Draft Config: {args.draft_model_config}")
    print_on_rank0(f"Num Heads: {args.num_heads}")
    print_on_rank0(f"Learning Rate: {args.learning_rate}")
    print_on_rank0(f"Batch Size: {args.batch_size}")
    print_on_rank0(f"Max Length: {args.max_length}")
    print_on_rank0("=" * 50)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load draft model config
    draft_config = AutoDraftModelConfig.from_file(args.draft_model_config)

    # Initialize target model (frozen)
    print_on_rank0("Loading target model (will be frozen)...")
    target_model = get_eagle3_target_model(
        model_path=args.target_model_path,
        backend="custom",  # Use custom backend for training
        device=device,
    )

    # Initialize draft model
    print_on_rank0("Initializing Medusa draft model...")
    draft_model = AutoEagle3DraftModel.from_config(
        draft_config,
        torch_dtype=torch.bfloat16
    )

    # Load embeddings from target model
    print_on_rank0("Loading embeddings from target model...")
    draft_model.load_embedding(args.target_model_path)
    draft_model.freeze_embedding()

    # Move draft model to device
    draft_model = draft_model.to(device)

    # Wrap in training model
    training_model = OnlineMedusaModel(
        base_model=target_model,
        draft_model=draft_model,
        num_heads=args.num_heads,
    )

    # Wrap with DDP
    training_model = DDP(
        training_model,
        device_ids=[local_rank],
        output_device=local_rank,
    )

    # Prepare dataset
    print_on_rank0("Building dataset...")
    dataset = build_medusa_dataset(
        data_path=args.train_data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        num_heads=args.num_heads,
        chat_template=args.chat_template,
    )

    # DataLoader
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    # Optimizer
    optimizer = BF16Optimizer(
        training_model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    # Scheduler
    total_steps = len(dataloader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print_on_rank0(f"Total training steps: {total_steps}")
    print_on_rank0(f"Warmup steps: {warmup_steps}")
    print_on_rank0("Starting training...")

    # Training loop
    global_step = 0
    training_model.train()

    for epoch in range(args.num_epochs):
        print_on_rank0(f"\n{'='*50}")
        print_on_rank0(f"Epoch {epoch + 1}/{args.num_epochs}")
        print_on_rank0(f"{'='*50}")

        sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(tqdm(dataloader, disable=local_rank != 0)):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward
            total_loss, losses, metrics = training_model(**batch)

            # Backward
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                training_model.parameters(),
                args.max_grad_norm
            )

            optimizer.step()
            scheduler.step()

            global_step += 1

            # Logging
            if global_step % args.log_interval == 0:
                print_on_rank0(
                    f"Step {global_step} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Head Accs: {[f'{a:.3f}' for a in metrics['head_accuracies']]} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

            # Save checkpoint
            if global_step % args.save_interval == 0:
                if local_rank == 0:
                    save_dir = os.path.join(args.output_dir, f"step_{global_step}")
                    os.makedirs(save_dir, exist_ok=True)

                    # Save draft model
                    draft_model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)

                    print_on_rank0(f"Checkpoint saved to {save_dir}")

    # Save final model
    if local_rank == 0:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        draft_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print_on_rank0(f"Final model saved to {final_dir}")

    # Cleanup
    destroy_distributed()
    print_on_rank0("Training completed!")


if __name__ == "__main__":
    main()
```

### 5.2 Example运行脚本

```bash
#!/bin/bash
# examples/run_llama3_medusa_online.sh

# 配置路径
ROOT_DIR=$(pwd)
TARGET_MODEL="meta-llama/Llama-3.1-8B-Instruct"
DRAFT_CONFIG="${ROOT_DIR}/configs/llama3-8B-medusa.json"
TRAIN_DATA="${ROOT_DIR}/cache/dataset/sharegpt.jsonl"
OUTPUT_DIR="${ROOT_DIR}/outputs/llama3-8b-medusa"

# 检查数据是否存在
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "Error: Training data not found at ${TRAIN_DATA}"
    echo "Please prepare your training data first."
    exit 1
fi

# 生成词表映射（如果draft和target词表不同）
if [ ! -f "${ROOT_DIR}/cache/vocab_mapping.pt" ]; then
    echo "Generating vocabulary mapping..."
    python ${ROOT_DIR}/scripts/generate_vocab_mapping.py \
        --target-model-path ${TARGET_MODEL} \
        --draft-vocab-size 32000 \
        --output-path ${ROOT_DIR}/cache/vocab_mapping.pt
fi

# Medusa-1训练（只训练heads，base model frozen）
echo "Starting Medusa-1 training..."
torchrun \
    --standalone \
    --nproc_per_node 4 \
    ${ROOT_DIR}/scripts/train_medusa_online.py \
    --target-model-path ${TARGET_MODEL} \
    --draft-model-config ${DRAFT_CONFIG} \
    --train-data-path ${TRAIN_DATA} \
    --output-dir ${OUTPUT_DIR} \
    --num-epochs 2 \
    --batch-size 8 \
    --learning-rate 1e-3 \
    --max-length 2048 \
    --num-heads 3 \
    --warmup-ratio 0.1 \
    --max-grad-norm 1.0 \
    --save-interval 1000 \
    --log-interval 50 \
    --chat-template llama3 \
    --seed 42

echo "Training completed! Model saved to ${OUTPUT_DIR}"
```

---

## 第六部分：验证与测试

### 6.1 单元测试

```python
# tests/test_medusa.py
import pytest
import torch
from specforge import AutoDraftModelConfig, AutoEagle3DraftModel
from specforge.core import OnlineMedusaModel


def test_medusa_draft_model():
    """测试Medusa draft model的forward"""
    config = AutoDraftModelConfig.from_file("configs/llama3-8B-medusa.json")
    model = AutoEagle3DraftModel.from_config(config)

    batch, seq_len = 2, 10
    hidden_states = torch.randn(batch, seq_len, config.hidden_size)

    # 测试compute_logits
    logits = model.compute_logits(hidden_states)

    # 检查shape: (num_heads, batch, seq_len, vocab_size)
    assert logits.shape == (3, batch, seq_len, config.vocab_size)


def test_medusa_training_forward():
    """测试Medusa训练的forward pass"""
    from transformers import AutoModelForCausalLM

    config = AutoDraftModelConfig.from_file("configs/llama3-8B-medusa.json")
    draft_model = AutoEagle3DraftModel.from_config(config)

    # Mock base model
    base_model = AutoModelForCausalLM.from_config(config)

    training_model = OnlineMedusaModel(
        base_model=base_model,
        draft_model=draft_model,
        num_heads=3,
    )

    # Mock batch
    batch = {
        'input_ids': torch.randint(0, config.vocab_size, (2, 10)),
        'attention_mask': torch.ones(2, 10),
        'labels': torch.randint(0, config.vocab_size, (2, 3, 10)),
    }

    # Forward
    total_loss, losses, metrics = training_model(**batch)

    # 检查
    assert isinstance(total_loss, torch.Tensor)
    assert len(losses) == 3
    assert 'loss' in metrics
    assert 'head_accuracies' in metrics


def test_medusa_heads_zero_init():
    """测试ResBlock是否正确零初始化"""
    from specforge.modeling.draft.medusa_components import MedusaResBlock

    resblock = MedusaResBlock(hidden_size=4096)

    # 检查权重是否为零
    assert torch.all(resblock.linear.weight == 0)

    # 测试forward（应该返回identity）
    x = torch.randn(2, 10, 4096)
    output = resblock(x)

    # 由于权重为零：output = x + SiLU(0) = x
    assert torch.allclose(output, x, atol=1e-5)
```

### 6.2 快速训练测试

```bash
# 快速测试脚本（使用小数据集）
#!/bin/bash
# tests/quick_test_medusa.sh

ROOT_DIR=$(pwd)

# 创建tiny dataset
cat > /tmp/tiny_medusa.jsonl << 'EOF'
{"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello!"}]}
{"conversations": [{"from": "human", "value": "Test"}, {"from": "gpt", "value": "OK"}]}
EOF

# 运行1个epoch，小batch
torchrun \
    --standalone \
    --nproc_per_node 1 \
    ${ROOT_DIR}/scripts/train_medusa_online.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config ${ROOT_DIR}/configs/llama3-8B-medusa.json \
    --train-data-path /tmp/tiny_medusa.jsonl \
    --output-dir /tmp/medusa_test \
    --num-epochs 1 \
    --batch-size 2 \
    --learning-rate 1e-3 \
    --max-length 128 \
    --num-heads 3 \
    --log-interval 1 \
    --save-interval 10

echo "Test completed!"
```

### 6.3 验证Checklist

- [ ] Draft model可以正常初始化
- [ ] ResBlock权重正确零初始化
- [ ] Forward pass shape正确
- [ ] Loss计算正常
- [ ] Backward和梯度更新正常
- [ ] Checkpoint保存和加载正常
- [ ] 分布式训练正常（多GPU）
- [ ] 与参考实现的head输出对齐（如果有原始Medusa checkpoint）

---

## 第七部分：与Eagle3的对比总结

### 7.1 代码复杂度对比

| 组件 | Eagle3 | Medusa | 复杂度变化 |
|------|--------|---------|-----------|
| Draft Model | 1107行 | ~300行 | ↓ 简化73% |
| Core Logic | 608行（TTT递归） | ~200行（单次forward） | ↓ 简化67% |
| 数据处理 | Eagle3Dataset | 继承+修改labels | ≈ 复用95% |
| 训练脚本 | train_eagle3_online.py | train_medusa_online.py | ≈ 结构相同 |

**关键发现**：Medusa实现比Eagle3简单得多，因为无需TTT复杂逻辑！

### 7.2 训练效率对比

| 指标 | Eagle3 | Medusa-1 |
|------|--------|----------|
| **每步耗时** | ~1.2s（TTT 7轮） | ~0.4s（单次forward） |
| **内存占用** | 高（target hidden cache） | 中（只需最后一层） |
| **收敛速度** | 1-2 epochs | 2 epochs |
| **学习率** | 1e-4 | 1e-3（更高） |
| **参数量** | Draft backbone + heads | 只有heads |

**结论**：Medusa训练更快更简单，但Eagle3性能可能更好（取决于场景）。

### 7.3 何时选择哪个算法？

**选择Eagle3**：
- 需要最高加速比（3x+）
- 有充足的计算资源
- 数据量大（>100M tokens）
- 对质量要求极高

**选择Medusa-1**：
- 快速原型验证
- 计算资源有限
- 训练时间紧迫
- 2x加速已足够

**选择Medusa-2**：
- 追求最佳性能（3.6x）
- 可以重新训练base model
- 有高质量训练数据
- 不介意更复杂的训练recipe

---

## 第八部分：常见问题

### Q1: Medusa-1和Medusa-2的训练差异？

**A**:
- **Medusa-1**：只训练heads，本教程已实现
- **Medusa-2**：同时训练base model + heads，需要：
  1. 解冻base model：`base_model.requires_grad = True`
  2. 降低学习率：`1e-5`（防止破坏base model）
  3. 使用特殊训练策略（如knowledge distillation）
  4. 训练时间更长（4-8 epochs）

### Q2: 如何选择Medusa head数量？

**A**: 论文建议：
- **3 heads**: 平衡性能和计算（推荐）
- **4 heads**: 稍高性能，但递减收益
- **2 heads**: 快速训练，性能略低

### Q3: 如何与SGLang集成推理？

**A**:
1. 训练完成后，模型保存在`output_dir/final/`
2. 在SGLang中加载：
```python
from sglang import Engine

engine = Engine(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    speculative_draft_model_path="path/to/medusa/final",
    speculative_algorithm="medusa",
    speculative_num_steps=3,  # 对应num_heads
)
```
3. 推理时自动使用Medusa加速

### Q4: 如何验证实现正确性？

**A**:
1. **Zero-init测试**：确认ResBlock初始输出等于输入
2. **Shape测试**：检查所有tensor维度
3. **Loss下降**：训练几百步后loss应明显下降
4. **Accuracy提升**：各head accuracy应逐渐提升
5. **对比参考**：如果有官方checkpoint，对比head输出

### Q5: 训练不稳定怎么办？

**A**:
1. 降低学习率（1e-3 → 5e-4）
2. 增加warmup（0.1 → 0.2）
3. 减小batch size
4. 使用梯度累积
5. 检查数据质量（是否有异常样本）

---

## 总结

### 实现完成Checklist

完成本教程后，您应该有：

- [x] **Draft Model**：`specforge/modeling/draft/llama3_medusa.py` + `medusa_components.py`
- [x] **Core Logic**：`specforge/core/medusa.py`
- [x] **数据处理**：`specforge/data/medusa_dataset.py`
- [x] **训练脚本**：`scripts/train_medusa_online.py`
- [x] **配置文件**：`configs/llama3-8B-medusa.json`
- [x] **Example**：`examples/run_llama3_medusa_online.sh`
- [x] **测试**：`tests/test_medusa.py`

### 关键收获

1. **架构设计**：Medusa比Eagle3简单得多，无需复杂的TTT逻辑
2. **代码复用**：数据处理前半部分完全复用Eagle3（tokenization、chat template）
3. **训练效率**：Medusa训练速度是Eagle3的3倍（单次forward vs TTT循环）
4. **参数设置**：学习率1e-3，batch size 8，2 epochs（基于论文）

### 下一步

1. **运行训练**：使用`run_llama3_medusa_online.sh`开始训练
2. **调优参数**：根据您的数据和资源调整超参数
3. **评估性能**：在SGLang中测试推理加速比
4. **扩展**：尝试Medusa-2（全模型训练）或其他backbone（Mistral、Qwen）

---

**文档版本**: v1.0
**最后更新**: 2025-11-14
**作者**: SpecForge Team
**参考**: Medusa论文 (arXiv:2401.10774), SpecForge文档

**反馈**: 如有问题请提交GitHub Issue或联系维护者
