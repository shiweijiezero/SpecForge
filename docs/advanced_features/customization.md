# 💡 自定义您的训练

### 🔧 自定义训练参数

```bash
torchrun \
    --standalone \
    --nproc_per_node 8 \
    ./scripts/train_eagle3_online.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config ./configs/llama3-8B-eagle3.json \
    --train-data-path ./cache/dataset/sharegpt.jsonl \
    --output-dir ./outputs/llama3-8b-eagle3 \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template llama3 \
    --cache-dir ./cache
```

如果您想了解每个参数的作用,可以运行 `python scripts/train_eagle3_online.py --help` 查看完整的参数列表。特别地,我们将在下面讨论一些重要的参数。
- `--chat-template`: 这应该是用于模型的对话模板,因此请确保将其设置为正确的值。
- `--cache-dir`: 该目录包含数据集缓存,包括 `input_ids`、`loss_mask`、`attention_mask` 和 `vocab_mapping`。一旦生成缓存,这些缓存可以使您的数据加载速度大大加快。缓存文件的名称是通过对数据集路径进行哈希得到的,以避免缓存冲突。

### 💬 自定义对话模板

您可以通过在 `specforge.data.template.py` 文件中向 `TEMPLATE_REGISTRY` 添加新条目来为您的模型注册新的对话模板。

```python
TEMPLATE_REGISTRY.register(
    name="your-template-name",
    template=ChatTemplate(
        assistant_header="xxx",
        user_header="xxx",
        system_prompt="xxx",
        end_of_turn_token="xxx",
    ),
)
```

### 🪅 自定义模型

#### 自定义目标模型

如果您希望为其他模型训练 Eagle3,需要修改 `--target-model-path` 的值。我们支持直接从 HuggingFace 加载这些模型。

但是,如果您的模型过大并且需要张量并行,您可以在 `specforge.modeling.target` 目录中自行实现其张量并行版本。CausalLM 模型应该继承 `specforge.modeling.target.base.py` 文件中的 `DistributedTargetModel` 类,并将 `ColumnParallelLinear` 和 `RowParallelLinear` 应用于其子模块。

```python
from .base import DistributedTargetModel
from specforge.layers.linear import ColumnParallelLinear, RowParallelLinear


class MyModelForCausalLM(MyModelPreTrainedModel, GenerationMixin, DistributedTargetModel):
    ...

    def load_weights(self, state_dict: Dict[str, torch.Tensor]):
        ...
```

之后,您需要在 `specforge.modeling.auto.py` 文件中将此模型注册到 `AutoEagle3TargetModel` 类。

```diff
class AutoDistributedTargetModel(AutoModelForCausalLMBase):
    _model_mapping = {
        Llama4TextConfig: [Llama4ForCausalLM],
+       MyModelConfig: [MyModelForCausalLM],
    }
```

当 `tp_size` 大于 1 时,脚本将自动加载模型的分布式版本以进行张量并行。

#### 自定义草稿模型

如果您想更改草稿模型配置,可以编写自己的配置文件并将其路径传递给 `--draft-model-config` 参数。或者,如果您不提供 `--draft-model-config` 参数,脚本将根据目标模型配置自动生成草稿模型配置。如果您希望使用 SGLang 为您的自定义草稿模型提供服务,请确保您也在 SGLang 中实现草稿模型,并且架构名称必须匹配。要实现您自己的草稿模型,您可以创建一个新类并从 `specforge.modeling.draft.base.py` 文件中的 `Eagle3DraftModel` 类继承它。


```python
from .base import Eagle3DraftModel
from transformers import PretrainedConfig


class MyModelConfig(PretrainedConfig):
    model_type = "mymodel"

    def __init__(self, **kwargs):
        ...


class MyModelEagle3(Eagle3DraftModel):

    config_class = MyModelConfig

    def __init__(self, config, quant_config=None) -> None:
        ...
```

然后,您可以在 `specforge.modeling.auto.py` 文件中将这些模型注册到 `AutoEagle3TargetModel` 和 `AutoDraftModelConfig` 类以实现自动模型加载。

```diff
class AutoEagle3DraftModel(AutoModelForCausalLMBase):
    # the model mapping is currently hardcoded, we should support lazy model mapping via registry
    _model_mapping = {
        LlamaConfig: [LlamaForCausalLMEagle3],
+       MyModelConfig: MyModelEagle3,
    }


class AutoDraftModelConfig:

    _config_mapping = {
        "LlamaForCausalLMEagle3": LlamaConfig,
+       "MyModelEagle3": MyModelConfig,
    }
```

这样,只要您的 `config.json` 指定了正确的架构名称,脚本就会自动为您加载正确的草稿模型。
