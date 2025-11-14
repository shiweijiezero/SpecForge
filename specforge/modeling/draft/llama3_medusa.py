# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on Medusa (https://github.com/FasterDecoding/Medusa)
# and adapted for SpecForge training framework.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
import torch.nn as nn
from transformers import LlamaConfig

from specforge.modeling.draft.base import Eagle3DraftModel
from specforge.utils import print_with_rank


class ResBlock(nn.Module):
    """
    A Residual Block module for Medusa heads.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class LlamaForCausalLMMedusa(Eagle3DraftModel):
    """
    Medusa draft model for Llama architecture.

    Unlike Eagle3 which uses TTT (Test-Time Training) with a single head,
    Medusa uses multiple independent heads for parallel prediction without recursion.

    Key differences from Eagle3:
    - No Transformer backbone layers (num_hidden_layers = 0)
    - Multiple prediction heads (medusa_num_heads, typically 4)
    - Each head: ResBlock(s) + Linear layer
    - Single forward pass (no TTT unrolling)
    """

    config_class = LlamaConfig

    def __init__(self, config, quant_config=None, attention_backend="sdpa") -> None:
        super().__init__(config)
        self.config = config
        self.quant_config = quant_config

        self.vocab_size = config.vocab_size
        self.draft_vocab_size = config.draft_vocab_size
        self.hidden_size = config.hidden_size

        # Medusa-specific parameters
        self.medusa_num_heads = getattr(config, "medusa_num_heads", 4)
        self.medusa_num_layers = getattr(config, "medusa_num_layers", 1)

        print_with_rank(
            f"Initializing Medusa model with {self.medusa_num_heads} heads, "
            f"{self.medusa_num_layers} ResBlock layers per head"
        )

        # Embedding layer (shared with target model)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )

        # Projection layer to map concatenated hidden states to hidden_size
        if hasattr(config, "target_hidden_size"):
            self.fc = torch.nn.Linear(
                config.target_hidden_size * 3, config.hidden_size, bias=False
            )
        else:
            self.fc = torch.nn.Linear(
                config.hidden_size * 3, config.hidden_size, bias=False
            )

        # Create multiple Medusa heads
        # Each head: [ResBlock(s)] + Linear
        self.medusa_heads = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * self.medusa_num_layers),
                    nn.Linear(self.hidden_size, config.draft_vocab_size, bias=False),
                )
                for _ in range(self.medusa_num_heads)
            ]
        )

        # Create vocab buffers for mapping
        t2d = torch.zeros(self.vocab_size, dtype=torch.bool)
        d2t = torch.zeros(self.draft_vocab_size, dtype=torch.int64)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ttt_length: int = 1,
    ):
        """
        Forward pass for Medusa model - returns combined hidden states.

        Unlike Eagle3, Medusa does NOT use TTT (ttt_length is ignored).
        This method only combines projected hidden states with embeddings.
        Use compute_logits() to get predictions from Medusa heads.

        Arguments:
            hidden_states (`torch.FloatTensor`): concatenated hidden states from target model
                of shape `(batch, seq_len, hidden_size * 3)`
            inputs_embeds (`torch.FloatTensor`): embedded input tokens
                of shape `(batch, seq_len, hidden_size)`
            attention_mask: Not used in Medusa (no Transformer layers)
            ttt_length: Ignored (Medusa does not use TTT)

        Returns:
            hidden_states: Combined hidden states ready for Medusa heads
                of shape `(batch, seq_len, hidden_size)`
        """
        if ttt_length != 1:
            print_with_rank(
                f"Warning: ttt_length={ttt_length} specified but Medusa does not use TTT. "
                "Using single forward pass."
            )

        # Step 1: Project concatenated hidden states
        # (batch, seq_len, hidden_size * 3) -> (batch, seq_len, hidden_size)
        hidden_states = self.fc(hidden_states)

        # Step 2: Add input embeddings (residual connection)
        # (batch, seq_len, hidden_size) + (batch, seq_len, hidden_size)
        hidden_states = hidden_states + inputs_embeds

        # Return combined hidden states
        # (logits are computed separately via compute_logits)
        return hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input token IDs."""
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project concatenated hidden states from target model."""
        return self.fc(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute logits from all Medusa heads.

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            medusa_logits: (medusa_num_heads, batch, seq_len, draft_vocab_size)
        """
        medusa_logits = []
        for head in self.medusa_heads:
            logits = head(hidden_states)
            medusa_logits.append(logits)
        return torch.stack(medusa_logits, dim=0)

    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Medusa backbone: just add embeddings to hidden states.

        Unlike Eagle3, Medusa has no Transformer layers.
        This is a simple residual connection.
        """
        # Simply add input embeddings to hidden states
        # No attention layers in Medusa
        return hidden_states + input_embeds

    def get_num_heads(self) -> int:
        """Get the number of Medusa heads."""
        return self.medusa_num_heads

    def get_medusa_head(self, index: int) -> nn.Module:
        """Get a specific Medusa head by index."""
        return self.medusa_heads[index]
