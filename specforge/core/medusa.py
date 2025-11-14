# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on Medusa (https://github.com/FasterDecoding/Medusa)
# and Eagle3 training logic, adapted for SpecForge training framework.
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

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.core.loss import LogSoftmaxLoss
from specforge.modeling.draft.llama3_medusa import LlamaForCausalLMMedusa
from specforge.utils import padding


def _compute_target_p_for_medusa(
    target: torch.Tensor,
    t2d: torch.Tensor,
    loss_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute target probabilities for Medusa heads.

    Args:
        target: (batch, seq_len, vocab_size) target logits/probs from target model
        t2d: (vocab_size,) boolean mask for target-to-draft vocab mapping
        loss_mask: (batch, seq_len) mask for valid tokens

    Returns:
        target_p: (batch, seq_len, draft_vocab_size) target probabilities
        position_mask: (batch, seq_len, 1) mask for valid positions
    """
    # target is logits from target model (batch, seq_len, vocab_size)
    target_head = target
    target_max_token = target_head.argmax(-1)  # (batch, seq_len)
    target_mask = t2d[target_max_token]  # (batch, seq_len)
    target_mask = target_mask[..., None].int()  # (batch, seq_len, 1)
    loss_mask = loss_mask[..., None]  # (batch, seq_len, 1)
    position_mask = target_mask * loss_mask  # (batch, seq_len, 1)

    # Filter to draft vocab
    target_head = target_head[..., t2d]  # (batch, seq_len, draft_vocab_size)
    target_head = target_head.float()
    target_p = nn.Softmax(dim=2)(target_head)  # (batch, seq_len, draft_vocab_size)
    target_p = target_p.detach()

    return target_p, position_mask


def _compute_metric_acc(
    logits: torch.Tensor,
    target_p: torch.Tensor,
    position_mask: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute prediction accuracy.

    Args:
        logits: (batch, seq_len, draft_vocab_size)
        target_p: (batch, seq_len, draft_vocab_size)
        position_mask: (batch, seq_len, 1)
        loss_mask: (batch, seq_len)

    Returns:
        accuracy: scalar tensor
    """
    # Get predictions
    pred = logits.argmax(dim=-1)  # (batch, seq_len)
    target_ids = target_p.argmax(dim=-1)  # (batch, seq_len)

    # Compute accuracy only on valid positions
    # position_mask is (batch, seq_len, 1), squeeze to (batch, seq_len)
    # loss_mask is already (batch, seq_len)
    mask = (position_mask.squeeze(-1) * loss_mask).bool()  # (batch, seq_len)
    correct = (pred == target_ids) & mask  # (batch, seq_len)
    accuracy = correct.sum().float() / mask.sum().float().clamp(min=1e-6)

    return accuracy


class MedusaModel(nn.Module):
    """Base class for Medusa models."""

    pass


class OnlineMedusaModel(MedusaModel):
    """
    Online Medusa training model.

    Unlike Eagle3's TTT (Test-Time Training) which unrolls the model multiple times,
    Medusa uses a single forward pass with multiple parallel heads.

    Key differences from Eagle3:
    1. No TTT unrolling (no recursive iterations)
    2. Multiple heads predict the same next token in parallel
    3. All heads are trained simultaneously with the same target
    4. Simpler training loop (no cache management for iterations)

    Training flow:
    1. Extract hidden states from target model (3 aux layers)
    2. Project concatenated hidden states to hidden_size
    3. Add input embeddings
    4. Forward through all Medusa heads in parallel
    5. Compute loss for each head against the same target
    6. Average losses from all heads
    """

    def __init__(
        self,
        draft_model: LlamaForCausalLMMedusa,
        attention_backend="sdpa",
    ):
        """
        Args:
            draft_model: Medusa draft model with multiple heads
            attention_backend: Not used in Medusa (no attention layers)
        """
        super().__init__()
        self.draft_model = draft_model
        self.attention_backend = attention_backend
        self.num_heads = draft_model.get_num_heads()
        # For compatibility with training loop that expects .length attribute
        self.length = self.num_heads

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Online Medusa model trainer.

        Args:
            input_ids: (batch, seq_len) input token IDs
            attention_mask: (batch, seq_len) attention mask (not used in Medusa)
            target: (batch, seq_len) target token IDs
            loss_mask: (batch, seq_len) mask for loss computation
            hidden_states: (batch, seq_len, hidden_size * 3) concatenated hidden states from target model
            past_key_values: Not used in Medusa
            position_ids: Not used in Medusa

        Returns:
            plosses: List of prediction losses (one per head)
            vlosses: Empty list (no value head in Medusa)
            acces: List of accuracies (one per head)
        """
        # Step 1: Compute target probabilities
        # Unlike Eagle3's TTT which shifts targets, all Medusa heads predict the SAME target
        target_p, position_mask = _compute_target_p_for_medusa(
            target=target,
            t2d=self.draft_model.t2d,
            loss_mask=loss_mask,
        )
        # target_p: (batch, seq_len, draft_vocab_size)
        # position_mask: (batch, seq_len, 1)
        del target

        # Step 2: Project hidden states
        # (batch, seq_len, hidden_size * 3) -> (batch, seq_len, hidden_size)
        hidden_states = self.draft_model.project_hidden_states(hidden_states)

        # Step 3: Embed input tokens
        inputs_embeds = self.draft_model.embed_input_ids(input_ids)
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)

        # Step 4: Combine hidden states and embeddings
        combined_hidden = hidden_states + inputs_embeds
        # (batch, seq_len, hidden_size)

        # Step 5: Forward through all Medusa heads
        # Output shape: (num_heads, batch, seq_len, draft_vocab_size)
        medusa_logits = self.draft_model.compute_logits(combined_hidden)

        # Step 6: Compute loss and accuracy for each head
        # All heads use the same target (no shifting like in Eagle3's TTT)
        plosses = []
        acces = []

        for head_idx in range(self.num_heads):
            # Get logits for this head
            head_logits = medusa_logits[head_idx]  # (batch, seq_len, draft_vocab_size)

            # All heads share the same target
            # (batch, seq_len, draft_vocab_size)
            head_target_p = target_p

            # Compute accuracy
            with torch.no_grad():
                acc = _compute_metric_acc(
                    logits=head_logits,
                    target_p=head_target_p,
                    position_mask=position_mask,
                    loss_mask=loss_mask,
                )
                acces.append(acc)

            # Compute loss (in-place modifies logits)
            loss = LogSoftmaxLoss.apply(head_logits, head_target_p, position_mask)
            plosses.append(loss)

        # vlosses is empty for Medusa (no value head)
        vlosses = []

        return plosses, vlosses, acces


class OfflineMedusaModel(OnlineMedusaModel):
    """
    Offline Medusa training model.

    Inherits from OnlineMedusaModel. The only difference is that
    hidden states are pre-computed and loaded from disk instead of
    being generated online from the target model.

    The forward pass is identical to online training.
    """

    pass
