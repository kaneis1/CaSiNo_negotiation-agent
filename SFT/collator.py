#!/usr/bin/env python3
"""
SFT/collator.py — Response-only data collator for Llama-3.1 Instruct SFT.

Replaces trl's removed DataCollatorForCompletionOnlyLM.

Strategy:
  1. Tokenize a batch of pre-formatted chat-template strings.
  2. Initialize labels = -100 (masked) for the entire sequence.
  3. For every occurrence of the assistant header token IDs, unmask
     tokens from after the header up to and including the next <|eot_id|>.

This means loss is computed only on the model's own response tokens.
"""

from __future__ import annotations

from typing import Dict, List

import torch


# Llama-3.1 Instruct assistant turn header
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"


class ResponseOnlyCollator:
    """
    Data collator that masks all non-assistant tokens with -100.

    Args:
        tokenizer:         The Llama tokenizer (must have padding configured).
        response_template: The exact string marking the start of each
                           assistant response. Defaults to the Llama-3.1
                           assistant header.
        max_length:        Truncation length. Should match SFTConfig.max_length.
    """

    def __init__(
        self,
        tokenizer,
        response_template: str = ASSISTANT_HEADER,
        max_length: int = 2048,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Encode the response template WITHOUT a leading BOS token so we
        # can search for it as a subsequence inside the full tokenized text.
        self.template_ids: List[int] = tokenizer.encode(
            response_template, add_special_tokens=False
        )
        self.eot_id: int = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # SFTTrainer pre-tokenizes the dataset, so features arrive as dicts
        # with "input_ids" and optionally "attention_mask".
        # Pad the batch manually, then apply response-only masking.

        input_ids_list = [torch.tensor(f["input_ids"]) for f in features]

        # Pad to the longest sequence in this batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        # Build attention mask: 1 for real tokens, 0 for padding
        if "attention_mask" in features[0]:
            attn_mask_list = [torch.tensor(f["attention_mask"]) for f in features]
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                attn_mask_list, batch_first=True, padding_value=0
            )
        else:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        labels: torch.Tensor = torch.full_like(input_ids, -100)
        tlen = len(self.template_ids)

        for i, ids in enumerate(input_ids.tolist()):
            for j in range(len(ids) - tlen + 1):
                if ids[j : j + tlen] == self.template_ids:
                    response_start = j + tlen
                    response_end = len(ids)
                    for k in range(response_start, len(ids)):
                        if ids[k] == self.eot_id:
                            response_end = k + 1  # include eot_id in loss
                            break
                    labels[i, response_start:response_end] = input_ids[
                        i, response_start:response_end
                    ]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
