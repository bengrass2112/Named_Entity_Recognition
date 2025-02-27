# AUTO-GENERATED (DO NOT MODIFY)
# NET IDS: BDG83

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from itertools import chain
from typing import List, Dict, Optional, Union

import datasets
import torch

from ner.data_processing.constants import PAD_TOKEN, UNK_TOKEN


class Tokenizer(object):
    def __init__(
        self,
        pad_token: str = PAD_TOKEN,
        unk_token: str = UNK_TOKEN,
        lowercase: bool = False,
    ) -> None:
        super().__init__()

        self.pad_token = pad_token
        self.unk_token = unk_token

        if lowercase:
            logging.warning(f"lowercase set to {lowercase}, which could impact named-entity recognition")
        self.lowercase = lowercase

        self.token2id = {pad_token: 0, unk_token: 1}

    @property
    def id2token(self) -> Dict[int, str]:
        return {token_id: token for token, token_id in self.token2id.items()}

    @property
    def vocab(self) -> List[str]:
        return list(self.token2id.keys())

    @property
    def vocab_size(self) -> int:
        return len(self.token2id.keys())

    def extend(self, tokens: List) -> None:
        existing_vocab_size = self.vocab_size
        if self.lowercase:
            tokens = [token.lower() for token in tokens]

        for token_id, token in enumerate(tokens):
            if token not in self.token2id:
                self.token2id.update({token: existing_vocab_size + token_id})

    def reset(self) -> None:
        self.token2id = {self.pad_token: 0, self.unk_token: 1}

    def __str__(self) -> str:
        return (
            f"Tokenizer(vocab_size={self.vocab_size}, "
            f"pad_token={self.pad_token}, "
            f"unk_token={self.unk_token}, "
            f"lowercase={self.lowercase})"
        )

    def train(
        self,
        train_data: datasets.Dataset,
        text_colname: str = "text",
        min_freq: int = 2,
        remove_frac: float = 0.3,
        reset: bool = True,
    ) -> None:
        if reset:
            self.reset()
        existing_vocab_size = self.vocab_size

        text_data = chain(*train_data[text_colname])
        if self.lowercase:
            text_data = [token.lower() for token in text_data]
        token_freqs = Counter(text_data)

        valid_tokens = [token for token, freq in token_freqs.items() if freq >= min_freq]
        logging.info(f"num of unique tokens retained after min freq of {min_freq} filtering: {len(valid_tokens)}")
        top_tokens = sorted(valid_tokens, key=lambda token: token_freqs[token], reverse=True)
        top_tokens = top_tokens[: len(top_tokens) - int(remove_frac * len(top_tokens))]
        logging.info(f"num of unique tokens retained after {remove_frac} fraction of tokens removed: {len(top_tokens)}")

        self.token2id.update({token: existing_vocab_size + token_id for token_id, token in enumerate(top_tokens)})

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as fp:
            json.dump(self.token2id, fp)

    def from_file(self, filepath: str) -> None:
        with open(filepath, "r") as fp:
            self.token2id = json.load(fp)

    def from_dict(self, token2id_dict: Dict[str, int]) -> None:
        self.token2id = token2id_dict

    def tokenize(
        self,
        input_seq: Union[List[str], str],
        max_length: Optional[int] = None,
        padding_side: str = "right",
        truncation_side: str = "right",
    ) -> Dict[str, torch.Tensor]:
        """Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.data_processing.tokenizer.html."""

        # If the input sequence is a string convert it to a list
        if type(input_seq) == str:
          input_seq = input_seq.split(" ")

        # If max_length is none change it to the lenght of the sequence
        if max_length == None:
          max_length = len(input_seq)

        input_ids = [self.token2id.get(token, self.token2id[self.unk_token]) for token in input_seq]
        padding_mask = [0]*len(input_ids)

        # Calculate padding/truncation
        n_pad = max_length - len(input_ids)

        # Padding is needed 
        if n_pad > 0:
          padding = [self.token2id[self.pad_token]]*n_pad
          p_mask = [1]*n_pad

          # Add padding to the left
          if padding_side == "left":
            input_ids = padding + input_ids

            # Update padding mask to reflect the left pad
            padding_mask = p_mask + padding_mask

          # Add padding to the right
          else:
            input_ids = input_ids + padding

            # Update padding mask to reflect the right pad
            padding_mask = padding_mask + p_mask

        # Truncation is needed (n_pad is negative)
        elif n_pad < 0:

          # Truncate Left
          if truncation_side == "left":
            input_ids = input_ids[(n_pad*-1):]

          # Truncate Right
          else:
            input_ids = input_ids[:n_pad]

        out_dict = {
          "input_ids": torch.LongTensor(input_ids),
          "padding_mask":torch.LongTensor(padding_mask)
        }

        return out_dict

    def decode(self, input_ids: torch.Tensor, return_as_list=False) -> Union[List[str], str]:
        if return_as_list:
            return [self.id2token[input_id] for input_id in input_ids.numpy()]
        return " ".join([self.id2token[input_id] for input_id in input_ids.numpy()])
