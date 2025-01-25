# AUTO-GENERATED (DO NOT MODIFY)
# NET IDS: BDG83

import logging
from dataclasses import dataclass
from typing import Union, Optional, List, Dict, Any

import numpy as np
import torch

from ner.data_processing.constants import NER_ENCODING_MAP, PAD_NER_TAG
from ner.data_processing.tokenizer import Tokenizer


class DataCollator(object):
    def __init__(
        self,
        tokenizer: Tokenizer,
        padding: Union[str, bool] = "longest",
        max_length: Optional[int] = None,
        padding_side: str = "right",
        truncation_side: str = "right",
        pad_tag: str = PAD_NER_TAG,
        text_colname: str = "text",
        label_colname: str = "NER",
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.padding_side = padding_side
        self.truncation_side = truncation_side
        self.pad_tag = pad_tag
        self.text_colname = text_colname
        self.label_colname = label_colname

    def _get_max_length(self, data_instances: List[Dict[str, Any]]) -> Optional[int]:
        if not ((self.padding == "longest" or self.padding) and self.max_length is None):
            logging.warning(
                f"both max_length={self.max_length} and padding={self.padding} provided; ignoring "
                f"padding={self.padding} and using max_length={self.max_length}"
            )
            self.padding = "max_length"

        if self.padding == "longest" or (isinstance(self.padding, bool) and self.padding):
            return max([len(data_instance[self.text_colname]) for data_instance in data_instances])
        elif self.padding == "max_length":
            return self.max_length
        elif isinstance(self.padding, bool) and not self.padding:
            return None
        raise ValueError(f"padding strategy {self.padding} is invalid")

    @staticmethod
    def _process_labels(labels: List) -> torch.Tensor:
        return torch.LongTensor([NER_ENCODING_MAP[label] for label in labels])

    def __call__(self, data_instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.data_processing.data_collator.html.
        """

        batch_size = len(data_instances)
        batch_max_length = self._get_max_length(data_instances)

        batch_input_ids = torch.empty((batch_size, batch_max_length), dtype=torch.long)
        batch_padding_mask = torch.empty((batch_size, batch_max_length), dtype=torch.long)
        batch_labels = torch.empty((batch_size, batch_max_length), dtype=torch.long)

        for i in range(len(data_instances)):
          instance = data_instances[i]

          # Get the text from the instance dict
          input_seq = instance[self.text_colname]

          # Tokenize the input_sequence
          tokenized_dict = self.tokenizer.tokenize(input_seq, batch_max_length, self.padding_side, self.truncation_side)

          instance_input_ids = tokenized_dict["input_ids"]
          instance_padding_mask = tokenized_dict["padding_mask"]

          # Assign to the batch matrices
          batch_input_ids[i][:] = instance_input_ids
          batch_padding_mask[i][:] = instance_padding_mask

          # If this is NOT running on test data
          if instance.get(self.label_colname, False):
            
            # Get instance labels 
            instance_labels = instance[self.label_colname]

            # Calculate padding/truncation
            n_pad = len(instance_padding_mask) - len(instance_labels)

            # Label Padding Needed
            if n_pad > 0:
              label_padding = [self.pad_tag]*n_pad

              # Add paddding to the left
              if self.padding_side == "left":
                instance_labels = label_padding + instance_labels

              # Add padding to the right
              else:
                instance_labels = instance_labels + label_padding

            # Label Truncation Needed
            elif n_pad < 0:

              # Truncate Left
              if self.truncation_side == "left":
                instance_labels = instance_labels[(n_pad*-1):]

              # Truncate Right
              else:
                instance_labels = instance_labels[:n_pad]
            
            # Assign to the match matrix
            labels_tensor = self._process_labels(instance_labels)

            batch_labels[i][:] = labels_tensor


        out_dict = {
          "input_ids": batch_input_ids,
          "padding_mask": batch_padding_mask,
          "labels": batch_labels
        }

        return out_dict
