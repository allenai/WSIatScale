import csv
import json
from dataclasses import dataclass
import logging
import os
import time
from typing import Optional, List

import torch
from torch.utils.data.dataset import Dataset

from transformers.data.processors.utils import DataProcessor, InputExample
from transformers.tokenization_utils import PreTrainedTokenizer

from data_processors.data_processor import InputFeatures # pylint: disable=import-error

logger = logging.getLogger(__name__)

MAX_LENGTH = 512

class WiC_TSVDataset(Dataset):
    def __init__(
        self,
        args,
        input_file: str,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        self.processor = WiC_TSVProcessor()

        logger.info(f"Creating features from dataset file at {args.data_dir}")

        examples = list(self.processor.get_examples(args.data_dir, input_file, tokenizer))
        if limit_length is not None:
            examples = examples[:limit_length]
        self.features, instance_id_to_target_pos = convert_examples_to_features(
            examples,
            tokenizer,
            max_length=args.max_seq_length,
            padding_strategy="max_length" if args.simple_sampler else "do_not_pad"
        )
        json.dump(instance_id_to_target_pos, open(os.path.join(args.out_dir, "instance_id_to_target_pos.json"), 'w'))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, x) -> InputFeatures:
        if isinstance(x, list):
            return [self.features[i] for i in x]
        return self.features[x]

@dataclass
class WiC_TSV_InputExample(InputExample):
    """
    position of the target word in the sentence.
    """
    target_pos: Optional[int] = -1

class WiC_TSVProcessor(DataProcessor):
    def get_examples(self, data_dir, input_file, tokenizer):
        return self._create_examples(tokenizer, self._read_csv(os.path.join(data_dir, input_file)))

    def _create_examples(self, tokenizer, lines):
        examples = []
        for i, line in enumerate(lines):
            _, pos, text = line
            pos = int(pos)
            splited_text = text.split()
            # assert splited_text[pos] == target_word #This is not the case for plurals
            before_target = ' '.join(splited_text[:pos])
            # len_target_word = len(tokenizer.encode(target_word, add_special_tokens=False))
            # target_position = [x + (len(tokenizer.encode(before_target)) - 1) for x in range(len_target_word)]
            target_position = len(tokenizer.encode(before_target)) - 1
            examples.append(WiC_TSV_InputExample(guid=i, text_a=text, target_pos=target_position))
        return examples

    @classmethod
    def _read_csv(cls, path):
        with open(path, 'r') as f:
            for row in f:
                row = row.split('\t')
                row[2] = row[2].strip()
                yield row

def convert_examples_to_features(
        examples: List[WiC_TSV_InputExample],
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
        padding_strategy: str = "max_length",
):
    if max_length is None or max_length == -1:
        max_length = tokenizer.max_len

    batch_encoding = tokenizer(
        [(example.text_a) for example in examples],
        max_length=max_length,
        padding=padding_strategy,
        truncation=True,
        add_special_tokens=True
    )

    features = []
    instance_id_to_target_pos = {}
    for i, example in enumerate(examples):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, guid=i)
        features.append(feature)
        instance_id_to_target_pos[example.guid] = example.target_pos

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features, instance_id_to_target_pos
