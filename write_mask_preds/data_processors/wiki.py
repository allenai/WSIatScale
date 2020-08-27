import json
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

class WikiDataset(Dataset):
    def __init__(
        self,
        args,
        input_file: str,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        self.processor = WikiProcessor()
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                tokenizer.__class__.__name__, str(args.max_seq_length), str(args.simple_sampler), input_file
            ),
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            start = time.time()
            self.features = torch.load(cached_features_file)
            logger.info(
                f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
            )
        else:
            logger.info(f"Creating features from dataset file at {args.data_dir}")

            examples = self.processor.get_examples(args.data_dir, input_file)
            if limit_length is not None:
                examples = examples[:limit_length]
            self.features = convert_examples_to_features(
                examples,
                tokenizer,
                max_length=args.max_seq_length,
                padding_strategy="max_length" if args.simple_sampler else "do_not_pad"
            )
            start = time.time()
            torch.save(self.features, cached_features_file)
            logger.info(
                "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
            )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, x) -> InputFeatures:
        if isinstance(x, list):
            return [self.features[i] for i in x]
        return self.features[x]

class WikiProcessor(DataProcessor):
    def get_examples(self, data_dir, input_file):
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, input_file)))

    def _create_examples(self, lines):
        examples = []
        for line in lines:
            text = line['metadata']['paragraph_']
            guid, _ = line['id'].split(':')
            examples.append(InputExample(guid=guid, text_a=text))
        return examples

    @classmethod
    def _read_jsonl(cls, path):
        with open(path, 'r') as file:
            for line in file:
                yield json.loads(line)

def convert_examples_to_features(
        examples: List[InputExample],
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
    for inputs, guid in merge_encodings(batch_encoding, examples):
        feature = InputFeatures(**inputs, guid=guid)
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features

def merge_encodings(encoding, examples):
    lengths = [len(x) for x in encoding['input_ids']]

    batch_length = 0
    concat_encoding = None
    for i, length in enumerate(lengths):
        end_of_batch = False
        batch_length += length
        curr_guid = examples[i].guid
        if batch_length > MAX_LENGTH:
            end_of_batch = True
        else:
            if len(lengths) == i+1 or \
                curr_guid != examples[i+1].guid or \
                (batch_length + lengths[i+1]) > MAX_LENGTH:
                end_of_batch = True

        if concat_encoding is None:
            concat_encoding = {'input_ids': encoding['input_ids'][i],
                               'attention_mask': encoding['attention_mask'][i]}
        else:
            concat_encoding = {k: concat_encoding[k] + encoding[k][i] for k in concat_encoding}

        if end_of_batch:
            yield concat_encoding, int(curr_guid)
            batch_length = 0
            concat_encoding = None