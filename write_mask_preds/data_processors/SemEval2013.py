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
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

MAX_LENGTH = 512

class SemEval2013Dataset(Dataset):
    def __init__(
        self,
        args,
        input_file: str,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        self.processor = SemEval2013Processor()
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

            examples = list(self.processor.get_examples(args.data_dir, input_file))
            if limit_length is not None:
                examples = examples[:limit_length]
            self.features, instance_id_to_doc_id = convert_examples_to_features(
                examples,
                tokenizer,
                max_length=args.max_seq_length,
                padding_strategy="max_length" if args.simple_sampler else "do_not_pad"
            )
            start = time.time()
            torch.save(self.features, cached_features_file)
            json.dump(instance_id_to_doc_id, open(os.path.join(args.out_dir, "instance_id_to_doc_id.json"), 'w'))
            logger.info(
                "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
            )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, x) -> InputFeatures:
        if isinstance(x, list):
            return [self.features[i] for i in x]
        return self.features[x]

class SemEval2013Processor(DataProcessor):

    def get_examples(self, data_dir, input_file):
        instance_ids_in_gold = self.peek_gold(data_dir)

        xml_path = os.path.join(data_dir, input_file)
        with open(xml_path, encoding="utf-8") as xml_file:
            et_xml = ElementTree.parse(xml_file)
            for word in et_xml.getroot():
                for inst in word.getchildren():
                    inst_id = inst.attrib['id']
                    if inst_id in instance_ids_in_gold:
                        context = inst.find("context")
                        before, target, after = list(context.itertext())
                        text = ''.join([before, target, after]).replace('  ', ' ')
                        yield InputExample(guid=inst_id, text_a=text)

    @staticmethod
    def peek_gold(data_dir):
        gold_path = os.path.join(data_dir, '/home/matane/matan/dev/SemEval/resources/SemEval-2013-Task-13-test-data/keys/gold/all.key')
        with open(gold_path, encoding='utf-8') as gold_file:
            instance_ids_in_gold = set()
            for line in gold_file:
                inst_id = line.strip().split(maxsplit=2)[1]
                instance_ids_in_gold.add(inst_id)
        return instance_ids_in_gold

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
    instance_id_to_doc_id = {}
    for i, example in enumerate(examples):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, guid=i)
        features.append(feature)
        instance_id_to_doc_id[example.guid] = i

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features, instance_id_to_doc_id
