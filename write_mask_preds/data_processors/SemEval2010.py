from dataclasses import dataclass
import json
import logging
import os
import time
import spacy
from typing import Optional, List

import torch
from torch.utils.data.dataset import Dataset

from transformers.data.processors.utils import DataProcessor, InputExample
from transformers.tokenization_utils import PreTrainedTokenizer

from data_processors.data_processor import InputFeatures # pylint: disable=import-error
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

MAX_LENGTH = 512

class SemEval2010Dataset(Dataset):
    def __init__(
        self,
        args,
        input_file: str,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        assert input_file == "SemEval2010" # Not using input_file
        self.processor = SemEval2010Processor()
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

            examples = list(self.processor.get_examples(args.data_dir, tokenizer))
            if limit_length is not None:
                examples = examples[:limit_length]
            self.features, instance_id_to_doc_id, instance_id_to_target_pos = convert_examples_to_features(
                examples,
                tokenizer,
                max_length=args.max_seq_length,
                padding_strategy="max_length" if args.simple_sampler else "do_not_pad"
            )
            start = time.time()
            torch.save(self.features, cached_features_file)
            json.dump(instance_id_to_doc_id, open(os.path.join(args.out_dir, "instance_id_to_doc_id.json"), 'w'))
            json.dump(instance_id_to_target_pos, open(os.path.join(args.out_dir, "instance_id_to_target_pos.json"), 'w'))
            logger.info(
                "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
            )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, x) -> InputFeatures:
        if isinstance(x, list):
            return [self.features[i] for i in x]
        return self.features[x]

@dataclass
class SemEval2010InputExample(InputExample):
    """
    position of the target word in the sentence.
    """
    local_pos: Optional[int] = -1

class SemEval2010Processor(DataProcessor):
    def get_examples(self, data_dir, tokenizer):
        additional_mapping = {'stuck': 'stick', 'swam': 'swim', 'lain': 'lie', 'swore': 'swear', 'lie': 'lay', 'lay': 'lie',
        'commissions': 'commission', 'shaving': 'shave', 'observing': 'observe', 'swimming': 'swim', 'separating': 'separate', 'questioning': 'question',  'waiting': 'wait', 'happening': 'happen', 'rooting': 'root', 'sniffing': 'sniff', 'laying': 'lay', 'straightened': 'straighten', 'account': 'accounting',
        'committed': 'commit', 'regained': 'regain',
        'figgere': 'figure', 'figger': 'figure', 'lah': 'lie'} # Last row are weird ones

        nlp = spacy.load('en', disable=['ner', 'parser'])
        for root_dir, _, files in os.walk(data_dir):
            for file in files:
                if '.xml' not in file: continue
                tree = ElementTree.parse(os.path.join(root_dir, file))
                root = tree.getroot()
                for child in root:
                    inst_id = child.tag
                    lemma = inst_id.split('.')[0]
                    target_sent = child[0].text

                    parsed = nlp(target_sent)
                    first_occur_idx = None
                    for idx, w in enumerate(parsed):
                        token_lemma = w.lemma_.lower()
                        if token_lemma == lemma or additional_mapping.get(token_lemma) == lemma:
                            first_occur_idx = idx
                            break
                    if first_occur_idx is None:
                        print(file, [x.lemma_ for x in parsed], target_sent)
                        print('could not find the correct lemma -probably spacy\'s lemmatizer had changed. '
                              'add the lemma from here to additional_mapping:')
                        # e.g. if you see lie.v was broken and in the list of lemmas you find 'lain'
                        # - add a mapping from 'lain' -> 'lie' in additional_mapping map above
                        raise Exception('Could not pin-point lemma in SemEval sentence')

                    pre = self.format_text(''.join(parsed[i].string for i in range(first_occur_idx)))
                    target = self.format_target(tokenizer, parsed[first_occur_idx].text, lemma)
                    post = self.format_text(''.join(parsed[i].string for i in range(first_occur_idx + 1, len(parsed))))
                    
                    target_position = len(tokenizer.encode(pre)) - 1
                    text = f"{pre}{target} {post}"
                    # assert tokenizer.encode(text)[target_position] == tokenizer.encode(target, add_special_tokens=False)[0]

                    yield SemEval2010InputExample(guid=inst_id, text_a=text, local_pos=target_position)

    @staticmethod
    def format_target(tokenizer, target, lemma):
        if len(tokenizer.encode(target, add_special_tokens=False)) > 1:
            # First two conditions don't have any single wordpiece except for these:
            if lemma == 'cultivate':
                return 'cultivated'
            elif lemma == 'presume':
                return 'presumed'
            # Not a single close wordpiece to reap
            elif lemma == 'reap': 
                return '[MASK]'
            elif len(tokenizer.encode(target.lower(), add_special_tokens=False)) == 1:
                return target.lower()
            else:
                return lemma
            if len(tokenizer.encode(lemma, add_special_tokens=False)) != 1:
                return '[MASK]' #Don't have anything better to do here.
        else:
            return target


    @staticmethod
    def format_text(text):
        text = text.replace(" 's ", "'s ")
        text = text.replace(" , ", ", ")
        text = text.replace(" . ", ". ")
        text = text.replace(" % ", "% ")
        text = text.replace(" $ ", " $")
        text = text.replace(" n't ", "n't ")
        text = text.replace("-LRB-", "(")
        text = text.replace("-RRB-", ")")
        text = text.replace("   ", " ")
        text = text.replace("  ", " ")

        return text

def convert_examples_to_features(
        examples: List[SemEval2010InputExample],
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
    instance_id_to_doc_id, instance_id_to_target_pos = {}, {}
    for i, example in enumerate(examples):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, guid=i)
        features.append(feature)
        instance_id_to_doc_id[example.guid] = i
        instance_id_to_target_pos[example.guid] = example.local_pos

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features, instance_id_to_doc_id, instance_id_to_target_pos