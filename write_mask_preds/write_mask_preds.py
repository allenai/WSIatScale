import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from transformers.data.data_collator import default_data_collator
from transformers import AutoTokenizer, BertForMaskedLM, RobertaForMaskedLM

from adaptive_sampler import MaxTokensBatchSampler, data_collator_for_adaptive_sampler
from data_processors import CORDDataset, WikiDataset # pylint: disable=import-error

TOP_N_WORDS = 100
PAD_ID = 0
dataset_params = {'cord': {'dataset_class': CORDDataset,
                           'model_class': BertForMaskedLM,
                           'model_hg_path': 'allenai/scibert_scivocab_uncased',},
                  'wiki': {'dataset_class': WikiDataset,
                           'model_class': RobertaForMaskedLM,
                           'model_hg_path': 'roberta-large',},
                 }

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    tokenizer, model = initialize_models(device, args)
    dataset_class = dataset_params[args.dataset]['dataset_class']

    i = 0
    similar_files = sorted([f for f in os.listdir(args.data_dir) if starts_with_and_in_range(f, args.starts_with, args.files_range)])
    for input_file in tqdm(similar_files):
        dataset = dataset_class(args, input_file, tokenizer, cache_dir='/tmp/')
        dataloader = simple_dataloader(args, dataset) if args.simple_sampler else adaptive_dataloader(args, dataset)

        for inputs in tqdm(dataloader):
            with torch.no_grad():
                dict_to_device(inputs, device)
                doc_ids = inputs.pop('guid')
                last_hidden_states = model(**inputs)[0]
                normalized = last_hidden_states.softmax(-1)
                probs, indices = normalized.topk(TOP_N_WORDS)

                write_replacements_to_file(f"{args.out_dir}/{input_file[:-6]}-{i}.npz", doc_ids, inputs, indices, probs)
            i += 1

def starts_with_and_in_range(f, starts_with, files_range):
    ret = f.startswith(starts_with)
    if files_range is not None:
        min_id, max_id = files_range.split('-')
        id = int(''.join(x for x in f if x.isdigit()))
        ret = ret and id >= int(min_id) and id <= int(max_id)
    return ret

def initialize_models(device, args):
    model_hg_path = dataset_params[args.dataset]['model_hg_path']
    model_class = dataset_params[args.dataset]['model_class']
    tokenizer = AutoTokenizer.from_pretrained(model_hg_path, use_fast=True)
    model = model_class.from_pretrained(model_hg_path)
    model.to(device)
    if args.fp16:
        from apex import amp # pylint: disable=import-error
        model = amp.initialize(model, opt_level="O2")
        
    assert tokenizer.vocab_size < 65535 # Saving pred_ids as np.uint16
    return tokenizer, model

def adaptive_dataloader(args, dataset):
    sampler = MaxTokensBatchSampler(dataset, max_tokens=args.max_tokens_per_batch, padding_noise=0.0)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=data_collator_for_adaptive_sampler,
    )
    return dataloader

def simple_dataloader(args, dataset):
    sampler = (
        RandomSampler(dataset)
        if args.local_rank == -1
        else DistributedSampler(dataset)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=default_data_collator,
    )
    return dataloader

def write_replacements_to_file(outfile, doc_ids, inputs, replacements, probs):
    b, l = inputs['input_ids'].size()
    attention_mask = inputs['attention_mask'].bool()

    doc_ids = doc_ids.cpu().numpy().astype(np.int32)
    sent_lengths = inputs['attention_mask'].sum(1).cpu().numpy().astype(np.int16)
    tokens = inputs['input_ids'].masked_select(attention_mask).cpu().numpy().astype(np.uint16)
    replacements = replacements.masked_select(attention_mask.unsqueeze(2)).view(-1, TOP_N_WORDS).cpu().numpy().astype(np.uint16)
    probs = probs.masked_select(attention_mask.unsqueeze(2)).view(-1, TOP_N_WORDS).cpu().numpy().astype(np.float16)

    np.savez(outfile,
        doc_ids=doc_ids,
        sent_lengths=sent_lengths,
        tokens=tokens,
        replacements=replacements,
        probs=probs,
    )

def dict_to_device(inputs, device):
    if device.type == 'cpu': return
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--starts_with", type=str, required=True)
    parser.add_argument("--files_range", type=str, help="should be splited with `-`")
    parser.add_argument("--out_dir", type=str, default="replacements")
    parser.add_argument("--dataset", type=str, required=True, choices=['cord', 'wiki'])
    parser.add_argument("--local_rank", type=int, default=-1, help="Not Maintained")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_tokens_per_batch", type=int, default=-1)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--simple_sampler", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()
    if args.simple_sampler:
        assert args.max_tokens_per_batch == -1 and \
            args.batch_size > 1 and \
            args.max_seq_length > -1, \
            "Expecting arguments for simple sampler"
    else:
        assert args.max_tokens_per_batch > 0 and \
            args.batch_size == 1 and \
            "Expecting arguments for adaptive sampler"

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    main(args)
