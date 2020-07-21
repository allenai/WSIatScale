import argparse
import csv
import os
from time import time
import torch
from tqdm import tqdm

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from transformers.data.data_collator import default_data_collator
from transformers import AutoTokenizer, BertForMaskedLM

from adaptive_sampler import MaxTokensBatchSampler, data_collator_for_adaptive_sampler
from data_processor import CORDDataset

TOP_N_WORDS = 100
PAD_ID = 0

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = BertForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased')
    model.to(device)
    dataset = CORDDataset(args, tokenizer)
    dataloader = adaptive_dataloader(args, dataset) if args.adaptive_sampler else simple_dataloader(args, dataset)
    writer = csv.writer(open(args.alternatives_file, 'w'), delimiter='\t')

    before = time()
    total_num_of_tokens = 0
    for inputs in tqdm(dataloader):
        total_num_of_tokens += inputs['input_ids'].numel()
        guids = inputs.pop('guid')
        with torch.no_grad():
            dict_to_device(inputs, device)
            last_hidden_states = model(**inputs)[0]
            normalized = last_hidden_states.softmax(-1)
            probs, indices = normalized.topk(TOP_N_WORDS)

            # Benchmarking two options here
            if args.iterativly_go_over_matrices:
                write_alternatives_to_file_loop_iteratively(guids, inputs, indices, probs, writer)
            else:
                write_alternatives_to_file(guids, inputs, indices, probs, writer)

    log_times(args, time() - before, total_num_of_tokens)

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

def write_alternatives_to_file(guids, inputs, indices, probs, writer):
    """More Pytorch like"""
    # TODO, not using guids yet
    attention_mask = inputs['attention_mask'].bool()
    unmasked_tokens = inputs['input_ids'].masked_select(attention_mask)
    unmasked_indices = indices.masked_select(attention_mask.unsqueeze(2)).view(-1, TOP_N_WORDS)
    unmasked_probs = probs.masked_select(attention_mask.unsqueeze(2)).view(-1, TOP_N_WORDS)
    for in_word, subs, sub_probs in zip(unmasked_tokens.tolist(), unmasked_indices.tolist(), unmasked_probs.tolist()):
        short_probs = list(map(lambda x: round(x, 4), sub_probs))
        writer.writerow([in_word, subs, short_probs])

def write_alternatives_to_file_loop_iteratively(guids, inputs, indices, probs, writer):
    """This style is not very pytorch like,
    This might be quicker when masks are mostly 1
    """
    for sent_id, sent, sent_subs, sent_probs in zip(guids, inputs['input_ids'].tolist(), indices.tolist(), probs.tolist()):
        for in_word, subs, sub_probs in zip(sent, sent_subs, sent_probs):
            if in_word != PAD_ID:
                short_probs = list(map(lambda x: round(x, 4), sub_probs))
                writer.writerow([sent_id, in_word, subs, short_probs])

def dict_to_device(inputs, device):
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

def log_times(args, time_took, total_num_of_tokens):
    filename = os.path.join("benchmarks", f"timing_{str(time()).split('.')[0]}.txt")
    with open(filename, 'w') as f:
        f.write(os.path.basename(__file__).split('.')[0])
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write(f"Token per second: {total_num_of_tokens/time_took}\n")
        f.write(f"Total hours: {time_took/60/60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--alternatives_file", type=str, default="alternatives.tsv")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_tokens_per_batch", type=int, default=-1)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--adaptive_sampler", action="store_true")
    parser.add_argument("--iterativly_go_over_matrices", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    if args.adaptive_sampler:
        assert args.max_tokens_per_batch > 0 and \
            args.batch_size == 1 and \
            "Expecting arguments for adaptive sampler"
    else:
        assert args.max_tokens_per_batch == -1 and \
            args.batch_size > 1 and \
            args.max_seq_length > -1, \
            "Expecting arguments for simple sampler"

    main(args)
