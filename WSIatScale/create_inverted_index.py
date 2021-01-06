# pylint: disable=import-error
import argparse
import json
from functools import partial
from multiprocessing import Pool, cpu_count
import os

import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer

from utils.utils import tokenizer_params
from utils.special_tokens import SpecialTokens

def main(replacements_dir, outdir, dataset):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_params[dataset], use_fast=True)
    special_tokens = SpecialTokens(tokenizer_params[dataset])
    tokens_to_index = special_tokens.full_words_tokens(tokenizer)

    assert len(os.listdir(outdir)) == 0, "No indexing already exists."
    number_to_tokens_files = len([f for f in os.listdir(replacements_dir) if f.endswith('tokens.npy')])
    print(f"total {number_to_tokens_files} files.")

    #Doing this because index is too big for memory so saving.
    files_step = 1000
    file_ranges = list(range(0, number_to_tokens_files, files_step)) + [number_to_tokens_files+1]
    which_files = []
    for i in range(len(file_ranges)-1):
        which_files.append((file_ranges[i], file_ranges[i+1]))

    partial_index = partial(index,
        special_tokens=special_tokens,
        tokens_to_index=tokens_to_index,
        replacements_dir=replacements_dir,
        outdir=outdir,
        dataset=dataset)

    with Pool(cpu_count()) as p:
        list(tqdm(p.imap(partial_index, which_files), total=len(which_files)))

def index(which_files, special_tokens, tokens_to_index, replacements_dir, outdir, dataset):
    index_dict = {}

    all_files = os.listdir(replacements_dir)
    all_files = sorted([f for f in all_files if f.endswith('tokens.npy')])
    if which_files:
        all_files = all_files[which_files[0]:which_files[1]] # Keeping the dict in memory is too expensive.
    for filename in all_files:
        file_id = filename.split('-tokens.npy')[0]
        file_tokens = np.load(os.path.join(os.path.join(replacements_dir, filename)))
        tok_to_positions = {}
        for pos, token in enumerate(file_tokens):
            if token not in tokens_to_index:
                continue
            lemma_token = special_tokens.lemmatize(token)
            if full_word(special_tokens, file_tokens, pos, dataset):
                if lemma_token not in tok_to_positions:
                    tok_to_positions[lemma_token] = []
                tok_to_positions[lemma_token].append(int(pos))

        for lemma_token, token_valid_positions in tok_to_positions.items():
            if lemma_token not in index_dict:
                index_dict[lemma_token] = {file_id: token_valid_positions}
            else:
                index_dict[lemma_token][file_id] = token_valid_positions

    for lemma_token, positions in index_dict.items():
        token_outfile = os.path.join(outdir, f"{lemma_token}.jsonl")
        with open(token_outfile, 'a') as f:
            f.write(json.dumps(positions)+'\n')

def full_word(special_tokens, file_tokens, pos, dataset):
    if dataset == 'CORD-19' or dataset == 'Wikipedia-BERT':
        if pos + 1 == len(file_tokens):
            return True
        if file_tokens[pos + 1] in special_tokens.half_words_list:
            return False
        return True
    else: #'Wikipedia-RoBERTa'
        raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--replacements_dir", type=str, default="replacements")
    parser.add_argument("--outdir", type=str, default='inverted_index')
    parser.add_argument("--dataset", type=str, choices=['CORD-19', 'Wikipedia-roberta', 'Wikipedia-BERT'])
    parser.add_argument("--words_file", type=str)

    args = parser.parse_args()

    main(args.replacements_dir, args.outdir, args.dataset)