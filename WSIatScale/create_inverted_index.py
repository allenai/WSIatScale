# pylint: disable=import-error
import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer

from utils.utils import tokenizer_params

def main(replacements_dir, outdir, dataset):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_params[dataset], use_fast=True)
    tokens_to_index = full_words_tokens(dataset, tokenizer)

    assert len(os.listdir(outdir)) == 0, "No indexing already exists."

    #Doing this because index is too big for memory so saving.
    files_step = 1000
    left_file_id = 0; right_file_id = 1000
    print(f"total {len(os.listdir(replacements_dir))} files.")
    while(len(os.listdir(replacements_dir)) > left_file_id):
        index(tokenizer, tokens_to_index, replacements_dir, outdir, dataset, which_files=(left_file_id, right_file_id))
        left_file_id = right_file_id
        right_file_id += files_step
        print(f"Done with {left_file_id} first files.")

def index(tokenizer, tokens_to_index, replacements_dir, outdir, dataset, bar=tqdm, which_files=None):
    index_dict = {}

    all_files = os.listdir(replacements_dir)
    if which_files:
        all_files = all_files[which_files[0]:which_files[1]] # Keeping the dict in memory is too expensive.
    for filename in tqdm(all_files):
        try: #getting BadZipFile
            npzfile = np.load(os.path.join(replacements_dir, filename))
            file_tokens = npzfile['tokens']
            file_id = filename.split('.')[0]
            tok_to_positions = {}
            for pos, token in enumerate(file_tokens):
                if token not in tokens_to_index:
                    continue
                if full_word(tokenizer, file_tokens, pos, dataset):
                    if token not in tok_to_positions:
                        tok_to_positions[token] = []
                    tok_to_positions[token].append(int(pos))

            for token, token_valid_positions in tok_to_positions.items():
                if token not in index_dict:
                    index_dict[token] = {file_id: token_valid_positions}
                else:
                    index_dict[token][file_id] = token_valid_positions
        except:
            print(f"Got BadZipFile. Couldn't read file {filename}")
            continue

    for token, positions in index_dict.items():
        token_outfile = os.path.join(outdir, f"{token}.jsonl")
        with open(token_outfile, 'a') as f:
            f.write(json.dumps(positions)+'\n')

def full_words_tokens(dataset, tokenizer):
    if dataset == 'Wikipedia-BERT':
        vocab = tokenizer.get_vocab()
        ret = set([token for word, token in vocab.items() if valid_word(token, word)])
        ret -= set([22755, 1232, 23567, 20262]) #Do not appear in Wikipedia.
        return ret
    elif dataset == 'allenai/scibert_scivocab_uncased':
        raise NotImplementedError
    elif dataset == 'Wikipedia-RoBERTa':
        raise NotImplementedError
    else:
        raise "Dataset not available"

def valid_word(token, word):
    #up until 1102 tokens are [unused], single tokens etc.
    if token > 1102 and not word.startswith('##'):
        return True
    return False

def full_word(tokenizer, file_tokens, pos, dataset):
    if dataset == 'allenai/scibert_scivocab_uncased' or dataset == 'Wikipedia-BERT':
        if pos + 1 == len(file_tokens):
            return True
        if tokenizer.decode([file_tokens[pos + 1]]).startswith('##'):
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