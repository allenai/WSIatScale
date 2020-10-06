import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer

tokenizer_params = {'CORD-19': 'allenai/scibert_scivocab_uncased',
                    'Wikipedia-roberta': 'roberta-large',
                    'Wikipedia-BERT': 'bert-large-cased-whole-word-masking',}

def main(replacements_dir, outdir, dataset, words_file, single_word):
    assert words_file is not None or single_word is not None

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_params[dataset], use_fast=True)
    if words_file:
        with open(words_file, 'r') as f:
            words = f.readlines()
        words = [w.rstrip() for w in words]
    else:
        words = [single_word]

    first_token = tokenizer(words[0], add_special_tokens=False)['input_ids']
    assert len(first_token) == 1, "pick a different word to check if files already exist."
    assert not os.path.exists(os.path.join(outdir, f"{first_token[0]}.txt")), "Files already exist."

    #Doing this because index is too big for memory so saving.
    files_step = 1000
    left_file_id = 0; right_file_id = 1000
    print(f"total {len(os.listdir(replacements_dir))} files.")
    while(len(os.listdir(replacements_dir)) > left_file_id):
        index(tokenizer, words[:], replacements_dir, outdir, dataset, which_files=(left_file_id, right_file_id))
        left_file_id = right_file_id
        right_file_id += files_step

def index(tokenizer, words, replacements_dir, outdir, dataset, bar=tqdm, which_files=None):
    index_dict = {}

    if dataset == 'Wikipedia-roberta':
        words_with_white_space = [f" {w}" for w in words]
        words += words_with_white_space

    words_tokens = tokenizer(words, add_special_tokens=False)['input_ids']
    words_tokens = [t[0] for t in words_tokens if len(t) == 1]
    words_without_spaces = [not tokenizer.convert_ids_to_tokens(t)[0].startswith('Ġ') for t in words_tokens]

    all_files = os.listdir(replacements_dir)
    if which_files:
        all_files = all_files[which_files[0]:which_files[1]] # Keeping the dict in memory is too expensive.
    for filename in tqdm(all_files):
        try: #TODO: getting BadZipFile
            npzfile = np.load(os.path.join(replacements_dir, filename))
            file_tokens = npzfile['tokens']
            file_id = filename.split('.')[0]
            for token, word_without_space in zip(words_tokens, words_without_spaces):
                valid_positions = []
                for pos in np.where(file_tokens == token)[0]:
                    if full_word(tokenizer, file_tokens, pos, dataset, word_without_space):
                        valid_positions.append(int(pos))
                if len(valid_positions) > 0:
                    if token not in index_dict:
                        index_dict[token] = {file_id: valid_positions}
                    else:
                        index_dict[token][file_id] = valid_positions
        except:
            continue

    for token, positions in index_dict.items():
        token_outfile = os.path.join(outdir, f"{token}.jsonl")
        with open(token_outfile, 'a') as f:
            f.write(json.dumps(positions)+'\n')

def full_word(tokenizer, file_tokens, pos, dataset, word_without_space):
    if dataset == 'allenai/scibert_scivocab_uncased' or dataset == 'Wikipedia-BERT':
        if pos + 1 == len(file_tokens):
            return True
        if tokenizer.decode([file_tokens[pos + 1]]).startswith('##'):
            return False
        return True
    else: #'Wikipedia-RoBERTa'
        if word_without_space:
            if file_tokens[pos - 1] not in [0, 4, 6, 12, 22, 35, 43, 60, 72]:
                return False

        if pos + 1 == len(file_tokens) or tokenizer.convert_ids_to_tokens([file_tokens[pos + 1]])[0].startswith('Ġ'):
            return True
        else:
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--replacements_dir", type=str, default="replacements")
    parser.add_argument("--outdir", type=str, default='inverted_index')
    parser.add_argument("--dataset", type=str, choices=['CORD-19', 'Wikipedia-roberta', 'Wikipedia-BERT'])
    parser.add_argument("--words_file", type=str)
    parser.add_argument("--single_word", type=str)

    args = parser.parse_args()

    main(args.replacements_dir, args.outdir, args.dataset, args.words_file, args.single_word)