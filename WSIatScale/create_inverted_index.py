import argparse
from collections import defaultdict
import json
import os

import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer

def main(replacements_dir, outfile, words_file, single_word):
    assert words_file is not None or single_word is not None

    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', use_fast=True)
    if words_file:
        create_new_index(tokenizer, words_file, replacements_dir, outfile)
    else:
        index_single_word(tokenizer, single_word, replacements_dir, outfile)

def index_single_word(tokenizer, word, replacements_dir, outfile, bar=tqdm):
    index = json.load(open(outfile, 'r'))
    word_token = tokenizer.encode(word, add_special_tokens=False)
    assert len(word_token) == 1
    word_token = word_token[0]

    assert word_token not in index
    all_files = sorted(os.listdir(replacements_dir), key=lambda k: int(k.split('.')[0]))
    for filename in bar(all_files):
        npzfile = np.load(os.path.join(replacements_dir, filename))
        file_tokens = npzfile['tokens']

        valid_positions = []
        for pos in np.where(file_tokens == word_token)[0]:
            if full_word(tokenizer, file_tokens, pos):
                valid_positions.append(int(pos))
        if len(valid_positions) > 0:
            file_id = filename.split('.')[0]
            if word_token not in index:
                index[word_token] = [[file_id, valid_positions]]
            else:
                index[word_token].append([file_id, valid_positions])

    json.dump(dict(index), open(outfile, 'w'))

    if word_token not in index:
        msg = f"Couldn't find any matches for {word}."
    else:
        msg = f"Found {len(index[word_token])} files with {word}. Please Rerun."

    return msg


def create_new_index(tokenizer, words_file, replacements_dir, outfile):
    index = defaultdict(list)

    with open(words_file, 'r') as f:
        words = f.readlines()
    words = [w.rstrip() for w in words]

    words_tokens = tokenizer(words, add_special_tokens=False)['input_ids']
    words_tokens = [t[0] for t in words_tokens if len(t) == 1]

    all_files = sorted(os.listdir(replacements_dir), key=lambda k: int(k.split('.')[0]))
    for filename in tqdm(all_files):
        npzfile = np.load(os.path.join(replacements_dir, filename))
        file_tokens = npzfile['tokens']
        for in_word in words_tokens:
            valid_positions = []
            for pos in np.where(file_tokens == in_word)[0]:
                if full_word(tokenizer, file_tokens, pos):
                    valid_positions.append(int(pos))
            if len(valid_positions) > 0:
                file_id = filename.split('.')[0]
                index[in_word].append([file_id, valid_positions])

    json.dump(dict(index), open(outfile, 'w'))

def full_word(tokenizer, file_tokens, pos):
    if pos + 1 == len(file_tokens):
        return True
    if tokenizer.decode([file_tokens[pos + 1]]).startswith('##'):
        return False

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--replacements_dir", type=str, default="replacements")
    parser.add_argument("--outfile", type=str, default='inverted_index.json')
    parser.add_argument("--words_file", type=str)
    parser.add_argument("--single_word", type=str)

    args = parser.parse_args()

    main(args.replacements_dir, args.outfile, args.words_file, args.single_word)