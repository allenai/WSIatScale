import argparse
from collections import defaultdict
import json
import os

import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer

index = defaultdict(list)

def main(args):
    with open(args.words_file, 'r') as f:
        words = f.readlines()
    words = [w.rstrip() for w in words]

    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', use_fast=True)
    words_tokens = tokenizer(words, add_special_tokens=False)['input_ids']
    words_tokens = [t[0] for t in words_tokens if len(t) == 1]

    all_files = sorted(os.listdir(args.replacements_dir), key=lambda k: int(k.split('.')[0]))
    for filename in tqdm(all_files):
        npzfile = np.load(os.path.join(args.replacements_dir, filename))
        file_tokens = npzfile['tokens']
        for in_word in words_tokens:
            valid_positions = []
            for pos in np.where(file_tokens == in_word)[0]:
                if full_word(tokenizer, file_tokens, pos):
                    valid_positions.append(int(pos))
            if len(valid_positions) > 0:
                file_id = filename.split('.')[0]
                index[in_word].append([file_id, valid_positions])

    json.dump(dict(index), open(args.outfile, 'w'))

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
    parser.add_argument("--words_file", type=str, required=True)
    args = parser.parse_args()

    main(args)