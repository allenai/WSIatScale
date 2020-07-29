import argparse
import numpy as np
import os
from tqdm import tqdm

from transformers import AutoTokenizer
MAX_PREDS_TO_DISPLAY = 4
NUM_PREDS = 100

def main(args):
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', use_fast=True)

    token = tokenizer.encode(args.word, add_special_tokens=False)
    if len(token) > 1:
        raise Exception('Word given is more than a single wordpiece.')
    with open(os.path.join(args.preds_dir, 'in_words.npy'), 'rb') as words_f, \
         open(os.path.join(args.preds_dir, 'sent_lengths.npy'), 'rb') as lengths_f, \
         open(os.path.join(args.preds_dir, 'pred_ids.npy'), 'rb') as preds_f, \
         open(os.path.join(args.preds_dir, 'probs.npy'), 'rb') as probs_f:

        fsz = os.fstat(words_f.fileno()).st_size
        pbar = tqdm(total=fsz+1)
        prev_words_offset = 0
        preds_offset = 0
        while words_f.tell() < fsz:
            tokens = np.load(words_f)
            lengths = np.load(lengths_f)
            token_idx_in_row = find_token_idx_in_row(tokens, token)

            pbar.update(words_f.tell()-prev_words_offset)
            prev_words_offset = words_f.tell()

            if len(token_idx_in_row) != 0:
                preds_f.seek(preds_offset)
                probs_f.seek(preds_offset)

                preds = np.load(preds_f)
                probs = np.load(probs_f)
                sents_with_token_data = list(find_sents_with_token_data(token_idx_in_row, tokens, lengths))
                assert len(token_idx_in_row) >= len(sents_with_token_data)
                pp_sents_with_token(tokenizer, sents_with_token_data, preds, probs, args.word)
            preds_offset += tokens.shape[0] * 2 * NUM_PREDS + 128

        pbar.close()

def pp_sents_with_token(tokenizer, sents_with_token_data, preds, probs, word):
    green = '\x1b[32m'
    orange = '\x1b[33m'
    back_to_white = '\x1b[00m'
    for sent, token_pos_in_sent, global_token_pos in sents_with_token_data:
        splits = np.split(sent, token_pos_in_sent)
        out = tokenizer.decode(splits[0])
        for i, split in enumerate(splits[1:]):
            if out:
                out += ' '
            out += f"{green}(#{i}) {word}{back_to_white} " + tokenizer.decode(split[1:])
        for i, global_pos in enumerate(global_token_pos):
            out += f"\n{orange}(#{i}):"
            for pred, prob in zip(preds[global_pos, :MAX_PREDS_TO_DISPLAY], probs[global_pos, :MAX_PREDS_TO_DISPLAY]):
                out += f" {tokenizer.decode([pred])}-{prob:.5f}"
            out += back_to_white
        print(out)

def find_token_idx_in_row(tokens, token):
    return np.where(tokens == token)[0]

def find_sents_with_token_data(token_idx_in_row, tokens, lengths):
    length_sum = 0
    for length in lengths:
        token_location = token_idx_in_row[np.where(np.logical_and(token_idx_in_row >= length_sum, token_idx_in_row < length_sum + length))[0]]
        if len(token_location) > 0:
            yield tokens[length_sum:length_sum + length], token_location-length_sum, token_location
        length_sum += length

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--word", type=str, required=True)
    parser.add_argument("--preds_dir", type=str, default="preds")
    args = parser.parse_args()

    main(args)