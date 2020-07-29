import os
import numpy as np

from find_occurances import *
from transformers import AutoTokenizer

preds_dir = 'tests/stubs'

def test_find_token_idx_in_row():
    token = 111 # vocab id for "the"
    with open(os.path.join(preds_dir, 'in_words.npy'), 'rb') as words_f:
        tokens = np.load(words_f)
    token_idx_in_row = find_token_idx_in_row(tokens, token)
    assert all(token_idx_in_row[:7] == [5, 32, 60, 66, 78, 96, 99])
    assert len(token_idx_in_row) == 50

def test_find_sents_with_token():
    token = 111 # vocab id for "have"
    with open(os.path.join(preds_dir, 'in_words.npy'), 'rb') as words_f, open(os.path.join(preds_dir, 'sent_lengths.npy'), 'rb') as lengths_f:
        tokens = np.load(words_f)
        lengths = np.load(lengths_f)
        token_idx_in_row = find_token_idx_in_row(tokens, token)
        sents_with_token = list(find_sents_with_token(token_idx_in_row, tokens, lengths))
        for sent, positions in sents_with_token:
            assert token in sent
            for pos in positions:
                assert sent[pos] == token