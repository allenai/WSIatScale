from create_dataset import merge_sents
from transformers import AutoTokenizer


temp_file = 'tmp.jsonl'
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', use_fast=True)


def test_merge_sents_1():
    sents = ['This is the first sentence.', 'This is the second sentence.', 'This is the third sentence.']
    ret = list(merge_sents(tokenizer, sents))
    gold = [(0, 'This is the first sentence. This is the second sentence. This is the third sentence.')]
    assert ret == gold

def test_merge_sents_2():
    sents = ['This is the first sentence.'*500, 'This is the second sentence.', 'This is the third sentence.']
    ret = list(merge_sents(tokenizer, sents))
    gold = [(0, 'This is the first sentence.'*500), (1, 'This is the second sentence. This is the third sentence.')]
    assert ret == gold

def test_merge_sents_3():
    sents = ['This is the first sentence.'*500, 'This is the second sentence.'*500, 'This is the third sentence.'*500]
    ret = list(merge_sents(tokenizer, sents))
    gold = [(0, 'This is the first sentence.'*500), (1, 'This is the second sentence.'*500), (2, 'This is the third sentence.'*500)]
    assert ret == gold

def test_merge_sents_4():
    sents = ['This is the first sentence.']
    ret = list(merge_sents(tokenizer, sents))
    gold = [(0, 'This is the first sentence.')]
    assert ret == gold

def test_merge_sents_5():
    sents = ['This is the first sentence.'*500]
    ret = list(merge_sents(tokenizer, sents))
    gold = [(0, 'This is the first sentence.'*500)]
    assert ret == gold