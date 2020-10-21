import argparse
import json
import os
import spacy
from tqdm import tqdm

from transformers import AutoTokenizer

def create_lemmatized_vocab(outdir, model_hf_path):
    lemmatized_vocab = prepare_lemmatized_vocab(model_hf_path)
    outfile = os.path.join(outdir, f"lemmatized_vocab.json")
    json.dump(lemmatized_vocab, open(outfile, 'w'))

def prepare_lemmatized_vocab(model_hf_path):
    same_count = 0
    lemmatized_vocab = {}
    nlp = spacy.load("en", disable=['ner', 'parser'])
    tokenizer = AutoTokenizer.from_pretrained(model_hf_path, use_fast=True)
    vocab = tokenizer.get_vocab()
    for word, index in tqdm(vocab.items()):
        lemma = lemmatize_with_exceptions(nlp, tokenizer, vocab, index, word)
        lemma_index = vocab[lemma]
        lemmatized_vocab[index] = lemma_index

    for k, v in lemmatized_vocab.items():
        if k == v:
            same_count += 1

    # for original_token, lemma_token in lemmatized_vocab.items():
    #     if original_token != lemma_token:
    #         print(f"{tokenizer.decode([original_token])} -> {tokenizer.decode([lemma_token])}")
    print(f"Out of {len(vocab)} tokens, {same_count} are left unchanged.")
    
    return lemmatized_vocab

def lemmatize_with_exceptions(nlp, tokenizer, vocab, index, word):
    if index in tokenizer.all_special_ids or \
        word.startswith('#') or word.startswith('[unused') or \
        word in ['im', 'id', 'cannot', 'wed', 'gotta']: #not good enough lemmatizing.
        ret = word
    else:
        spacy_token = nlp(word)[0]
        if spacy_token.lemma_ == '-PRON-':
            ret = word
        ret = spacy_token.lemma_

    if ret not in vocab:
        ret = word

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--model_hf_path", type=str, default='bert-large-uncased')
    args = parser.parse_args()
    create_lemmatized_vocab(args.outdir, args.model_hf_path)