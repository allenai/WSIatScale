import argparse
import json
import os
import spacy
from tqdm import tqdm

from transformers import AutoTokenizer

def create_lemmatized_vocab(outdir, model):
    lemmatized_vocab = prepare_lemmatized_vocab(model)
    outfile = os.path.join(outdir, f"lemmatized_vocabs-{model.replace('/', '_')}.json")
    json.dump(lemmatized_vocab, open(outfile, 'w'))

def prepare_lemmatized_vocab(model):
    same_count = 0
    lemmatized_vocab = {}
    nlp = spacy.load("en_core_web_lg", disable=['ner', 'parser'])
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    vocab = tokenizer.get_vocab()
    for word, index in tqdm(vocab.items()):
        lemma = lemmatize_with_exceptions(nlp, tokenizer, vocab, index, word)
        if lemma:
            lemma_index = vocab[lemma]
            lemmatized_vocab[index] = lemma_index

    for k, v in lemmatized_vocab.items():
        if k == v:
            same_count += 1

    for original_token, lemma_token in lemmatized_vocab.items():
        if original_token != lemma_token:
            print(f"{tokenizer.decode([original_token])} -> {tokenizer.decode([lemma_token])}")
    print(f"Out of {len(vocab)} tokens, {same_count} are left unchanged.")

    return lemmatized_vocab

def lemmatize_with_exceptions(nlp, tokenizer, vocab, index, word):
    if index in tokenizer.all_special_ids or \
        word.startswith('#') or word.startswith('[unused'):
        return None

    keep_as_is = ["McGee", "McGraw", "McPherson", "McCartney", "McCarthy", "MacDonald", "PlayStation", "McKenna", "McMahon", "McIntyre", "McKinley", "McGrath", "iOS", "McCoy", "McLean", "McLaren", "MiG", "McCormick", "GmbH", "PhD", "McCain", "McLaughlin", "McGuire", "McDonnell", "McGregor", "MHz", "MacArthur", "AllMusic", "YouTube", "McGill", "SmackDown", "McDonald", "McKenzie", "MacKenzie", "McKay", "McBride"]
    should_singalize_with_caps = ['DVDs', 'MPs', 'CDs', 'DJs', 'RBIs', 'NGOs']
    capitalized_letters = []
    for i, l in enumerate(word):
        if l.isupper():
            capitalized_letters.append(i)

    if word in keep_as_is:
        return word
    elif word in should_singalize_with_caps:
        return word[:-1]

    if word in ['cannot', 'gotta', '']: #not good enough lemmatizing.
        ret = word
    else:
        spacy_token = nlp(word)[0]
        if spacy_token.lemma_ == '-PRON-':
            ret = word
        else:
            ret = spacy_token.lemma_

            if len(capitalized_letters) > 0:
                if len(capitalized_letters) == 1:
                    ret = ret.capitalize()
                else:
                    ret = ret.upper()

            if ret not in vocab:
                ret = word

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--model", type=str, choices=['bert-large-uncased', 'bert-large-cased-whole-word-masking', 'allenai/scibert_scivocab_uncased'])
    args = parser.parse_args()
    create_lemmatized_vocab(args.outdir, args.model)