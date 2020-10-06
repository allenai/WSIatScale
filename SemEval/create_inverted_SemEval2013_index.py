import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer
from xml.etree import ElementTree

tokenizer_params = {'RoBERTa': 'roberta-large',
                    'bert-large-uncased': 'bert-large-uncased',}

def main(replacements_dir, outdir, model, data_file):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_params[model], use_fast=True)

    instance_id_to_doc_id = json.load(open(os.path.join(replacements_dir, "instance_id_to_doc_id.json"), 'r'))
    doc_id_to_inst_id = {v:k for k,v in instance_id_to_doc_id.items()}
    inst_id_to_word = get_inst_id_to_word(data_file)
    index(tokenizer, replacements_dir, outdir, model, doc_id_to_inst_id, inst_id_to_word)

def get_inst_id_to_word(data_file):
    inst_id_to_word = {}
    with open(data_file, encoding="utf-8") as xml_file:
        et_xml = ElementTree.parse(xml_file)
        for word in et_xml.getroot():
            for inst in word.getchildren():
                inst_id = inst.attrib['id']
                context = inst.find("context")
                before, target_word, _ = list(context.itertext())
                which_target = before.count(f" {target_word} ") #This ±. If same target word appears in the same sentence
                inst_id_to_word[inst_id] = [target_word, which_target]

    return inst_id_to_word

def index(tokenizer, replacements_dir, outdir, model, doc_id_to_inst_id, inst_id_to_word, bar=tqdm):
    index_dict = {}

    all_files = os.listdir(replacements_dir)
    all_files = [f for f in all_files if f.endswith('npz')]
    for filename in tqdm(all_files):
        npzfile = np.load(os.path.join(replacements_dir, filename))
        file_tokens = npzfile['tokens']
        doc_ids = npzfile['doc_ids']
        sent_lengths = npzfile['sent_lengths']
        inst_ids = [doc_id_to_inst_id[k] for k in doc_ids]
        lemmas = [k.split('.')[0] for k in inst_ids]
        words_to_index = [inst_id_to_word[k][0] for k in inst_ids]
        which_targets = [inst_id_to_word[k][1] for k in inst_ids]
        if model == 'RoBERTa':
            lemmas = [f" {l}" for l in lemmas]
            words_to_index = [f" {w}" for w in words_to_index]
        lemma_tokens = [tokenizer.encode(l, add_special_tokens=False) for l in lemmas]
        tokens_to_index = [tokenizer.encode(w, add_special_tokens=False) for w in words_to_index]

        file_id = filename.split('.npz')[0]
        length_sum = 0
        for lemma_token, word_to_index, curr_len, which_target in zip(lemma_tokens, tokens_to_index, sent_lengths, which_targets):
            assert len(lemma_token) == 1; lemma_token = lemma_token[0]
            if len(word_to_index) != 1:
                print(f"Can't process multi wordpiece word {tokenizer.decode(word_to_index)}")
                length_sum += curr_len
                continue

            pos = np.where(file_tokens[length_sum:length_sum + curr_len] == word_to_index)[0] + length_sum
            pos = pos[which_target]
            if full_word(tokenizer, file_tokens, pos, model, word_to_index):
                valid_position = int(pos)

                if lemma_token not in index_dict:
                    index_dict[lemma_token] = {file_id: [valid_position]}
                else:
                    if file_id not in index_dict[lemma_token]:
                        index_dict[lemma_token][file_id] = [valid_position]
                    else:
                        index_dict[lemma_token][file_id].append(valid_position)
            length_sum += curr_len

    for token, positions in index_dict.items():
        token_outfile = os.path.join(outdir, f"{token}.jsonl")
        with open(token_outfile, 'a') as f:
            f.write(json.dumps(positions)+'\n')

def full_word(tokenizer, file_tokens, pos, model, word_without_space):
    if model == 'bert-large-uncased':
        if pos + 1 == len(file_tokens):
            return True
        if tokenizer.decode([file_tokens[pos + 1]]).startswith('##'):
            return False
        return True
    else: #'RoBERTa'
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
    parser.add_argument("--model", type=str, choices=['RoBERTa', 'bert-large-uncased'])
    parser.add_argument("--data_file", type=str)

    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    main(args.replacements_dir, args.outdir, args.model, args.data_file)