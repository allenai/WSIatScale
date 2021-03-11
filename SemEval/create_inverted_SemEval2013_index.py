import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer
from xml.etree import ElementTree

tokenizer_params = {'RoBERTa': 'roberta-large',
                    'bert-large-uncased': 'bert-large-uncased',
                    'bert-large-cased-whole-word-masking': 'bert-large-cased-whole-word-masking',}

def main(data_dir, outdir, model, data_file):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_params[model], use_fast=True)

    instance_id_to_doc_id = json.load(open(os.path.join(data_dir, "instance_id_to_doc_id.json"), 'r'))
    doc_id_to_inst_id = {v:k for k,v in instance_id_to_doc_id.items()}
    inst_id_to_word = get_inst_id_to_word(data_file)
    index(tokenizer, data_dir, outdir, model, doc_id_to_inst_id, inst_id_to_word)

def get_inst_id_to_word(data_file):
    inst_id_to_word = {}
    with open(data_file, encoding="utf-8") as xml_file:
        et_xml = ElementTree.parse(xml_file)
        for word in et_xml.getroot():
            for inst in word.getchildren():
                inst_id = inst.attrib['id']
                context = inst.find("context")
                before, target_word, _ = list(context.itertext())
                which_target = before.count(f" {target_word} ") # This is meh. If same target word appears in the same sentence
                inst_id_to_word[inst_id] = [target_word, which_target]

    return inst_id_to_word

def index(tokenizer, data_dir, outdir, model, doc_id_to_inst_id, inst_id_to_word):
    index_dict = {}
    replacements_dir = os.path.join(data_dir, 'replacements')

    all_files = os.listdir(replacements_dir)
    all_files = [f[:-len('-tokens.npy')] for f in all_files if f.endswith('tokens.npy')]
    for filename in tqdm(all_files):
        base_path = os.path.join(replacements_dir, filename)
        file_tokens = np.load(f"{base_path}-tokens.npy")
        doc_ids = np.load(f"{base_path}-doc_ids.npy")
        sent_lengths = np.load(f"{base_path}-lengths.npy")

        inst_ids = [doc_id_to_inst_id[k] for k in doc_ids]
        lemmas = ['.'.join(k.split('.', 2)[:2]) for k in inst_ids]
        words_to_index = [inst_id_to_word[k][0].lower() for k in inst_ids]
        which_targets = [inst_id_to_word[k][1] for k in inst_ids]
        if model == 'RoBERTa':
            lemmas = [f" {l}" for l in lemmas]
            words_to_index = [f" {w}" for w in words_to_index]
        tokens_to_index = [tokenizer.encode(w, add_special_tokens=False) for w in words_to_index]

        length_sum = 0
        for lemma, token_to_index, curr_len, which_target in zip(lemmas, tokens_to_index, sent_lengths, which_targets):
            if len(token_to_index) != 1:
                token_to_index = tokenizer.encode(lemma, add_special_tokens=False) #Checked, this is alright.
            token_to_index = token_to_index[0]

            pos = np.where(file_tokens[length_sum:length_sum + curr_len] == token_to_index)[0] + length_sum
            pos = pos[which_target]

            valid_position = int(pos)

            if lemma not in index_dict:
                index_dict[lemma] = {filename: [valid_position]}
            else:
                if filename not in index_dict[lemma]:
                    index_dict[lemma][filename] = [valid_position]
                else:
                    index_dict[lemma][filename].append(valid_position)
            length_sum += curr_len

    for token, positions in index_dict.items():
        token_outfile = os.path.join(outdir, f"{token}.jsonl")
        with open(token_outfile, 'a') as f:
            f.write(json.dumps(positions)+'\n')

# deprecated
# def full_word(tokenizer, file_tokens, pos, model, word_without_space):
#     if model == 'bert-large-uncased':
#         if pos + 1 == len(file_tokens):
#             return True
#         if tokenizer.decode([file_tokens[pos + 1]]).startswith('##'):
#             return False
#         return True
#     else: #'RoBERTa'
#         if word_without_space:
#             if file_tokens[pos - 1] not in [0, 4, 6, 12, 22, 35, 43, 60, 72]:
#                 return False

#         if pos + 1 == len(file_tokens) or tokenizer.convert_ids_to_tokens([file_tokens[pos + 1]])[0].startswith('Ġ'):
#             return True
#         else:
#             return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--outdir", type=str, default='inverted_index')
    parser.add_argument("--model", type=str, choices=['RoBERTa', 'bert-large-uncased', 'bert-large-cased-whole-word-masking'])
    parser.add_argument("--data_file", type=str)

    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    main(args.data_dir, args.outdir, args.model, args.data_file)
