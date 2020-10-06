import argparse
import json
import os

import numpy as np
from tqdm import tqdm
import spacy
from tqdm import tqdm

from transformers import AutoTokenizer
from xml.etree import ElementTree

tokenizer_params = {'RoBERTa': 'roberta-large',
                    'bert-large-uncased': 'bert-large-uncased',}

def main(replacements_dir, outdir, model):

    inst_id_to_doc_id = json.load(open(os.path.join(replacements_dir, "../instance_id_to_doc_id.json"), 'r'))
    inst_id_to_target_pos = json.load(open(os.path.join(replacements_dir, "../instance_id_to_target_pos.json"), 'r'))
    doc_id_to_inst_id = {v:k for k,v in inst_id_to_doc_id.items()}
    index(replacements_dir, outdir, model, doc_id_to_inst_id, inst_id_to_target_pos)

def index(replacements_dir, outdir, model, doc_id_to_inst_id, inst_id_to_target_pos, bar=tqdm):
    index_dict = {}

    all_files = os.listdir(replacements_dir)
    all_files = [f for f in all_files if f.endswith('npz')]
    for filename in tqdm(all_files):
        npzfile = np.load(os.path.join(replacements_dir, filename))
        doc_ids = npzfile['doc_ids']
        sent_lengths = npzfile['sent_lengths']
        inst_ids = [doc_id_to_inst_id[k] for k in doc_ids]
        target_positions = [int(inst_id_to_target_pos[k]) for k in inst_ids]
        lemmas = [k.split('.')[0] for k in inst_ids]
        if model == 'RoBERTa':
            lemmas = [f" {l}" for l in lemmas]
            words_to_index = [f" {w}" for w in words_to_index]

        file_id = filename.split('.npz')[0]
        length_sum = 0
        for lemma, curr_len, local_target_pos in zip(lemmas, sent_lengths, target_positions):
            global_pos = length_sum + local_target_pos
            if lemma not in index_dict:
                index_dict[lemma] = {file_id: [global_pos]}
            else:
                if file_id not in index_dict[lemma]:
                    index_dict[lemma][file_id] = [global_pos]
                else:
                    index_dict[lemma][file_id].append(global_pos)
            length_sum += int(curr_len)

    for token, positions in index_dict.items():
        token_outfile = os.path.join(outdir, f"{token}.jsonl")
        with open(token_outfile, 'w') as f:
            f.write(json.dumps(positions)+'\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--replacements_dir", type=str, default="replacements")
    parser.add_argument("--outdir", type=str, default='inverted_index')
    parser.add_argument("--model", type=str, choices=['RoBERTa', 'bert-large-uncased'])

    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    main(args.replacements_dir, args.outdir, args.model)