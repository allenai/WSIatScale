# pylint: disable=not-callable
import argparse
import csv
import json
import os
from spacy.lang.en import English
from tqdm import tqdm

from transformers import AutoTokenizer
MAX_LENGTH = 512

class Sentencizer:
    """
    Singleton for spacy's nlp
    """
    __instance = None
    def __new__(cls):
        if Sentencizer.__instance is None:

            nlp = English()
            sentencizer = nlp.create_pipe("sentencizer")
            nlp.add_pipe(sentencizer)
            Sentencizer.__instance = nlp
        return Sentencizer.__instance

def split_to_sents(text):
    sentencizer = Sentencizer()
    ret = []
    for sent in sentencizer(text).sents:
        ret.append(sent.text)
    return ret

def remove_citations(text, citations):
    clean_text = text
    for citation in reversed(citations):
        assert clean_text[citation['start']:citation['end']] == citation['text']
        clean_text = clean_text[:citation['start']] + clean_text[citation['end']+1:]
    return clean_text

def read_full_body(file_path):
    with open(file_path, 'r') as file:
        body = []
        file_dict = json.load(file)
        for para_dict in file_dict['body_text']:
            paragraph = remove_citations(para_dict['text'], para_dict['cite_spans'])
            body += split_to_sents(paragraph)
        return body

def csv_length(path):
    with open(path, 'r') as file:
        reader = csv.reader(file)
        return sum(1 for row in reader)-1

def read_data_files(data_dir):
    csv_path = os.path.join(data_dir, 'metadata.csv')
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in tqdm(reader, total=csv_length(csv_path)):
            full_body_sents = None
            abstract_sents = split_to_sents(row[8])
            if len(abstract_sents) == 0:
                continue
            pdf_json_files = row[15]
            if pdf_json_files:
                pdf_json_files = pdf_json_files.split(';')[0]
                full_body_sents = read_full_body(os.path.join(data_dir, pdf_json_files))

            yield (row[0], abstract_sents, full_body_sents)

def merge_sents_and_write(file, tokenizer, cord_uid, section, sents):
    for offset, text in merge_sents(tokenizer, sents):
        write(file, cord_uid, section, offset, text)

def merge_sents(tokenizer, sents):
    i = 0
    if len(sents) == 0: return
    encoding = tokenizer(
        [(sent) for sent in sents],
        max_length=512,
        padding="do_not_pad",
        truncation=True,
        add_special_tokens=True
    )
    lengths = [len(x) for x in encoding['input_ids']]

    batch_length = 0
    text = None
    offset = 0
    for i, (length, sent) in enumerate(zip(lengths, sents)):
        should_write = False
        batch_length += length
        if batch_length > MAX_LENGTH:
            should_write = True
        else:
            if not (len(lengths) > i+1 and (batch_length + lengths[i+1]) <= MAX_LENGTH):
                should_write = True

        if text is None:
            text = sent
        else:
            text += f" {sent}"

        if should_write:
            yield offset, text
            batch_length = 0
            text = None
            offset += 1

def write(file, cord_uid, section, offset, text):
    record = {'cord_uid': cord_uid,
              'section': section,
              'offset': offset,
              'text': text}
    json_record = json.dumps(record)
    file.write(json_record+'\n')

def write_data_to_jsonl(out_path, tokenizer, data):
    with open(out_path, 'w') as file:
        for cord_uid, abstract_sents, full_body_sents in data:
            merge_sents_and_write(file, tokenizer, cord_uid, 'abstract', abstract_sents)
            if full_body_sents:
                merge_sents_and_write(file, tokenizer, cord_uid, 'body', full_body_sents)

def main(args):
    data = read_data_files(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_hg_path, use_fast=True)
    write_data_to_jsonl(args.out_file, tokenizer, data)
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--model_hg_path", type=str, choices=['allenai/scibert_scivocab_uncased', 'roberta-large'])
    args = parser.parse_args()
    main(args)