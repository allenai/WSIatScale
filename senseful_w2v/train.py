import argparse
from pathlib import Path
import numpy as np
import csv
from multiprocessing import Pool, cpu_count
from functools import partial

from utils.special_tokens import SpecialTokens

from gensim.models import Word2Vec

from transformers import AutoTokenizer
from tqdm import tqdm

tokenizer_params = {'CORD-19': 'allenai/scibert_scivocab_uncased',
                    'Wikipedia': 'bert-large-cased-whole-word-masking',}


def main(args):
    model_hf_path = tokenizer_params[args.dataset]
    tokenizer = AutoTokenizer.from_pretrained(model_hf_path, use_fast=True)
    special_tokens = SpecialTokens(model_hf_path)

    if args.dataset == 'Wikipedia':
        iter = WikipediaIterator(args.data_dir, tokenizer, special_tokens, args.processed_sents_cache_dir)
    elif args.dataset == 'CORD-19':
        iter = CORDIterator(args.data_dir, tokenizer, special_tokens, args.processed_sents_cache_dir)
    else:
        raise NotImplementedError


    sg = 0
    if args.alg == 'SG':
        sg = 1

    model = Word2Vec(sentences=iter,
                     vector_size=args.dims,
                     window=5,
                     min_count=5,
                     workers=cpu_count(),
                     sg=sg,
                     epochs=args.epochs)

    word_vectors = model.wv
    word_vectors.save(f"senseful_w2v/word_vectors/{args.dataset}/senseful_w2v.word_vectors-{args.name}-{args.epochs}epochs-{args.dims}dim-{args.alg}")

class DatasetIterator:
    def __init__(self, data_dir, tokenizer, special_tokens, processed_sents_cache_dir):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.processed_sents_cache_dir = processed_sents_cache_dir

        self.filenames = [f.name.split('-tokens.npy')[0] for f in self.replacements_dir().iterdir() if 'tokens' in f.name]
        for filename in self.filenames:
            assert self.senses_file(filename).exists()

        if len(list(self.processed_sents_cache_dir.iterdir())) == 0:
            self.cache_processed_sents()

    def __iter__(self):
        all_files = list(self.processed_sents_cache_dir.iterdir())
        for file in tqdm(all_files, total=len(all_files)):
            with open(file, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    yield row

    def cache_processed_sents(self):
        with Pool(cpu_count()) as p:
            list(tqdm(p.imap(self.write_processed_sents, self.filenames), total=len(self.filenames)))

    def write_processed_sents(self, filename):
        with open(self.processed_sents_cache_dir/filename, 'w') as cache_file:
            csv_writer = csv.writer(cache_file)

            tokens = np.load(self.tokens_file(filename), mmap_mode='r')
            senses = np.load(self.senses_file(filename), mmap_mode='r')
            assert len(tokens) == len(senses)
            wordpieces = [self.tokenizer.decode([t]) for t in tokens]

            sent = []
            for token, wordpiece, sense in zip(tokens, wordpieces, senses):
                if token == self.special_tokens.CLS:
                    continue

                if token == self.special_tokens.SEP:
                    csv_writer.writerow(sent)
                    sent = []
                    continue

                if sense != -1:
                    sent.append(wordpiece+f'_{sense}')
                else:
                    if '##' in wordpiece:
                        wordpiece = wordpiece.replace('##', '')
                        sent[-1] += wordpiece
                    else:
                        sent.append(wordpiece)

    def senses_file(self, f):
        return self.data_dir/'aligned_sense_idx'/f"{f}.npy"

    def tokens_file(self, f):
        return self.replacements_dir()/f"{f}-tokens.npy"

    def replacements_dir(self):
        raise NotImplementedError

class WikipediaIterator(DatasetIterator):
    def replacements_dir(self):
        return self.data_dir/'..'/'replacements'

class CORDIterator(DatasetIterator):
    def replacements_dir(self):
        return self.data_dir/'replacements'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--processed_sents_cache_dir", type=Path, required=True)
    # {data_dir}/aligned_sense_idx/processed_sents
    parser.add_argument("--dataset", choices=['Wikipedia', 'CORD-19'], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dims", type=int, default=100)
    parser.add_argument("--alg", type=str, choices=['CBOW', 'SG'], default='CBOW')
    parser.add_argument("--name", type=str, default="")
    args = parser.parse_args()
    main(args)