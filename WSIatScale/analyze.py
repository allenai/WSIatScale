import argparse
from dataclasses import dataclass
import json
import os
import random
from typing import Tuple

import numpy as np
from tqdm import tqdm

SEED = 111

MAX_REPS = 100

REPS_DIR = 'replacements'
INVERTED_INDEX_DIR = 'inverted_index'
LEMMATIZED_VOCAB_FILE = 'lemmatized_vocab.json'

@dataclass
class Instance:
    doc_id: int
    reps: Tuple
    probs: np.array
    sent: np.array

class RepInstances:
    def __init__(self, lemmatized_vocab_path):
        self.data = []
        self.lemmatized_vocab_path = lemmatized_vocab_path
        if self.lemmatized_vocab_path:
            self.lemmatized_vocab = {int(k): v for k, v in json.load(open(self.lemmatized_vocab_path, 'r')).items()}

    def populate(self, sent_and_positions, reps, probs, full_stop_index):
        for sent, token_pos_in_sent, global_token_pos, doc_id in sent_and_positions:
            for local_pos, global_pos in zip(token_pos_in_sent, global_token_pos):
                single_sent = self.find_single_sent_around_token(sent, local_pos, full_stop_index)
                curr_reps = reps[global_pos]
                curr_probs = probs[global_pos]
                if self.lemmatized_vocab:
                    curr_reps, curr_probs = self.lemmatize_reps_and_probs(curr_reps, curr_probs)
                self.data.append(Instance(doc_id=doc_id,
                                          reps=curr_reps,
                                          probs=curr_probs,
                                          sent=single_sent))

    def lemmatize_reps_and_probs(self, curr_reps, curr_probs):
        curr_reps = list(map(lambda x: self.lemmatized_vocab[x], curr_reps))
        new_reps = []
        seen_lemmas = set()
        element_indices_to_delete = []
        for i, rep in enumerate(curr_reps):
            if rep in seen_lemmas:
                element_indices_to_delete.append(i)
            else:
                new_reps.append(rep)
            seen_lemmas.add(rep)
        curr_probs = np.delete(curr_probs, element_indices_to_delete)
        return new_reps, curr_probs

    def populate_specific_size(self, n_reps):
        if n_reps == MAX_REPS:
            return self

        for instance in self.data:
            instance.reps = instance.reps[:n_reps]
            instance.probs = instance.probs[:n_reps]
            instance.probs /= instance.probs.sum()

    def remove_certain_words(self, remove_query_word, remove_stop_words, tokenizer, word):
        words_to_remove = []
        if remove_query_word:
            words_to_remove += [word.lower().lstrip(), f" {word.lower().lstrip()}", word.lower().title().lstrip(), f" {word.title().lstrip()}"]
        tokens_to_remove = []
        for w in words_to_remove:
            t = tokenizer.encode(w, add_special_tokens=False)
            if len(t) == 1:
                tokens_to_remove.append(t[0])
        if remove_stop_words:
            tokens_to_remove += [2022, 1997, 2012, 2011, 1999, 1012, 1024, 1010, 1998, 1996, 2007, 1037, 2049, 1013, 2025, 1011, 1000]
                              # ['be', 'of', 'at', 'by', 'in', '.', ':', ',', 'and', 'the', 'with', 'a', 'its', '/', 'not', '-', '"']

        for instance in self.data:
            if not all([tokenizer.decode([r]).startswith('#') for r in instance.reps]):
                new_reps, new_probs = zip(*[(r, p) for r, p in zip(instance.reps, instance.probs) if not tokenizer.decode([r]).startswith('#')])
            if not all([r in tokens_to_remove for r in new_reps]):
                new_reps, new_probs = zip(*[(r, p) for r, p in zip(new_reps, new_probs) if r not in tokens_to_remove])
            if len(new_reps) != len(instance.reps):
                instance.reps = new_reps
                instance.probs = np.array(new_probs)
                instance.probs /= instance.probs.sum()

    @staticmethod
    def find_single_sent_around_token(concated_sents, local_pos, full_stop_index):
        if full_stop_index == None:
            return concated_sents

        full_stops_indices = np.where(concated_sents == full_stop_index)[0]
        if len(full_stops_indices) == 0:
            return concated_sents
        end_index = full_stops_indices.searchsorted(local_pos)
        start = full_stops_indices[end_index-1]+1 if end_index != 0 else 0
        if end_index == len(full_stops_indices):
            return concated_sents[start:]
        end = full_stops_indices[end_index]+1
        out = concated_sents[start:end]
        out = out[out!=0]
        out = out[out!=2]
        return out

    def merge(self, other):
        for key, sents in other.data.items():
            if key not in self.data:
                self.data[key] = sents
            else:
                self.data[key].append(sents)

def tokenize(tokenizer, word):
    token = tokenizer.encode(word, add_special_tokens=False)
    if len(token) > 1:
        raise ValueError('Word given is more than a single wordpiece.')
    token = token[0]
    return token

def read_files(token, data_dir, sample_n_files, full_stop_index, should_lemmatize=True, bar=tqdm):
    files_to_pos = read_inverted_index(os.path.join(data_dir, INVERTED_INDEX_DIR), token)
    if sample_n_files > 0 and len(files_to_pos) > sample_n_files:
        random.seed(SEED)
        sampled_keys = random.sample(files_to_pos.keys(), sample_n_files)
        files_to_pos = {k: files_to_pos[k] for k in sampled_keys}

    n_matches = 0
    lemmatized_vocab_path = os.path.join(data_dir, LEMMATIZED_VOCAB_FILE) if should_lemmatize else None
    rep_instances = RepInstances(lemmatized_vocab_path)

    replacements_dir = os.path.join(data_dir, REPS_DIR)
    for file, token_positions in bar(files_to_pos.items()):
        data = np.load(os.path.join(replacements_dir, f"{file}.npz"))

        tokens = data['tokens']
        lengths = data['sent_lengths']
        reps = data['replacements']
        probs = data['probs']
        doc_ids = data['doc_ids']

        sent_and_positions = list(find_sent_and_positions(token_positions, tokens, lengths, doc_ids))

        rep_instances.populate(sent_and_positions, reps, probs, full_stop_index)

        n_matches += len(token_positions)

    msg = f"Found Total of {n_matches} Matches in {len(files_to_pos)} Files."
    return rep_instances, msg

def find_sent_and_positions(token_positions, tokens, lengths, doc_ids):
    token_positions = np.array(token_positions)
    length_sum = 0
    for length, doc_id in zip(lengths, doc_ids):
        token_pos = token_positions[np.where(np.logical_and(token_positions >= length_sum, token_positions < length_sum + length))[0]]
        if len(token_pos) > 0:
            yield tokens[length_sum:length_sum + length], token_pos-length_sum, token_pos, doc_id
        length_sum += length

def read_inverted_index(inverted_index, token):
    inverted_index_file = os.path.join(inverted_index, f"{token}.jsonl")
    if not os.path.exists(inverted_index_file):
        raise ValueError('token is not in inverted index')
    index = {}
    with open(inverted_index_file, 'r') as f:
        for line in f:
            index.update(json.loads(line))
    return index

def prepare_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--replacements_dir", type=str, default="/home/matane/matan/dev/datasets/processed_for_WSI/CORD-19/replacements/done")
    parser.add_argument("--word", type=str, default='race')
    parser.add_argument("--inverted_index", type=str, default='/home/matane/matan/dev/datasets/processed_for_WSI/CORD-19/inverted_index.json')
    parser.add_argument("--n_reps", type=int, default=5)
    parser.add_argument("--sample_n_files", type=int, default=1000)
    parser.add_argument("--n_bow_reps_to_report", type=int, default=10, help="How many different replacements to report")
    parser.add_argument("--n_sents_to_print", type=int, default=2, help="Num sents to print")
    parser.add_argument("--show_top_n_clusters", type=int, default=20)
    parser.add_argument("--show_top_n_words_per_cluster", type=int, default=100)

    parser.add_argument("--cluster_alg", type=str, default=None, choices=['kmeans', 'agglomerative_clustering', 'dbscan'])
    parser.add_argument("--n_clusters", type=int, help="n_clusters, for kmeans and agglomerative_clustering")
    parser.add_argument("--distance_threshold", type=float, help="for agglomerative_clustering")
    parser.add_argument("--affinity", type=str, help="for agglomerative_clustering")
    parser.add_argument("--linkage", type=str, help="for agglomerative_clustering", default='complete')
    parser.add_argument("--eps", type=float, help="for dbscan")
    parser.add_argument("--min_samples", type=float, help="for dbscan")
    args = parser.parse_args()

    return args

def assert_arguments(args):
    if args.cluster_alg == 'kmeans':
        assert args.n_clusters is not None, \
            "kmeans requires --n_clusters"
    elif args.cluster_alg == 'agglomerative_clustering':
        assert args.n_clusters is not None or args.distance_threshold is not None, \
            "agglomerative_clustering requires either --n_clusters or --distance_threshold"
    elif args.cluster_alg == 'dbscan':
        assert args.eps is not None and args.min_samples is not None, \
            "dbscan requires either --eps or --min_samples"
        assert args.n_clusters is None, \
            "dbscan doesn't need --n_clusters"