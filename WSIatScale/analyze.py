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
    def __init__(self, lemmatized_vocab_path=None):
        self.data = []
        self.lemmatized_vocab_path = lemmatized_vocab_path
        if self.lemmatized_vocab_path:
            self.lemmatized_vocab = {int(k): v for k, v in json.load(open(self.lemmatized_vocab_path, 'r')).items()}

    def populate(self, paragraph_and_positions, reps, probs, full_stop_index, instance_attributes):
        for paragraph, token_pos_in_paragraph, global_token_pos, doc_id in paragraph_and_positions:
            for local_pos, global_pos in zip(token_pos_in_paragraph, global_token_pos):
                single_sent, _ = self.find_single_sent_around_token(paragraph, local_pos, full_stop_index) \
                    if 'sent' in instance_attributes else (None, None)
                curr_reps = np.array(reps[global_pos]) if 'reps' in instance_attributes else None
                curr_probs = np.array(probs[global_pos]) if 'probs' in instance_attributes else None
                if self.lemmatized_vocab_path:
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
            if instance.probs is not None:
                instance.probs = instance.probs[:n_reps]
                instance.probs /= instance.probs.sum()

    def remove_certain_words(self, tokenizer, word, remove_query_word, half_words_list=None):
        words_to_remove = []
        if remove_query_word:
            words_to_remove += [word.lower().lstrip(), f" {word.lower().lstrip()}", word.lower().title().lstrip(), f" {word.title().lstrip()}"]
        tokens_to_remove = []
        for w in words_to_remove:
            t = tokenizer.encode(w, add_special_tokens=False)
            if len(t) == 1:
                tokens_to_remove.append(t[0])

        if len(tokens_to_remove) > 0:
            for instance in self.data:
                new_reps, new_probs = instance.reps, instance.probs
                if half_words_list is not None:
                    if new_probs is None:
                        new_reps = [r for r in new_reps if r not in half_words_list]
                    else:
                        new_reps, new_probs = zip(*[(r, p) for r, p in zip(new_reps, new_probs) if r not in half_words_list])
                if not all([r in tokens_to_remove for r in new_reps]):
                    if new_probs is None:
                        new_reps = [r for r in new_reps if r not in tokens_to_remove]
                    else:
                        new_reps, new_probs = zip(*[(r, p) for r, p in zip(new_reps, new_probs) if r not in tokens_to_remove])
                if len(new_reps) > 0 and len(new_reps) != len(instance.reps):
                    instance.reps = new_reps
                    if instance.probs is not None:
                        instance.probs = np.array(new_probs)
                        instance.probs /= instance.probs.sum()

    @staticmethod
    def find_single_sent_around_token(concated_sents, local_pos, full_stop_index):
        if full_stop_index is None:
            return concated_sents, local_pos

        full_stops_indices = np.where(concated_sents == full_stop_index)[0]
        if len(full_stops_indices) == 0:
            return concated_sents, local_pos
        sent_idx_to_ret = full_stops_indices.searchsorted(local_pos)
        start = full_stops_indices[sent_idx_to_ret-1]+1 if sent_idx_to_ret != 0 else 0
        if sent_idx_to_ret == len(full_stops_indices):
            return concated_sents[start:], local_pos-start
        end = full_stops_indices[sent_idx_to_ret]+1
        out = concated_sents[start:end]
        # TODO
        out = out[out!=0]
        out = out[out!=2]
        return out, local_pos-start

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

def read_files(token,
               data_dir,
               sample_n_instances,
               full_stop_index,
               should_lemmatize=False,
               instance_attributes=['doc_id', 'reps', 'probs', 'sent'],
               inverted_index_dir=INVERTED_INDEX_DIR,
               bar=tqdm):
    files_to_pos = read_inverted_index(os.path.join(data_dir, inverted_index_dir), token, sample_n_instances)

    n_matches = 0
    lemmatized_vocab_path = os.path.join(data_dir, LEMMATIZED_VOCAB_FILE) if should_lemmatize else None
    rep_instances = RepInstances(lemmatized_vocab_path)

    for file, token_positions in bar(files_to_pos.items()):
        tokens = np.load(npy_file_path(data_dir, file, 'tokens'), mmap_mode='r')
        lengths = np.load(npy_file_path(data_dir, file, 'lengths'), mmap_mode='r')
        reps = np.load(npy_file_path(data_dir, file, 'reps'), mmap_mode='r')
        probs = np.load(npy_file_path(data_dir, file, 'probs'), mmap_mode='r') if 'probs' in instance_attributes else None
        doc_ids = np.load(npy_file_path(data_dir, file, 'doc_ids'), mmap_mode='r')

        paragraph_and_positions = list(find_paragraph_and_positions(token_positions, tokens, lengths, doc_ids))

        rep_instances.populate(paragraph_and_positions, reps, probs, full_stop_index, instance_attributes)

        n_matches += len(token_positions)

    msg = f"Found Total of {n_matches} Matches in {len(files_to_pos)} Files."
    return rep_instances, msg

def npy_file_path(data_dir, f, a):
    return os.path.join(os.path.join(data_dir, REPS_DIR), f"{f}-{a}.npy")

def find_paragraph_and_positions(token_positions, tokens, lengths, doc_ids):
    token_positions = np.array(token_positions)
    length_sum = 0
    for length, doc_id in zip(lengths, doc_ids):
        token_pos = token_positions[np.where(np.logical_and(token_positions >= length_sum, token_positions < length_sum + length))[0]]
        if len(token_pos) > 0:
            yield tokens[length_sum:length_sum + length], token_pos-length_sum, token_pos, doc_id
        length_sum += length

def read_inverted_index(inverted_index, token, sample_n_instances):
    inverted_index_file = os.path.join(inverted_index, f"{token}.jsonl")
    if not os.path.exists(inverted_index_file):
        raise ValueError(f'token {token} is not in inverted index')
    index = {}
    with open(inverted_index_file, 'r') as f:
        for line in f:
            index.update(json.loads(line))

    index = sample_instances(index, sample_n_instances)
    return index

def sample_instances(index, sample_n_instances):
    random.seed(SEED)

    ret = {}
    sample_n_instances = min(sample_n_instances, len(index))
    if sample_n_instances > 0:
        files = random.sample(list(index.keys()), sample_n_instances)
        ret = {file: [index[file][0]] for file in files}
    return ret

def prepare_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--replacements_dir", type=str, default="/home/matane/matan/dev/datasets/processed_for_WSI/CORD-19/replacements/done")
    parser.add_argument("--word", type=str, default='race')
    parser.add_argument("--inverted_index", type=str, default='/home/matane/matan/dev/datasets/processed_for_WSI/CORD-19/inverted_index.json')
    parser.add_argument("--n_reps", type=int, default=5)
    parser.add_argument("--sample_n_instances", type=int, default=1000)
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