import argparse
from collections import defaultdict
from enum import Enum
import json
import os

import numpy as np
import termplotlib as tpl
from tqdm import tqdm

from sklearn.cluster import AgglomerativeClustering
from transformers import AutoTokenizer

class Color(Enum):
    green = '\x1b[32m'
    orange = '\x1b[33m'
    back_to_white = '\x1b[00m'

    def __str__(self):
        return f"{self.value}"

class Jaccard:
    def __init__(self):
        self.matrix = None

    def init_matrix(self, length):
        self.matrix = np.zeros((length, length), dtype=np.float16)

    def pairwise_distance(self, X):
        length = len(X)
        self.init_matrix(length)
        for i in range(length):
            for j in range(i+1, length):
                distance = self.distance(X[i], X[j])
                self.matrix[i, j] = distance
                self.matrix[j, i] = distance

        return self.matrix

    def distance(self, x, y):
        return 1 - self.similarity(x, y)

    def similarity(self, x, y):
        intersection = len(x.intersection(y))
        union = len(x) + len(y) - intersection
        return float(intersection) / union

def main(args):
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', use_fast=True)

    token = tokenizer.encode(args.word, add_special_tokens=False)
    if len(token) > 1:
        raise Exception('Word given is more than a single wordpiece.')
    token = token[0]
    files = inverted_index(args, token)

    bag_of_reps = read_files(files, tokenizer, token)

    if args.report_reps_diversity:
        print_bag_of_reps(args, bag_of_reps, tokenizer)

    if args.cluster:
        cluster(args, bag_of_reps, tokenizer)

def read_files(files, tokenizer, token):
    if args.cluster or args.report_reps_diversity:
        bag_of_reps = defaultdict(list)

    for file in tqdm(files):
        data = np.load(os.path.join(args.replacements_dir, f"{file}.npz"))

        tokens = data['tokens']
        lengths = data['sent_lengths']
        reps = data['replacements']
        probs = data['probs']

        token_idx_in_row = find_token_idx_in_row(tokens, token)
        sent_and_positions = list(find_sent_and_positions(token_idx_in_row, tokens, lengths))

        if args.print:
            print_sents_with_token(args, tokenizer, sent_and_positions, reps, probs)
        if args.cluster or args.report_reps_diversity:
            populate_bag_of_reps(args, bag_of_reps, sent_and_positions, reps)

    return bag_of_reps

def cluster(args, bag_of_reps, tokenizer):
    top_n_to_cluster = args.top_n_to_cluster
    n_clusters = args.n_clusters
    distance_threshold = args.distance_threshold

    sorted_bag_of_reps = [k for k, _ in \
        sorted(bag_of_reps.items(), key=lambda kv: len(kv[1]), reverse=True)[:top_n_to_cluster]
    ]
    jaccard_matrix = Jaccard().pairwise_distance(sorted_bag_of_reps)

    clustering = AgglomerativeClustering(n_clusters=n_clusters,
                                         distance_threshold=distance_threshold,
                                         affinity='precomputed',
                                         linkage='average')
    clusters = clustering.fit_predict(jaccard_matrix)
    clustered_reps = defaultdict(list)
    for c, alter in zip(clusters, sorted_bag_of_reps):
        clustered_reps[c].append(alter)

    for i, cluster_reps in enumerate(clustered_reps.values()):
        print(f"Cluster {i}:")
        for reps in cluster_reps:
            print(tokenizer.decode(list(reps)))
        print()

def print_bag_of_reps(args, bag_of_reps, tokenizer):
    n_sents_to_print = args.n_sents_to_print
    n_bow_reps_to_report = args.n_bow_reps_to_report

    for k, sent_and_positions in sorted(bag_of_reps.items(), key=lambda kv: len(kv[1]), reverse=True)[:n_bow_reps_to_report]:
        print(f"{Color.orange}{tokenizer.decode(list(k))} - {len(sent_and_positions)}{Color.back_to_white}")
        for sent in sent_and_positions[:n_sents_to_print]:
            print(tokenizer.decode(sent))
        print()

    reps_to_num = defaultdict(int)
    for _, v in bag_of_reps.items():
        reps_to_num[len(v)] += 1

    keys, values = zip(*sorted(reps_to_num.items(), key=lambda kv: kv[0], reverse=True))
    if len(keys) > n_bow_reps_to_report:
        keys = keys[:int(n_bow_reps_to_report/2)] + keys[-int(n_bow_reps_to_report/2):]
        values = values[:int(n_bow_reps_to_report/2)] + values[-int(n_bow_reps_to_report/2):]

    print(f"Top {int(n_bow_reps_to_report/2)} and bottom {int(n_bow_reps_to_report/2)}")
    print(f"{values[0]} replacement/s appear/s {keys[0]} time/s.")
    fig = tpl.figure()
    fig.barh(list(values), keys)
    fig.show()

def populate_bag_of_reps(args, bag_of_reps, sent_and_positions, reps):
    for sent, token_pos_in_sent, global_token_pos in sent_and_positions:
        for local_pos, global_pos in zip(token_pos_in_sent, global_token_pos):
            single_sent = find_single_sent_around_token(sent, local_pos)
            key = frozenset(reps[global_pos][:args.n_reps])
            value = single_sent
            bag_of_reps[key].append(value)

def find_single_sent_around_token(concated_sents, local_pos):
    full_stop_token = 205
    full_stops_indices = np.where(concated_sents == full_stop_token)[0]
    if len(full_stops_indices) == 0:
        return concated_sents
    end_index = full_stops_indices.searchsorted(local_pos)
    start = full_stops_indices[end_index-1]+1 if end_index != 0 else 0
    if end_index == len(full_stops_indices):
        return concated_sents[start:]
    end = full_stops_indices[end_index]+1
    return concated_sents[start:end]

def print_sents_with_token(args, tokenizer, sent_and_positions, reps, probs):
    for sent, token_pos_in_sent, global_token_pos in sent_and_positions:
        splits = np.split(sent, token_pos_in_sent)
        out = tokenizer.decode(splits[0])
        for i, split in enumerate(splits[1:]):
            if out:
                out += ' '
            out += f"{Color.green}(#{i}) {args.word}{Color.back_to_white} " + tokenizer.decode(split[1:])
        for i, global_pos in enumerate(global_token_pos):
            out += f"\n{Color.orange}(#{i}):"
            for rep, prob in zip(reps[global_pos, :args.n_reps], probs[global_pos, :args.n_reps]):
                out += f" {tokenizer.decode([rep])}-{prob:.5f}"
            out += str(Color.back_to_white)
        print(out)

def find_token_idx_in_row(tokens, token):
    return np.where(tokens == token)[0]

def find_sent_and_positions(token_idx_in_row, tokens, lengths):
    length_sum = 0
    for length in lengths:
        token_location = token_idx_in_row[np.where(np.logical_and(token_idx_in_row >= length_sum, token_idx_in_row < length_sum + length))[0]]
        if len(token_location) > 0:
            yield tokens[length_sum:length_sum + length], token_location-length_sum, token_location
        length_sum += length

def inverted_index(args, token):
    index = json.load(open(args.inverted_index, 'r'))
    return index[str(token)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--replacements_dir", type=str, default="replacements")
    parser.add_argument("--word", type=str, required=True)
    parser.add_argument("--inverted_index", type=str, required=True)
    parser.add_argument("--n_reps", type=int, required=True)
    parser.add_argument("--stop_after_n_matches", type=int, default=None) #TODO
    parser.add_argument("--print", action='store_true')
    parser.add_argument("--report_reps_diversity", action='store_true')
    parser.add_argument("--n_bow_reps_to_report", type=int, default=10, help="How many different replacements to report")
    parser.add_argument("--n_sents_to_print", type=int, default=2, help="Num sents to prin for report_reps_diversity")
    parser.add_argument("--cluster", action='store_true')
    parser.add_argument("--top_n_to_cluster", type=int, default=100)
    parser.add_argument("--n_clusters", type=int)
    parser.add_argument("--distance_threshold", type=int)

    args = parser.parse_args()
    assert args.print or args.report_reps_diversity or args.cluster, \
        "At least one of `print`, `report_reps_diversity` `cluster` should be available"

    if args.cluster:
        assert (args.n_clusters is not None or args.distance_threshold is not None) \
            and (args.n_clusters is None or args.distance_threshold is None), \
            "Pass one of `n_clusters` and `distance_threshold`"

    main(args)