import argparse
from collections import defaultdict, Counter
from enum import Enum
import json
import os
from random import sample, seed

import numpy as np
import termplotlib as tpl
from tqdm import tqdm

from sklearn import cluster as sk_cluster
from sklearn.metrics import pairwise_distances
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

class ClusterFactory():
    @staticmethod
    def make(alg_name, *args, **kwargs):
        alg_name = alg_name.lower()
        if alg_name == 'kmeans': return MyKMeans(*args, **kwargs)
        if alg_name == 'agglomerative_clustering': return MyAgglomerativeClustering(*args, **kwargs)
        if alg_name == 'dbscan': return MyDBSCAN(*args, **kwargs)

    def reps_to_their_clusters(self, clusters, sorted_bag_of_reps):
        clustered_reps = {i: [] for i in self.clusters_range(clusters)}
        for c, rep_with_examples in zip(clusters, sorted_bag_of_reps):
            clustered_reps[c].append(rep_with_examples)

        return clustered_reps

    @staticmethod
    def group_for_display(args, tokenizer, clustered_reps, cluster_sents):
        show_top_n_clusters = args.show_top_n_clusters
        show_top_n_words_per_cluster = args.show_top_n_words_per_cluster
        max_length = max([sum(len(r['examples']) for r in reps) for reps in clustered_reps.values()])
        sorted_zipped = sorted(zip(clustered_reps.values(), cluster_sents), key = lambda x: sum(len(reps['examples']) for reps in x[0]), reverse=True)

        sorted_clustered_reps, sorted_average_sents = zip(*sorted_zipped)
        top_clustered_reps = sorted_clustered_reps[:show_top_n_clusters]
        for i, cluster_reps in enumerate(top_clustered_reps):
            words_in_cluster = Counter()
            for reps in cluster_reps:
                for rep in reps['reps']:
                    words_in_cluster[rep] += len(reps['examples'])
            msg = {'header': f"Cluster {i}",
                   'found': f"Found total {sum(len(reps['examples']) for reps in cluster_reps)} matches"}
            words_in_cluster = words_in_cluster.most_common(show_top_n_words_per_cluster)
            words_in_cluster = [(tokenizer.decode([t]), c) for t, c in words_in_cluster]

            yield words_in_cluster, sorted_average_sents[i], msg
            # keys, values = zip(*words_in_cluster)
            # fig = tpl.figure()
            # fig.barh(values, keys, max_width=int(120*(max(values)/max_length)+1))
            # fig.show()
            # for sent in sorted_average_sents[i]:
            #     print(f"{Color.orange}Example:{Color.back_to_white} {tokenizer.decode(sent)}")
            # print()

        if show_top_n_clusters < len(sorted_clustered_reps):
            print(f"There are additional {len(sorted_clustered_reps) - show_top_n_clusters} that are not displayed.")

class MyKMeans(sk_cluster.KMeans, ClusterFactory):
    def __init__(self, args):
        self.n_clusters = args.n_clusters
        super().__init__(n_clusters=self.n_clusters, random_state=args.seed)

    def representative_sents(self, clusters, sorted_bag_of_reps, distance_matrix, n_sents_to_print):
        cluster_sents = [[] for _ in self.clusters_range(clusters)]
        closest_centers = np.argsort(pairwise_distances(self.cluster_centers_, distance_matrix))
        for i, closest_sents in enumerate(closest_centers):
            for c in closest_sents:
                if clusters[c] == i:
                    cluster_sents[i].append(sorted_bag_of_reps[c]['examples'][0])
                if len(cluster_sents[i]) == n_sents_to_print:
                    break
        return cluster_sents

    def clusters_range(self, clusters):
        return range(self.n_clusters)

class MyAgglomerativeClustering(sk_cluster.AgglomerativeClustering, ClusterFactory):
    def __init__(self, args):
        self.n_clusters = args.n_clusters
        super().__init__(n_clusters=self.n_clusters,
                         distance_threshold=args.distance_threshold,
                         affinity=args.affinity,
                         linkage=args.linkage)

    def representative_sents(self, clusters, sorted_bag_of_reps, distance_matrix, n_sents_to_print):
        cluster_sents = [[] for _ in self.clusters_range(clusters)]
        for i, c in enumerate(clusters):
            if len(cluster_sents[c]) == n_sents_to_print:
                continue
            cluster_sents[c].append(sorted_bag_of_reps[i]['examples'][0])

        return cluster_sents
    
    def clusters_range(self, clusters):
        return range(0, max(clusters)+1)

class MyDBSCAN(sk_cluster.DBSCAN, ClusterFactory):
    def __init__(self, args):
        super().__init__(eps=args.eps, min_samples=args.min_samples)

    def representative_sents(self, clusters, sorted_bag_of_reps, distance_matrix, n_sents_to_print):
        #TODO can I find a way to get the central ones!?
        cluster_sents = {i:[] for i in self.clusters_range(clusters)}
        for i, c in enumerate(clusters):
            if len(cluster_sents[c]) == n_sents_to_print:
                continue
            cluster_sents[c].append(sorted_bag_of_reps[i]['examples'][0])

        return cluster_sents

    def clusters_range(self, clusters):
        return range(min(clusters), max(clusters)+1)

    @staticmethod
    def group_for_display(args, tokenizer, clustered_reps, cluster_sents):
        generators = []
        num_classes_without_outliers = max(clustered_reps.keys())
        non_outlier_clustered_reps = {i: clustered_reps[i] for i in range(num_classes_without_outliers)}
        non_outlier_cluster_sents = [cluster_sents[i] for i in range(num_classes_without_outliers)]
        if len(non_outlier_clustered_reps) > 0:
            generators.append(ClusterFactory.group_for_display(args, tokenizer, non_outlier_clustered_reps, non_outlier_cluster_sents))

        if -1 in clustered_reps:
            outlier_clustered_reps = {0: clustered_reps[-1]}
            outlier_cluster_sents = [cluster_sents[-1]]
            generators.append(ClusterFactory.group_for_display(args, tokenizer, outlier_clustered_reps, outlier_cluster_sents))

        for i, g in enumerate(generators):
            for (msg, words_in_cluster, sents) in g:
                if len(generators) == i+1:
                    msg['header'] = "Outliers Cluster"
                yield (words_in_cluster, sents, msg)

def tokenize(tokenizer, word):
    token = tokenizer.encode(word, add_special_tokens=False)
    if len(token) > 1:
        raise ValueError('Word given is more than a single wordpiece.')
    token = token[0]
    return token

def read_files(args, tokenizer, token, bar=tqdm):
    files_with_pos = read_inverted_index(args, token)
    if args.sample_n_files > 0:
        files_with_pos = sample(files_with_pos, args.sample_n_files)

    n_matches = 0
    bag_of_reps = defaultdict(list)

    for file, token_idx_in_row in bar(files_with_pos):
        data = np.load(os.path.join(args.replacements_dir, f"{file}.npz"))

        tokens = data['tokens']
        lengths = data['sent_lengths']
        reps = data['replacements']

        sent_and_positions = list(find_sent_and_positions(token_idx_in_row, tokens, lengths))

        if args.print:
            probs = data['probs']
            print_sents_with_token(args, tokenizer, sent_and_positions, reps, probs)

        populate_bag_of_reps(args, bag_of_reps, sent_and_positions, reps)

        n_matches += len(token_idx_in_row)

    msg = f"Found Total of {n_matches} Matches in {len(files_with_pos)} Files."
    return bag_of_reps, msg

def cluster(args, bag_of_reps, tokenizer):
    sorted_bag_of_reps = [{'reps': k, 'examples': v} for k, v in sorted(bag_of_reps.items(), key=lambda kv: len(kv[1]), reverse=True)]
    jaccard_matrix = Jaccard().pairwise_distance([x['reps'] for x in sorted_bag_of_reps])

    clustering = ClusterFactory.make(args.cluster_alg, args)
    clusters = clustering.fit_predict(jaccard_matrix)

    clustered_reps = clustering.reps_to_their_clusters(clusters, sorted_bag_of_reps)

    representative_sents = clustering.representative_sents(clusters, sorted_bag_of_reps, jaccard_matrix, args.n_sents_to_print)
    clustering.group_for_display(args, tokenizer, clustered_reps, representative_sents)

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

#TODO deprecated?
def print_sents_with_token(args, tokenizer, sent_and_positions, reps, probs):
    for sent, token_pos_in_sent, global_token_pos in sent_and_positions:
        splits = np.split(sent, token_pos_in_sent)
        out = tokenizer.decode(splits[0])
        for i, split in enumerate(splits[1:]):
            if out:
                out += ' '
            out += f"{Color.green}({i}) {args.word}{Color.back_to_white} " + tokenizer.decode(split[1:])
        for i, global_pos in enumerate(global_token_pos):
            out += f"\n{Color.orange}({i}):"
            for rep, prob in zip(reps[global_pos, :args.n_reps], probs[global_pos, :args.n_reps]):
                out += f" {tokenizer.decode([rep])}-{prob:.5f}"
            out += str(Color.back_to_white)
        print(out)

def find_sent_and_positions(token_idx_in_row, tokens, lengths):
    token_idx_in_row = np.array(token_idx_in_row)
    length_sum = 0
    for length in lengths:
        token_location = token_idx_in_row[np.where(np.logical_and(token_idx_in_row >= length_sum, token_idx_in_row < length_sum + length))[0]]
        if len(token_location) > 0:
            yield tokens[length_sum:length_sum + length], token_location-length_sum, token_location
        length_sum += length

def read_inverted_index(args, token):
    index = json.load(open(args.inverted_index, 'r'))
    if str(token) not in index:
        raise ValueError('token is not in inverted index. Dynamically indexing will be available soon.')
    return index[str(token)]


def prepare_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--replacements_dir", type=str, default="data/replacements")
    parser.add_argument("--word", type=str, default='race')
    parser.add_argument("--inverted_index", type=str, default='data/inverted_index.json')
    parser.add_argument("--n_reps", type=int, default=5)
    parser.add_argument("--sample_n_files", type=int, default=-1)
    parser.add_argument("--print", action='store_true') #TODO deprecated?
    parser.add_argument("--report_reps_diversity", action='store_true')
    parser.add_argument("--n_bow_reps_to_report", type=int, default=10, help="How many different replacements to report")
    parser.add_argument("--n_sents_to_print", type=int, default=2, help="Num sents to print")
    parser.add_argument("--show_top_n_clusters", type=int, default=10)
    parser.add_argument("--show_top_n_words_per_cluster", type=int, default=100)

    parser.add_argument("--cluster_alg", type=str, default=None, choices=['kmeans', 'agglomerative_clustering', 'dbscan'])
    parser.add_argument("--n_clusters", type=int, help="n_clusters, for kmeans and agglomerative_clustering")
    parser.add_argument("--distance_threshold", type=float, help="for agglomerative_clustering")
    parser.add_argument("--affinity", type=str, help="for agglomerative_clustering", default='precomputed')
    parser.add_argument("--linkage", type=str, help="for agglomerative_clustering", default='complete')
    parser.add_argument("--eps", type=float, help="for dbscan")
    parser.add_argument("--min_samples", type=float, help="for dbscan")
    parser.add_argument("--seed", type=int, default=111)
    args = parser.parse_args()

    return args

def assert_arguments(args):
    assert args.print or args.report_reps_diversity or args.cluster_alg is not None, \
        "At least one of `print`, `report_reps_diversity` `cluster` should be available"

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

def main(args):
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', use_fast=True)
    token = tokenize(tokenizer, args.word)

    bag_of_reps = read_files(args, tokenizer, token)

    #TODO deprecated?
    if args.report_reps_diversity:
        print_bag_of_reps(args, bag_of_reps, tokenizer)

    if args.cluster_alg is not None:
        cluster(args, bag_of_reps, tokenizer)

if __name__ == "__main__":
    args = prepare_arguments()
    assert_arguments(args)

    seed(args.seed)
    main(args)