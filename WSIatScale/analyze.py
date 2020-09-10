import argparse
from collections import defaultdict, Counter
from enum import Enum
from itertools import combinations
import json
import os
import random

from efficient_apriori import apriori, itemsets_from_transactions
import numpy as np
from tqdm import tqdm

from networkx.algorithms import community
import networkx as nx
import community as community_louvain

from sklearn import cluster as sk_cluster
from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids

SEED = 111

MAX_REPS = 100
class RepsToInstances:
    def __init__(self):
        self.data = defaultdict(list)

    def populate(self, sent_and_positions, reps):
        for sent, token_pos_in_sent, global_token_pos in sent_and_positions:
            for local_pos, global_pos in zip(token_pos_in_sent, global_token_pos):
                single_sent = self.find_single_sent_around_token(sent, local_pos)
                key = tuple(reps[global_pos])
                value = single_sent
                self.data[key].append(value)

    def populate_specific_size(self, n_reps):
        if n_reps == MAX_REPS:
            return self

        reps_to_instances = RepsToInstances()
        for key, sents in self.data.items():
            new_key = key[:n_reps]
            reps_to_instances.data[new_key].extend(sents)

        return reps_to_instances

    def remove_query_word(self, tokenizer, word, merge_same_keys):
        similar_words = [word.lower().lstrip(), f" {word.lower().lstrip()}", word.lower().title().lstrip(), f" {word.title().lstrip()}"]
        similar_tokens = []
        for w in similar_words:
            t = tokenizer.encode(w, add_special_tokens=False)
            if len(t) == 1:
                similar_tokens.append(t[0])

        reps_to_instances = RepsToInstances()
        for key, sents in self.data.items():
            new_key = tuple(t for t in key if t not in similar_tokens)
            if merge_same_keys:
                new_key = tuple(sorted(new_key))
            reps_to_instances.data[new_key].extend(sents)

        self.data = reps_to_instances.data

    @staticmethod
    def find_single_sent_around_token(concated_sents, local_pos):
        def full_stop_index_hack():
            if concated_sents[0] == 0: #roberta
                return 4
            else:
                return 205
        full_stop_index = full_stop_index_hack()
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
        x, y = set(x), set(y)
        intersection = len(x.intersection(y))
        union = len(x) + len(y) - intersection
        return float(intersection) / union

class ClusterFactory():
    @staticmethod
    def make(alg_name, *args, **kwargs):
        alg_name = alg_name.lower()
        if alg_name == 'kmeans': return MyKMeans(*args, **kwargs)
        if alg_name == 'kmedoids': return MyKMedoids(*args, **kwargs)
        if alg_name == 'agglomerative_clustering': return MyAgglomerativeClustering(*args, **kwargs)
        if alg_name == 'dbscan': return MyDBSCAN(*args, **kwargs)

    def reps_to_their_clusters(self, clusters, sorted_reps_to_instances_data):
        clustered_reps = {i: [] for i in self.clusters_range(clusters)}
        for c, rep_with_examples in zip(clusters, sorted_reps_to_instances_data):
            clustered_reps[c].append(rep_with_examples)

        return clustered_reps

    @staticmethod
    def group_for_display(args, tokenizer, clustered_reps, cluster_sents):
        show_top_n_clusters = args.show_top_n_clusters
        show_top_n_words_per_cluster = args.show_top_n_words_per_cluster
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

        if show_top_n_clusters < len(sorted_clustered_reps):
            msg = {'header': f"There are additional {len(sorted_clustered_reps) - show_top_n_clusters} that are not displayed.",
                   'found': ''}
            yield None, None, msg

class MyKMeans(sk_cluster.KMeans, ClusterFactory):
    def __init__(self, args):
        self.n_clusters = args.n_clusters
        super().__init__(n_clusters=self.n_clusters, random_state=SEED)

    def representative_sents(self, clusters, sorted_reps_to_instances_data, distance_matrix, n_sents_to_print):
        cluster_sents = [[] for _ in self.clusters_range(clusters)]
        closest_centers = np.argsort(pairwise_distances(self.cluster_centers_, distance_matrix))
        for i, closest_sents in enumerate(closest_centers):
            for c in closest_sents:
                if clusters[c] == i:
                    cluster_sents[i].append(sorted_reps_to_instances_data[c]['examples'][0])
                if len(cluster_sents[i]) == n_sents_to_print:
                    break
        return cluster_sents

    def clusters_range(self, clusters):
        return range(self.n_clusters)

    def fit_predict(self, X):
        X = 1 - X #Kmeans expects similarity and not distsance matrix
        return super().fit_predict(X)

class MyKMedoids(KMedoids, ClusterFactory):
    def __init__(self, args):
        self.n_clusters = args.n_clusters
        super().__init__(n_clusters=self.n_clusters, random_state=SEED)

    def representative_sents(self, clusters, sorted_reps_to_instances_data, distance_matrix, n_sents_to_print):
        cluster_sents = [[] for _ in self.clusters_range(clusters)]
        closest_centers = np.argsort(pairwise_distances(self.cluster_centers_, distance_matrix))
        for i, closest_sents in enumerate(closest_centers):
            for c in closest_sents:
                if clusters[c] == i:
                    cluster_sents[i].append(sorted_reps_to_instances_data[c]['examples'][0])
                if len(cluster_sents[i]) == n_sents_to_print:
                    break
        return cluster_sents

    def clusters_range(self, clusters):
        return range(self.n_clusters)

    def fit_predict(self, X):
        X = 1 - X
        return super().fit_predict(X)

class MyAgglomerativeClustering(sk_cluster.AgglomerativeClustering, ClusterFactory):
    def __init__(self, args):
        self.n_clusters = args.n_clusters
        self.affinity = args.affinity
        super().__init__(n_clusters=self.n_clusters,
                         distance_threshold=args.distance_threshold,
                         affinity=args.affinity,
                         linkage=args.linkage)

    def representative_sents(self, clusters, sorted_reps_to_instances_data, _, n_sents_to_print):
        cluster_sents = [[] for _ in self.clusters_range(clusters)]
        for i, c in enumerate(clusters):
            if len(cluster_sents[c]) == n_sents_to_print:
                continue
            cluster_sents[c].append(sorted_reps_to_instances_data[i]['examples'][0])

        return cluster_sents

    def clusters_range(self, clusters):
        return range(0, max(clusters)+1)

    def fit_predict(self, X):
        if self.affinity != 'precomputed':
            #"If "precomputed", a distance matrix (instead of a similarity matrix) is needed as input for the fit method."
            X = 1 - X
        return super().fit_predict(X)

class MyDBSCAN(sk_cluster.DBSCAN, ClusterFactory):
    def __init__(self, args):
        super().__init__(eps=args.eps, min_samples=args.min_samples)

    def representative_sents(self, clusters, sorted_reps_to_instances_data, _, n_sents_to_print):
        #TODO can I find a way to get the central ones!?
        cluster_sents = {i:[] for i in self.clusters_range(clusters)}
        for i, c in enumerate(clusters):
            if len(cluster_sents[c]) == n_sents_to_print:
                continue
            cluster_sents[c].append(sorted_reps_to_instances_data[i]['examples'][0])

        return cluster_sents

    def clusters_range(self, clusters):
        return range(min(clusters), max(clusters)+1)

    @staticmethod
    def group_for_display(args, tokenizer, clustered_reps, cluster_sents):
        num_classes_without_outliers = max(clustered_reps.keys()) + 1
        non_outlier_clustered_reps = {i: clustered_reps[i] for i in range(num_classes_without_outliers)}
        non_outlier_cluster_sents = [cluster_sents[i] for i in range(num_classes_without_outliers)]
        if len(non_outlier_clustered_reps) > 0:
            generator = ClusterFactory.group_for_display(args, tokenizer, non_outlier_clustered_reps, non_outlier_cluster_sents)
            for (words_in_cluster, sents, msg) in generator:
                yield (words_in_cluster, sents, msg)

        if -1 in clustered_reps:
            outlier_clustered_reps = {0: clustered_reps[-1]}
            outlier_cluster_sents = [cluster_sents[-1]]
            generator = ClusterFactory.group_for_display(args, tokenizer, outlier_clustered_reps, outlier_cluster_sents)

            for (words_in_cluster, sents, msg) in generator:
                msg['header'] = "Outliers Cluster"
                yield (words_in_cluster, sents, msg)

def tokenize(tokenizer, word):
    token = tokenizer.encode(word, add_special_tokens=False)
    if len(token) > 1:
        raise ValueError('Word given is more than a single wordpiece.')
    token = token[0]
    return token

def read_files(token, replacements_dir, inverted_index, sample_n_files, bar=tqdm):
    files_with_pos = read_inverted_index(inverted_index, token)
    if sample_n_files > 0 and len(files_with_pos) > sample_n_files:
        random.seed(SEED)
        files_with_pos = random.sample(files_with_pos, sample_n_files)

    n_matches = 0
    reps_to_instances = RepsToInstances()

    for file, token_idx_in_row in bar(files_with_pos):
        data = np.load(os.path.join(replacements_dir, f"{file}.npz"))

        tokens = data['tokens']
        lengths = data['sent_lengths']
        reps = data['replacements']

        sent_and_positions = list(find_sent_and_positions(token_idx_in_row, tokens, lengths))

        reps_to_instances.populate(sent_and_positions, reps)

        n_matches += len(token_idx_in_row)

    msg = f"Found Total of {n_matches} Matches in {len(files_with_pos)} Files."
    return reps_to_instances, msg

def cluster(args, reps_to_instances, tokenizer):
    sorted_reps_to_instances_data = [{'reps': k, 'examples': v} for k, v in sorted(reps_to_instances.data.items(), key=lambda kv: len(kv[1]), reverse=True)]
    jaccard_matrix = Jaccard().pairwise_distance([x['reps'] for x in sorted_reps_to_instances_data])

    clustering = ClusterFactory.make(args.cluster_alg, args)
    clusters = clustering.fit_predict(jaccard_matrix)

    clustered_reps = clustering.reps_to_their_clusters(clusters, sorted_reps_to_instances_data)

    representative_sents = clustering.representative_sents(clusters, sorted_reps_to_instances_data, jaccard_matrix, args.n_sents_to_print)
    clustering.group_for_display(args, tokenizer, clustered_reps, representative_sents)

def cluster_words(args, reps_to_instances, tokenizer):
    sorted_reps_to_instances_data = [{'reps': k, 'examples': v} for k, v in sorted(reps_to_instances.data.items(), key=lambda kv: len(kv[1]), reverse=True)]
    jaccard_matrix = Jaccard().pairwise_distance([x['reps'] for x in sorted_reps_to_instances_data])

    clustering = ClusterFactory.make(args.cluster_alg, args)
    clusters = clustering.fit_predict(jaccard_matrix)

    clustered_reps = clustering.reps_to_their_clusters(clusters, sorted_reps_to_instances_data)

    representative_sents = clustering.representative_sents(clusters, sorted_reps_to_instances_data, jaccard_matrix, args.n_sents_to_print)
    clustering.group_for_display(args, tokenizer, clustered_reps, representative_sents)

class CommunityFinder:
    def __init__(self, reps_to_instances):
        self.create_cooccurrence_matrix(reps_to_instances)

    def create_cooccurrence_matrix(self, reps_to_instances):
        self.create_empty_cooccurrence_matrix(reps_to_instances)

        for reps, sents in reps_to_instances.data.items():
            combs = combinations(reps, 2)
            for comb in combs:
                self.update_matrix(comb, len(sents))

    def update_matrix(self, comb, value):
        key0 = self.token2node[comb[0]]
        key1 = self.token2node[comb[1]]

        self.cooccurrence_matrix[key0, key1] += value
        self.cooccurrence_matrix[key1, key0] += value

    def create_empty_cooccurrence_matrix(self, reps_to_instances):
        node2token = []
        for reps in reps_to_instances.data.keys():
            for w in reps:
                if w not in node2token:
                    node2token.append(w)

        mat_size = len(node2token)
        self.node2token = node2token
        self.token2node = {k: i for i, k in enumerate(self.node2token)}
        self.cooccurrence_matrix = np.zeros((mat_size, mat_size))

    def find(self, method='girvan_newman', top_n_nodes_to_keep=None):
        G = nx.from_numpy_matrix(self.cooccurrence_matrix)
        if top_n_nodes_to_keep:
            cutoff_degree = sorted(dict(G.degree()).values())[-top_n_nodes_to_keep] # pylint: disable=invalid-unary-operand-type
            nodes_with_lower_degree = [node for node, degree in dict(G.degree()).items() if degree < cutoff_degree]
            G.remove_nodes_from(nodes_with_lower_degree)
        if method == 'Girvan-Newman':
            communities_generator = community.girvan_newman(G)
            communities = next(communities_generator)
        if method == 'Weighted Girvan-Newman':
            from networkx import edge_betweenness_centrality as betweenness
            def most_central_edge(G):
                centrality = betweenness(G, weight="weight")
                return max(centrality, key=centrality.get)

            communities_generator = community.girvan_newman(G, most_valuable_edge=most_central_edge)
            communities = next(communities_generator)
        elif method == 'async LPA':
            communities = list(community.label_propagation.asyn_lpa_communities(G, seed=SEED))
        elif method == 'Weighted async LPA':
            communities = list(community.label_propagation.asyn_lpa_communities(G, weight='weight', seed=SEED))
        elif method == 'Clauset-Newman-Moore (no weights)':
            communities = list(community.modularity_max.greedy_modularity_communities(G))
        elif method == 'Louvain':
            partition = community_louvain.best_partition(G, random_state=SEED)
            communities = [[] for _ in range(max(partition.values())+1)]
            for n, c in partition.items():
                communities[c].append(n)

        return sorted(map(sorted, communities), key=len, reverse=True)

    def community_tokens_with_sents(self, communities, reps_to_instances):
        community_tokens = [[self.node2token[c] for c in comm] for comm in communities]
        token_to_comm = {t: i for i, c in enumerate(community_tokens) for t in c}
        communities_sents = [[] for c in range(len(communities))]

        for reps, sents in reps_to_instances.data.items():
            counter = Counter(map(lambda r: token_to_comm[r], reps))
            argmax = counter.most_common()[0][0]
            communities_sents[argmax].extend(sents)

        community_tokens, communities_sents = zip(*sorted(zip(community_tokens, communities_sents), key=lambda x: len(x[1]), reverse=True))
        return community_tokens, communities_sents

def find_sent_and_positions(token_idx_in_row, tokens, lengths):
    token_idx_in_row = np.array(token_idx_in_row)
    length_sum = 0
    for length in lengths:
        token_location = token_idx_in_row[np.where(np.logical_and(token_idx_in_row >= length_sum, token_idx_in_row < length_sum + length))[0]]
        if len(token_location) > 0:
            yield tokens[length_sum:length_sum + length], token_location-length_sum, token_location
        length_sum += length

def read_inverted_index(inverted_index, token):
    inverted_index_file = os.path.join(inverted_index, f"{token}.txt")
    if not os.path.exists(inverted_index_file):
        raise ValueError('token is not in inverted index')
    index = {}
    with open(inverted_index_file, 'r') as f:
        for line in f:
            index.update(json.loads(line))
    return [[k, v] for k, v in index.items()] #TODO keep in dictionary form

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

def run_apriori(reps_to_instances, min_support):
    keys = [k for k, v in reps_to_instances.data.items() for _ in range(len(v))]
    itemsets, _ = itemsets_from_transactions(keys, min_support=min_support, output_transaction_ids=True)
    return itemsets

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