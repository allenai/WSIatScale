from collections import Counter
from itertools import combinations

import community as community_louvain
import networkx as nx
import numpy as np

class CommunityFinder:
    def __init__(self, rep_instances, query_size=100):
        self.query_size = query_size
        self.node2token, self.token2node, self.cooccurrence_matrix = \
            self.create_empty_cooccurrence_matrix(rep_instances)
        self.create_cooccurrence_matrix(rep_instances)

    def create_cooccurrence_matrix(self, rep_instances):
        for rep_inst in rep_instances.data:
            combs = combinations(rep_inst.reps, 2)
            for comb in combs:
                self.update_matrix(comb)

    def update_matrix(self, comb, value=1):
        key0 = self.token2node[comb[0]]
        key1 = self.token2node[comb[1]]

        self.cooccurrence_matrix[key0, key1] += value
        self.cooccurrence_matrix[key1, key0] += value

    def create_empty_cooccurrence_matrix(self, rep_instances):
        node2token = []
        for rep_inst in rep_instances.data:
            for w in rep_inst.reps:
                if w not in node2token:
                    node2token.append(w)
        node2token = sorted(node2token) # for determinism's sake

        mat_size = len(node2token)
        token2node = {k: i for i, k in enumerate(node2token)}
        cooccurrence_matrix = np.zeros((mat_size, mat_size))
        return node2token, token2node, cooccurrence_matrix

    def find(self, method='louvain', resolution=1, seed=111):
        G = nx.from_numpy_matrix(self.cooccurrence_matrix)
        if method == 'louvain':
            best_partition = community_louvain.best_partition(G, random_state=seed, resolution=resolution)
            communities = [[] for _ in range(max(best_partition.values())+1)]
            for n, c in best_partition.items():
                communities[c].append(n)
        elif method == 'leiden':
            from cdlib import algorithms
            communities = algorithms.leiden(G, weights='weight').communities
        else:
            raise "Couldn't find community algorithm"

        return sorted(map(sorted, communities), key=len, reverse=True)

    def argmax_voting(self, communities, rep_instances):
        community_tokens, dist_with_instances = self.voting_distribution(communities, rep_instances)
        communities_sents_data = [[] for c in range(len(communities))]
        communities_dists = [[] for c in range(len(communities))]

        for dist, rep_inst in dist_with_instances:
            argmax = max(dist, key=dist.get)
            communities_sents_data[argmax].append(rep_inst)
            communities_dists[argmax].append(dist)

        community_tokens, communities_sents_data, communities_dists = zip(*sorted(zip(community_tokens, communities_sents_data, communities_dists), key=lambda x: len(x[1]), reverse=True))
        return community_tokens, communities_sents_data, communities_dists

    def voting_distribution(self, communities, rep_instances):
        community_tokens = [[self.node2token[n] for n in comm] for comm in communities]
        token_to_comm = {t: i for i, c in enumerate(community_tokens) for t in c}
        voting_dist = []

        for rep_inst in rep_instances.data:
            counter = Counter([token_to_comm[r] for r in rep_inst.reps[:self.query_size] if r in token_to_comm])
            voting_dist.append((counter, rep_inst))

        return community_tokens, voting_dist

def find_communities_and_vote(rep_instances, query_n_reps, resolution, seed):
    community_finder = CommunityFinder(rep_instances, query_n_reps)
    communities = community_finder.find(resolution=resolution, seed=seed)

    communities_tokens, communities_sents_data, communities_dists = community_finder.argmax_voting(communities, rep_instances)
    # communities_tokens, communities_sents_data = community_finder.merge_small_clusters(communities_tokens,
    #     communities_sents_data,
    #     communities_dists,
    #     minimal_community_proportional_ratio)
    presenting_payload = (community_finder, communities, communities_tokens, communities_dists)
    return communities_sents_data, presenting_payload

def label_by_comms(communities_sents_data, doc_id_to_inst_id):
    lemma_labeling = {}
    for cluster, rep_instances in enumerate(communities_sents_data):
        for rep_inst in rep_instances:
            lemma_inst_id = doc_id_to_inst_id[rep_inst.doc_id]
            lemma_labeling[lemma_inst_id] = cluster
    return lemma_labeling

def label_by_comms_dist(communities_sents_data, communities_dists, doc_id_to_inst_id):
    lemma_labeling = {}
    for rep_instances, dists in zip(communities_sents_data, communities_dists):
        for rep_inst, dist in zip(rep_instances, dists):
            lemma_inst_id = doc_id_to_inst_id[rep_inst.doc_id]
            lemma_labeling[lemma_inst_id] = dist
    return lemma_labeling

    # # Deprecated
    # def prune_infrequent_edges(self):
    #     cutoff = 3
    #     # cutoff = np.sort(self.cooccurrence_matrix.flatten())[-250]
    #     self.cooccurrence_matrix[self.cooccurrence_matrix < cutoff] = 0
    #     zero_rows = self.cooccurrence_matrix.sum(0) == 0

    #     self.cooccurrence_matrix = self.cooccurrence_matrix[~zero_rows][:, ~zero_rows]
    #     self.node2token = [t for t, zero_row in zip(self.node2token, zero_rows) if not zero_row]
    #     self.token2node = {k: i for i, k in enumerate(self.node2token)}

    # # Deprecated
    # def louvain_communities(self, G):
    #     inital_node_partition = None
    #     inital_node_partition = self.clustering_for_initial_partition()
    #     best_partition = community_louvain.best_partition(G, random_state=SEED, partition=inital_node_partition, resolution=1)
    #     communities = [[] for _ in range(max(best_partition.values())+1)]
    #     for n, c in best_partition.items():
    #         communities[c].append(n)

    #     return communities

    # # Deprecated
    # def clustering_for_initial_partition(self):
    #     from WSIatScale.clustering import MyBOWHierarchicalLinkage
    #     clustering_model = MyBOWHierarchicalLinkage()
    #     inital_partition = clustering_model.fit_predict(self.rep_instances)
    #     #TODO <better this>
    #     inital_node_partition, words_clusters_counter = {}, {}
    #     for rep_inst in self.rep_instances.data:
    #         rep_cluster = inital_partition[rep_inst.doc_id]
    #         for r in rep_inst.reps:
    #             if r not in words_clusters_counter:
    #                 words_clusters_counter[r] = Counter()
    #             words_clusters_counter[r][rep_cluster] += 1
    #     for token, counter in words_clusters_counter.items():
    #         inital_node_partition[token] = counter.most_common()[0][0]
    #     #TODO </better this>
    #     inital_node_partition = {self.token2node[k]: v for k, v in inital_node_partition.items()}
    #     return inital_node_partition

    # # Deprecated
    # def jaccard_distribution(self, communities, rep_instances):
    #     community_tokens = [set([self.node2token[c] for c in comm]) for comm in communities]
    #     jaccard = []

    #     for rep_inst in rep_instances.data:
    #         jaccard_score = {}
    #         for i, comm in enumerate(community_tokens):
    #             intersection = comm.intersection(rep_inst.reps)
    #             union = comm.union(rep_inst.reps)
    #             jaccard_score[i] = (len(intersection) / len(union)) * len(comm)

    #         jaccard.append((jaccard_score, rep_inst))

    #     return community_tokens, jaccard

    # # Deprecated
    # def voting_distribution_with_probabilities(self, communities, rep_instances):
    #     community_tokens = [[self.node2token[c] for c in comm] for comm in communities]
    #     token_to_comm = {t: i for i, c in enumerate(community_tokens) for t in c}
    #     dist_with_instances = []

    #     for rep_inst in rep_instances.data:
    #         voting_dist = {}
    #         for r, p in zip(rep_inst.reps, rep_inst.probs):
    #             sense = token_to_comm[r]
    #             if sense not in voting_dist:
    #                 voting_dist[sense] = 0
    #             voting_dist[sense] += p

    #         dist_with_instances.append((voting_dist, rep_inst))

    #     return community_tokens, dist_with_instances


    # # Deprecated
    # def merge_small_clusters(self, community_tokens, communities_sents_data, communities_dists, minimal_community_proportional_ratio):
    #     community_tokens_without_small_clusters, communities_sents_data_without_small_clusters = [], []
    #     number_of_instances = sum(len(sents) for sents in communities_sents_data)
    #     community_ids_added = {}
    #     for i, (comm_tokens, comm_sents_data, coms_dist) in enumerate(zip(community_tokens, communities_sents_data, communities_dists)):
    #         if i == 0 or len(comm_sents_data) > number_of_instances * minimal_community_proportional_ratio:
    #             community_tokens_without_small_clusters.append(comm_tokens)
    #             communities_sents_data_without_small_clusters.append(comm_sents_data)
    #             community_ids_added[coms_dist[0].most_common(1)[0][0]] = i
    #         else:
    #             for sent_data, dist in zip(comm_sents_data, coms_dist):
    #                 for com_idx, _ in dist.most_common():
    #                     if com_idx in community_ids_added:
    #                         communities_sents_data_without_small_clusters[community_ids_added[com_idx]].append(sent_data)
    #                         break

    #     return community_tokens_without_small_clusters, communities_sents_data_without_small_clusters