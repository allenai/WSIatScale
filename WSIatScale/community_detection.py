from collections import Counter
import numpy as np

from networkx.algorithms import community
import networkx as nx
import community as community_louvain
from itertools import combinations

SEED = 111

class CommunityFinder:
    def __init__(self, reps_to_instances):
        self.node2token, self.token2node, self.cooccurrence_matrix = \
            self.create_empty_cooccurrence_matrix(reps_to_instances)
        self.create_cooccurrence_matrix(reps_to_instances)

    def create_cooccurrence_matrix(self, reps_to_instances):
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
        token2node = {k: i for i, k in enumerate(node2token)}
        cooccurrence_matrix = np.zeros((mat_size, mat_size))

        return node2token, token2node, cooccurrence_matrix

    def find(self, method='Louvain', top_n_nodes_to_keep=None):
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
            best_partition = community_louvain.best_partition(G, random_state=SEED)
            communities = [[] for _ in range(max(best_partition.values())+1)]
            for n, c in best_partition.items():
                communities[c].append(n)

        return sorted(map(sorted, communities), key=len, reverse=True)

    def argmax_voting(self, communities, reps_to_instances):
        community_tokens, voting_dist = self.voting_distribution(communities, reps_to_instances)
        communities_sents_data = [[] for c in range(len(communities))]

        for counter, inst, reps in voting_dist:
            argmax = counter.most_common()[0][0]
            communities_sents_data[argmax].append((inst, reps))

        community_tokens, communities_sents_data = zip(*sorted(zip(community_tokens, communities_sents_data), key=lambda x: len(x[1]), reverse=True))
        return community_tokens, communities_sents_data

    def voting_distribution(self, communities, reps_to_instances):
        community_tokens = [[self.node2token[c] for c in comm] for comm in communities]
        token_to_comm = {t: i for i, c in enumerate(community_tokens) for t in c}
        voting_dist = []

        for reps, instances in reps_to_instances.data.items():
            counter = Counter(map(lambda r: token_to_comm[r], reps))
            for inst in instances:
                voting_dist.append((counter, inst, reps))

        return community_tokens, voting_dist