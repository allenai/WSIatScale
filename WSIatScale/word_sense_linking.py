# pylint: disable=import-error, no-name-in-module
from typing import List, Dict
import heapq
from collections import Counter
from copy import deepcopy
from itertools import product, combinations
from functools import lru_cache
from dataclasses import dataclass
import numpy as np

from utils.utils import jaccard_score_between_elements
from WSIatScale.analyze import tokenize
from WSIatScale.cluster_reps_per_token import read_clustering_data

def infer_senses_by_list(tokenizer, data_dir, method, n_reps, words, cluster_reps_to_use, semi_greedy_search):
    tokens = [tokenize(tokenizer, w) for w in words]
    token_clusters = read_clusters_for_multiple_tokens(data_dir, tokens, method, n_reps)

    return find_top_matches(token_clusters, cluster_reps_to_use, semi_greedy_search)

def find_top_matches(token_clusters, cluster_reps_to_use, semi_greedy_search):
    if semi_greedy_search:
        most_fitting_clusters = find_top_matches_semi_greedy_search(token_clusters, cluster_reps_to_use)
        return most_fitting_clusters
    else:
        
        flat_token_clusters_names, jaccard_sims_heap = brute_force_find_top_matches(token_clusters, cluster_reps_to_use)
        return flat_token_clusters_names, jaccard_sims_heap

@dataclass
class HeapElement:
    words: List[str]
    cluster_ids: List[int]
    tokens_counter: Dict[int, int]

    def update(self, word, cluster_id, cluster_tokens):
        self.words.append(word)
        self.cluster_ids.append(cluster_id)
        for t in cluster_tokens:
            self.tokens_counter[t] += 1

        return self

    def __lt__(self, other):
        return False

    def jaccard_with_a_counter(self, other):
        intersection = set(self.tokens_counter.keys()).intersection(other)
        intersection_len = sum([self.tokens_counter[v] for v in intersection])
        union_len = len(set(self.tokens_counter.keys()).union(other))
        return intersection_len / union_len

# from utils.utils import timeit
# @timeit
def read_clusters_for_multiple_tokens(data_dir, tokens, method, n_reps):
    token_clusters = {}

    for token in tokens:
        token_clusters[token] = {}
        clustering_data = read_clustering_data(data_dir, token)[method][str(n_reps)]
        cluster_idxs = list(range(len(clustering_data)))

        # I think there was I reason I didn't use zip. Need to check.
        for cluster_idx, cluster in zip(cluster_idxs, clustering_data):
            # token_clusters[word][cluster_idx] = [token] + [r for r, count in cluster]
            token_clusters[token][cluster_idx] = [token] + [r for r, count in cluster if r != 1103] #REMOVE

    return token_clusters

# from utils.utils import timeit
# @timeit
def find_top_matches_semi_greedy_search(token_clusters, cluster_reps_to_use):
    heap_size = 100
    n_closest_comms_to_save = 3

    best_overall_jaccard = []
    for token, clusters in token_clusters.items():
        if not best_overall_jaccard:
            for cluster_id, cluster_tokens in clusters.items():
                heapq.heappush(best_overall_jaccard, (0, HeapElement([token], [cluster_id], Counter(cluster_tokens[:cluster_reps_to_use]))))
        else:
            best_jaccards_with_new_word = []
            min_score_in_heap = 0
            for curr_jaccard_score, heap_element in heapq.nlargest(len(best_overall_jaccard), best_overall_jaccard):
                for cluster_id, cluster_tokens in clusters.items():
                    aggregated_score = curr_jaccard_score
                    aggregated_score += np.log(heap_element.jaccard_with_a_counter(cluster_tokens[:cluster_reps_to_use]) + 0.001)

                    if len(best_jaccards_with_new_word) < heap_size or aggregated_score > min_score_in_heap:
                        copied_heap_element = deepcopy(heap_element)
                        if len(best_jaccards_with_new_word) < heap_size:
                            heapq.heappush(best_jaccards_with_new_word,
                                (aggregated_score,
                                copied_heap_element.update(token, cluster_id, set(cluster_tokens[:cluster_reps_to_use]))))
                        else:
                            heapq.heappushpop(best_jaccards_with_new_word,
                                (aggregated_score,
                                copied_heap_element.update(token, cluster_id, set(cluster_tokens[:cluster_reps_to_use]))))
                        min_score_in_heap = heapq.nsmallest(1, best_jaccards_with_new_word)[0][0]
            best_overall_jaccard = best_jaccards_with_new_word

    best_overall_jaccard = [heapq.heappop(best_overall_jaccard) for _ in range(len(best_overall_jaccard))]
    best_overall_jaccard.reverse()

    return best_overall_jaccard[:n_closest_comms_to_save]

# Deprecated
# from utils.utils import timeit
# @timeit
def brute_force_find_top_matches(token_clusters, cluster_reps_to_use):
    flat_token_clusters_dict, flat_token_clusters_names, sense_counts = flatten_token_clusters(token_clusters, cluster_reps_to_use)

    sense_positions = create_sense_positions(list(token_clusters.keys()), sense_counts)

    n_closest_comms_to_save = 3
    @lru_cache(maxsize=None)
    def cached_jaccard_scores(i, j):
        return jaccard_score_between_elements(flat_token_clusters_dict[i], flat_token_clusters_dict[j])

    jaccard_sims_heap = []
    for sense_indices in product(*sense_positions.values()):
        aggregated_jaccard_score = 0
        for i, j in combinations(sense_indices, 2):
            score = cached_jaccard_scores(i, j)
            # aggregated_jaccard_score += score
            if score == 0:
                aggregated_jaccard_score += -10
            else:
                aggregated_jaccard_score += np.log(score)
        if len(jaccard_sims_heap) < n_closest_comms_to_save:
            heapq.heappush(jaccard_sims_heap, (aggregated_jaccard_score, sense_indices))
        else:
            heapq.heappushpop(jaccard_sims_heap, (aggregated_jaccard_score, sense_indices))

    jaccard_sims_heap = [heapq.heappop(jaccard_sims_heap) for _ in range(len(jaccard_sims_heap))]
    jaccard_sims_heap.reverse()

    return flat_token_clusters_names, jaccard_sims_heap

def flatten_token_clusters(token_clusters, cluster_reps_to_use):
    sense_counts = {w: len(cluster) for w, cluster in token_clusters.items()}

    flat_token_clusters_dict = []
    flat_token_clusters_names = []
    for word, clusters in token_clusters.items():
        for cluster_id, cluster in clusters.items():
            flat_token_clusters_dict.append(set(cluster[:cluster_reps_to_use]))
            flat_token_clusters_names.append((word, cluster_id))

    return flat_token_clusters_dict, flat_token_clusters_names, sense_counts

# from utils.utils import timeit
# @timeit
def create_sense_positions(tokens, sense_counts):
    sense_positions = {tokens[0]: list(range(sense_counts[tokens[0]]))}
    for i, w in enumerate(tokens[1:]):
        prev_count = sense_positions[tokens[i]][-1]
        sense_positions[w] = list(range(prev_count+1, prev_count+1+sense_counts[w]))

    return sense_positions