# pylint: disable=import-error
import os
import altair as alt
from collections import Counter
import heapq
import streamlit as st
from annotated_text import annotated_text
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import tokenizers
import sys
from functools import lru_cache
from requests import get
from itertools import product, combinations
from dataclasses import dataclass
from typing import List, Dict
from copy import deepcopy

sys.path.append('/home/matane/matan/dev/WSIatScale')

from WSIatScale.analyze import tokenize, npy_file_path, RepInstances
from WSIatScale.cluster_reps_per_token import read_clustering_data
from WSIatScale.assign_clusters_to_tokens import SENTS_BY_CLUSTER
from WSIatScale.look_for_similar_communities import read_close_communities
from utils.utils import SpecialTokens, jaccard_score_between_elements

@st.cache(hash_funcs={tokenizers.Tokenizer: id}, suppress_st_warning=True, allow_output_mutation=True)
def cached_tokenizer(model_hf_path):
    tokenizer = AutoTokenizer.from_pretrained(model_hf_path, use_fast=True)
    return tokenizer

@st.cache(hash_funcs={tokenizers.Tokenizer: id}, suppress_st_warning=True, allow_output_mutation=True)
def cached_read_clustering_data_of_word(tokenizer, data_dir, word):
    token = tokenize(tokenizer, word)
    return read_clustering_data(data_dir, token)

EXTERNAL_IP = get('https://api.ipify.org').text

def main():
    app_state = st.experimental_get_query_params()
    app_state = {k: v[0] if isinstance(v, list) else v for k, v in app_state.items()}

    st.title('WSI at Scale')
    single_word_app = ['WSI at Scale', 'List Expansion']
    non_single_word_app = ['IE by Sense', 'Infer Senses by List']
    app_str = st.sidebar.radio(
        "Choose Application", single_word_app + non_single_word_app, app_index(app_str_format_func(app_state.get('app')))
        )

    if app_str != app_str_format_func(app_state.get("app")):
        app_state["app"] = app_str_format_func(app_str)
        st.experimental_set_query_params(**app_state)

    dataset = 'Wikipedia'
    model_hf_path, data_dir, example_word, special_tokens = dataset_configs(dataset)
    tokenizer = cached_tokenizer(model_hf_path)

    if app_str in single_word_app:
        word = st.sidebar.text_input('Word to disambiguate', app_state.get('word') or example_word)
        if word != app_state.get("word"):
            app_state["word"] = word
            st.experimental_set_query_params(**app_state)

    method = method_format_func(st.sidebar.selectbox('Method', ('Agglomerative Clustering', 'Community Detection'), 1))
    n_reps = st.sidebar.select_slider(f"Number of replacements", options=[5, 20, 50])

    if app_str != 'List Expansion':
        num_sent_to_print = st.sidebar.slider(f"Number of cluster sentences to present", 1, 100, 3)

    if app_str == 'WSI at Scale':
        cluster_reps_to_present = st.sidebar.slider(f"Number of cluster replacements to present", 1, 100, 10)
        wsi_at_scale(tokenizer, data_dir, word, method, n_reps, cluster_reps_to_present, special_tokens, num_sent_to_print)
    elif app_str == 'List Expansion':
        list_expansion(tokenizer, data_dir, word, method, n_reps)
    elif app_str == 'IE by Sense':
        sense_ie(app_state, tokenizer, data_dir, method, n_reps, special_tokens, num_sent_to_print)
    elif app_str == 'Infer Senses by List':
        cluster_reps_to_use = st.sidebar.slider(f"Number of cluster replacements to use", 1, 100, 10)
        infer_senses_by_list(tokenizer, data_dir, method, n_reps, cluster_reps_to_use)

def infer_senses_by_list(tokenizer, data_dir, method, n_reps, cluster_reps_to_use):
    words = st.text_input("Word Senses to infer, comma seperated.", "bass, grunt, bleak, drum")
    words = words.replace(" ", "").split(',')

    word_clusters = read_clusters_for_multiple_words(tokenizer, data_dir, words, method, n_reps)

    # flat_word_clusters_names, jaccard_sims_heap = find_top_matches(words, word_clusters, cluster_reps_to_use)
    # print_top_matches(tokenizer, data_dir, method, n_reps, cluster_reps_to_use, jaccard_sims_heap, flat_word_clusters_names)

    most_fitting_clusters = find_top_matches_beam_search(word_clusters, cluster_reps_to_use)
    print_top_matches(tokenizer, data_dir, method, n_reps, cluster_reps_to_use, most_fitting_clusters)

from utils.utils import timeit
@timeit
def print_top_matches(tokenizer, data_dir, method, n_reps, cluster_reps_to_use, most_fitting_clusters):
    for jaccard_score, heap_element in most_fitting_clusters:
        out_str = []
        for word, cluster_id in zip(heap_element.words, heap_element.cluster_ids):
            curr_out = f"*{word}-{cluster_id}* "
            top_reps = top_n_reps(tokenizer, data_dir, word, cluster_id, method, n_reps, cluster_reps_to_use)

            curr_out += f"({', '.join(top_reps)})"
            curr_out += '\n'
            out_str.append(curr_out)
            group_name = tokenizer.decode([heap_element.tokens_counter.most_common(1)[0][0]])
        out_str = [f'**Possible Name <span style="color:red">{group_name.title()}</span> ({str(round(jaccard_score, 3))}):**\n'] + out_str
        st.write("\n".join(out_str), unsafe_allow_html=True)

from utils.utils import timeit
@timeit
def create_sense_positions(words, sense_counts):
    sense_positions = {words[0]: list(range(sense_counts[words[0]]))}
    for i, w in enumerate(words[1:]):
        prev_count = sense_positions[words[i]][-1]
        sense_positions[w] = list(range(prev_count+1, prev_count+1+sense_counts[w]))

    return sense_positions

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

from utils.utils import timeit
@timeit
def find_top_matches_beam_search(word_clusters, cluster_reps_to_use):
    beam_size = 100
    # beam_size = 3 #REMOVE
    n_closest_comms_to_save = 3

    best_overall_jaccard = []
    for word, clusters in word_clusters.items():
        if not best_overall_jaccard:
            for cluster_id, cluster_tokens in clusters.items():
                heapq.heappush(best_overall_jaccard, (0, HeapElement([word], [cluster_id], Counter(cluster_tokens[:cluster_reps_to_use]))))
        else:
            best_jaccards_with_new_word = []
            min_score_in_heap = 0
            for curr_jaccard_score, heap_element in heapq.nlargest(len(best_overall_jaccard), best_overall_jaccard):
                for cluster_id, cluster_tokens in clusters.items():
                    aggregated_score = curr_jaccard_score
                    aggregated_score += np.log(heap_element.jaccard_with_a_counter(cluster_tokens[:cluster_reps_to_use]) + 0.001)

                    # if word == 'key':
                    if len(best_jaccards_with_new_word) < beam_size or aggregated_score > min_score_in_heap:
                        copied_heap_element = deepcopy(heap_element)
                        if len(best_jaccards_with_new_word) < beam_size:
                            heapq.heappush(best_jaccards_with_new_word,
                                (aggregated_score,
                                copied_heap_element.update(word, cluster_id, set(cluster_tokens[:cluster_reps_to_use]))))
                        else:
                            heapq.heappushpop(best_jaccards_with_new_word,
                                (aggregated_score,
                                copied_heap_element.update(word, cluster_id, set(cluster_tokens[:cluster_reps_to_use]))))
                        min_score_in_heap = heapq.nsmallest(1, best_jaccards_with_new_word)[0][0]
            best_overall_jaccard = best_jaccards_with_new_word

    best_overall_jaccard = [heapq.heappop(best_overall_jaccard) for _ in range(len(best_overall_jaccard))]
    best_overall_jaccard.reverse()

    return best_overall_jaccard[:n_closest_comms_to_save]

from utils.utils import timeit
@timeit
def read_clusters_for_multiple_words(tokenizer, data_dir, words, method, n_reps):
    word_clusters = {}

    for word in words:
        token = tokenize(tokenizer, word)
        word_clusters[word] = {}
        clustering_data = cached_read_clustering_data_of_word(tokenizer, data_dir, word)[method][str(n_reps)]
        cluster_idxs = list(range(len(clustering_data)))

        for cluster_idx, cluster in zip(cluster_idxs, clustering_data):
            word_clusters[word][cluster_idx] = [token] + [r for r, count in cluster]

    return word_clusters

def flatten_word_clusters(word_clusters, cluster_reps_to_use):
    sense_counts = {w: len(cluster) for w, cluster in word_clusters.items()}

    flat_word_clusters_dict = []
    flat_word_clusters_names = []
    for word, clusters in word_clusters.items():
        for cluster_id, cluster in clusters.items():
            flat_word_clusters_dict.append(set(cluster[:cluster_reps_to_use]))
            flat_word_clusters_names.append(f'{word}-{cluster_id}')

    return flat_word_clusters_dict, flat_word_clusters_names, sense_counts

def sense_ie(app_state, tokenizer, data_dir, method, n_reps, special_tokens, num_sent_to_print):
    senses = st.text_input("Senses to search for", app_state.get('senses') or 'bass-2, grunt-1, bleak-2, drum-3')
    if senses != app_state.get("senses"):
        app_state["senses"] = senses
        st.experimental_set_query_params(**app_state)
    senses = senses.split(',')
    for sense in senses:
        word_sense = sense.split('-')
        if len(word_sense) == 1:
            st.write(f'Please choose sense for word {word_sense[0]}')
            return

    for sense in senses:
        word, cluster_idx = sense.split('-')
        token = tokenize(tokenizer, word)
        with st.beta_expander(f"{sense} Matches"):
            present_example_sents(data_dir, token, method, n_reps, cluster_idx, special_tokens, tokenizer, num_sent_to_print)

def list_expansion(tokenizer, data_dir, word, method, n_reps):
    global COMMS_ALREADY_PRESENTED
    global SELECTED_WORD_SENSES
    COMMS_ALREADY_PRESENTED = set()
    SELECTED_WORD_SENSES = []
    clustering_data = cached_read_clustering_data_of_word(tokenizer, data_dir, word)[method][str(n_reps)]
    cluster_idxs = list(range(len(clustering_data)))

    word_sense_selections = []
    for cluster_idx, cluster in zip(cluster_idxs, clustering_data):
        word_sense = f"{word}-{cluster_idx}"
        COMMS_ALREADY_PRESENTED.add(word_sense)
        if st.checkbox(f"{word_sense} - ({', '.join([tokenizer.decode([t]) for t, count in cluster[:5]])})"):
            word_sense_selections.append(word_sense)

    SELECTED_WORD_SENSES += word_sense_selections

    present_word_sense_selections(tokenizer, word_sense_selections, data_dir, method, n_reps)

    if len(SELECTED_WORD_SENSES) > 0:
        st.markdown(f"[Search for selected senses](http://{EXTERNAL_IP}:8501/?app=senseie&senses={','.join(SELECTED_WORD_SENSES)})")

def present_word_sense_selections(tokenizer, word_sense_selections, data_dir, method, n_reps):
    for selection in word_sense_selections:
        curr_word, curr_word_sense = selection.split('-')
        st.beta_expander(f"Because you chose {curr_word}-{curr_word_sense}")
        checkbox_close_communities(tokenizer, data_dir, curr_word, curr_word_sense, method, n_reps)

def top_n_reps(tokenizer, data_dir, word, cluster_idx, method, n_reps, max_reps_to_show=5):
    clustering_data = cached_read_clustering_data_of_word(tokenizer, data_dir, word)[method][str(n_reps)]
    return [word] + [tokenizer.decode([t]) for t, count in clustering_data[int(cluster_idx)][:max_reps_to_show-1]]

def checkbox_close_communities(tokenizer, data_dir, word, cluster_idx, method, n_reps):
    global COMMS_ALREADY_PRESENTED
    global SELECTED_WORD_SENSES
    close_communities = read_close_communities(data_dir, word, cluster_idx, method, n_reps)
    word_sense_selections = []
    for word_sense, score in close_communities:
        if float(score) > 0.1 and word_sense not in COMMS_ALREADY_PRESENTED:
            col1, col2 = st.beta_columns((1, 20))
            COMMS_ALREADY_PRESENTED.add(word_sense)
            curr_word, curr_word_sense = word_sense.split('-')
            col1.markdown(f"[link](http://{EXTERNAL_IP}:8501/?word={curr_word}#{curr_word_sense})")
            if col2.checkbox(f"{word_sense} ({score}) - ({', '.join(top_n_reps(tokenizer, data_dir, *word_sense.split('-'), method, n_reps))})"):
                word_sense_selections.append(word_sense)
    SELECTED_WORD_SENSES += word_sense_selections
    present_word_sense_selections(tokenizer, word_sense_selections, data_dir, method, n_reps)

def wsi_at_scale(tokenizer, data_dir, word, method, n_reps, cluster_reps_to_present, special_tokens, num_sent_to_print):
    clustering_data = cached_read_clustering_data_of_word(tokenizer, data_dir, word)[method][str(n_reps)]
    cluster_idxs = list(range(len(clustering_data)))

    for cluster_idx, cluster in zip(cluster_idxs, clustering_data):
        st.markdown(f"<h3 id='{cluster_idx}'>{word} sense #{cluster_idx}</h3>", unsafe_allow_html=True)
        present_histogram(cluster, cluster_reps_to_present, tokenizer)
        if st.checkbox('Show Examples', key=f'Cluster {cluster_idx}'):
            present_example_sents(data_dir, tokenize(tokenizer, word), method, n_reps, cluster_idx, special_tokens, tokenizer, num_sent_to_print)
        if st.checkbox(f'Show Close Communities', key=f'{cluster_idx}'):
            show_close_communities(data_dir, word, cluster_idx, method, n_reps)

def show_close_communities(data_dir, word, cluster_idx, method, n_reps):
    close_communities = read_close_communities(data_dir, word, cluster_idx, method, n_reps)
    for word_sense, score in close_communities:
        if float(score) > 0.1:
            curr_word, curr_word_sense = word_sense.split('-')
            st.markdown(f"[{word_sense}](http://{EXTERNAL_IP}:8501/?word={curr_word}#{curr_word_sense})")

def present_example_sents(data_dir, token, method, n_reps, cluster_idx, special_tokens, tokenizer, num_sent_to_print):
    sent_in_cluster_dir = os.path.join(data_dir, SENTS_BY_CLUSTER)
    cluster_sents_file = os.path.join(sent_in_cluster_dir, f"{token}-{method}-{n_reps}.{cluster_idx}")
    with open(cluster_sents_file) as f:
        sents_presented = 0
        for row in f:
            filename, positions = row.split('\t')
            positions = [int(p.split(',')[0]) for p in positions.rstrip().split()]
            tokens = np.load(npy_file_path(f"{data_dir}/..", filename, 'tokens'), mmap_mode='r')
            lengths = np.load(npy_file_path(f"{data_dir}/..", filename, 'lengths'), mmap_mode='r')
            paragraph_and_positions = list(find_paragraph_and_positions(positions, tokens, lengths))
            for paragraph, token_pos_in_paragraph, _ in paragraph_and_positions:
                for local_pos in token_pos_in_paragraph:
                    show_sent(paragraph, local_pos, special_tokens, tokenizer)
                    sents_presented += 1
                    if sents_presented >= num_sent_to_print:
                        return

def show_sent(paragraph, local_pos, special_tokens, tokenizer):
    single_sent, word_pos = RepInstances.find_single_sent_around_token(paragraph, local_pos, special_tokens)
    before , word, after = single_sent[:word_pos], single_sent[word_pos], single_sent[word_pos+1:]
    before = before[before!=special_tokens.CLS]
    before = before[before!=special_tokens.SEP]
    after = after[after!=special_tokens.CLS]
    after = after[after!=special_tokens.SEP]
    before_text = tokenizer.decode(before)
    word_text = tokenizer.decode([word])
    after_text = tokenizer.decode(after)
    annotated_text(before_text,
        (word_text.strip(), '', "#8ef"),
        after_text,
        height=100
        )

def present_histogram(cluster, cluster_reps_to_present, tokenizer):
    keys, values = zip(*cluster[:cluster_reps_to_present])
    keys = [tokenizer.decode([k]) for k in keys]
    source = pd.DataFrame({'words': keys, 'occurrences': values})
    chart = alt.Chart(source).mark_bar().encode(
        x='occurrences:Q',
        y=alt.Y('words:N', sort='-x')
    ).configure_axis(
        labelFontSize=13,
    )
    chart.configure_axis(labelFontSize=0)
    st.altair_chart(chart, use_container_width=True)

def find_paragraph_and_positions(token_positions, tokens, lengths):
    token_positions = np.array(token_positions)
    length_sum = 0
    for length in lengths:
        token_pos = token_positions[np.where(np.logical_and(token_positions >= length_sum, token_positions < length_sum + length))[0]]
        if len(token_pos) > 0:
            yield tokens[length_sum:length_sum + length], token_pos-length_sum, token_pos
        length_sum += length

def dataset_configs(dataset):
    if dataset == 'Wikipedia':
        model_hf_path = 'bert-large-cased-whole-word-masking'
        data_dir = '/mnt/disks/mnt2/datasets/processed_for_WSI/wiki/bert/v1'
        example_word = 'bass'
        special_tokens = SpecialTokens(model_hf_path)
    else:
        raise NotImplementedError
    return model_hf_path, data_dir, example_word, special_tokens

def method_format_func(method_str):
    if method_str == 'Agglomerative Clustering':
        return 'agglomerative_clustering'
    elif method_str == 'Community Detection':
        return 'community_detection'
    else:
        return method_str

def app_str_format_func(app_str):
    if app_str == 'WSI at Scale':
        return 'home'
    elif app_str == 'List Expansion':
        return 'lstexp'
    elif app_str == 'IE by Sense':
        return 'senseie'
    elif app_str == 'Infer Senses by List':
        return 'seninfer'
    else:
        return app_str

def app_index(app_str):
    if app_str in ['home', None]:
        return 0
    if app_str == 'lstexp':
        return 1
    if app_str == 'senseie':
        return 2
    if app_str == 'seninfer':
        return 3

# Deprecated
# from utils.utils import timeit
# @timeit
# def find_top_matches(words, word_clusters, cluster_reps_to_use):
#     flat_word_clusters_dict, flat_word_clusters_names, sense_counts = flatten_word_clusters(word_clusters, cluster_reps_to_use)

#     sense_positions = create_sense_positions(words, sense_counts)

#     n_closest_comms_to_save = 3
#     @lru_cache(maxsize=None)
#     def cached_jaccard_scores(i, j):
#         return jaccard_score_between_elements(flat_word_clusters_dict[i], flat_word_clusters_dict[j])

#     jaccard_sims_heap = []
#     for sense_indices in product(*sense_positions.values()):
#         aggregated_jaccard_score = 0
#         for i, j in combinations(sense_indices, 2):
#             score = cached_jaccard_scores(i, j)
#             # aggregated_jaccard_score += score
#             if score == 0:
#                 aggregated_jaccard_score += -10
#             else:
#                 aggregated_jaccard_score += np.log(score)
#         if len(jaccard_sims_heap) < n_closest_comms_to_save:
#             heapq.heappush(jaccard_sims_heap, (aggregated_jaccard_score, sense_indices))
#         else:
#             heapq.heappushpop(jaccard_sims_heap, (aggregated_jaccard_score, sense_indices))

#     jaccard_sims_heap = [heapq.heappop(jaccard_sims_heap) for _ in range(len(jaccard_sims_heap))]
#     jaccard_sims_heap.reverse()

#     return flat_word_clusters_names, jaccard_sims_heap

# Deprecated
# from utils.utils import timeit
# @timeit
# def print_top_matches(tokenizer, data_dir, method, n_reps, cluster_reps_to_use, jaccard_sims_heap, flat_word_clusters_names):
#     for jaccard_score, prod in jaccard_sims_heap:
#         words_counter = Counter()
#         out_str = []
#         for i in prod:
#             word_sense = flat_word_clusters_names[i]
#             curr_out = f"*{word_sense}* "
#             top_reps = top_n_reps(tokenizer, data_dir, *word_sense.split('-'), method, n_reps, cluster_reps_to_use)
#             words_counter.update(top_reps)

#             curr_out += f"({', '.join(top_reps)})"
#             curr_out += '\n'
#             out_str.append(curr_out)
#             group_name = words_counter.most_common(1)[0][0]
#         out_str = [f'**Possible Name <span style="color:red">{group_name.title()}</span> ({str(round(jaccard_score, 3))}):**\n'] + out_str
#         st.write("\n".join(out_str), unsafe_allow_html=True)

if __name__ == "__main__":
    main()