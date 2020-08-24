# pylint: disable=no-member
from typing import Dict
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer
import tokenizers

from WSIatScale.analyze import (prepare_arguments,
                                read_files,
                                ClusterFactory,
                                Jaccard,
                                tokenize,
                                run_apriori,
                                RepsToInstances)
from WSIatScale.create_inverted_index import index_single_word
from streamlit_utils.utils import StreamlitTqdm
import altair as alt

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def cached_read_files(token, replacements_dir, inverted_index, sample_n_files):
    return read_files(token, replacements_dir, inverted_index, sample_n_files, bar=StreamlitTqdm)

@st.cache(hash_funcs={tokenizers.Tokenizer: id}, suppress_st_warning=True, allow_output_mutation=True)
def cached_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', use_fast=True)
    return tokenizer

@st.cache(hash_funcs={RepsToInstances: id}, suppress_st_warning=True, allow_output_mutation=True)
def cached_jaccard(reps_to_instances):
    sorted_reps_to_instances = [{'reps': k, 'examples': v} for k, v in sorted(reps_to_instances.data.items(), key=lambda kv: len(kv[1]), reverse=True)]
    jaccard_matrix = Jaccard().pairwise_distance([x['reps'] for x in sorted_reps_to_instances])
    return jaccard_matrix, sorted_reps_to_instances

def main():
    st.title('WSI at Scale - CORD19')
    args = prepare_arguments()

    tokenizer = cached_tokenizer()

    word = st.text_input('Word to disambiguate: (Split multiple words by `;` no space)', '')
    n_reps = st.slider("Number of replacements (Taken from SciBERT's masked LM)", 1, 100, 5)
    args.n_reps = n_reps

    sample_n_files = st.slider("Number of files", 1, 10000, 1000)
    args.sample_n_files = sample_n_files
    if word == '': return

    reps_to_instances = None
    for w in word.split(';'):
        token = None
        args.word = w

        try:
            token = tokenize(tokenizer, w)
        except ValueError as e:
            st.write('Word given is more than a single wordpiece. Please choose a different word.')

        if token:
            curr_word_reps_to_instances = read_files_from_cache(args, tokenizer, token, n_reps, sample_n_files)
            if reps_to_instances is None:
                reps_to_instances = curr_word_reps_to_instances
            else:
                reps_to_instances.merge(curr_word_reps_to_instances)

    if reps_to_instances and st.checkbox('cluster'):
        display_clustering(args, tokenizer, reps_to_instances)

    if reps_to_instances and st.checkbox('apriori'):
        display_apriori(tokenizer, reps_to_instances)

def read_files_from_cache(args, tokenizer, token, n_reps, sample_n_files):
    try:
        all_reps_to_instances, msg = cached_read_files(token, args.replacements_dir, args.inverted_index, sample_n_files)
        reps_to_instances = all_reps_to_instances.populate_specific_size(n_reps)
        st.write(msg)
        return reps_to_instances
    except ValueError as e:
        st.write('token is not in the inverted index.')
        if st.button('Index this word'):
            st.write('This might take a few minutes.')
            msg = index_single_word(tokenizer, args.word, args.replacements_dir, args.inverted_index, bar=StreamlitTqdm)
            st.write(msg)

def display_clustering(args, tokenizer, reps_to_instances):
    cluster_alg = option = st.selectbox(
        'Clustering Algorithm',
        ('kmeans', 'agglomerative_clustering', 'dbscan'))

    if option == 'kmeans':
        n_clusters = st.slider('Num of Clusters', 1, 20, 2)
        args.n_clusters = n_clusters

    if option == 'agglomerative_clustering':
        n_clusters = st.slider('Num of Clusters', 0, 20, 0)
        if n_clusters != 0:
            args.n_clusters = n_clusters
        distance_threshold = st.slider('Distance Threshold', 0.0, 1.0, 0.5)
        args.distance_threshold = distance_threshold
        affinity = st.selectbox('Affinity', ("euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"), index=5)
        args.affinity = affinity
        linkage = st.selectbox('Linkage', ("ward", "complete", "average", "single"), index=1)
        args.linkage = linkage

    if option == 'dbscan':
        eps = st.slider('eps: max distance between two samples for one to be considered as in the neighborhood of the other', 0.0, 1.0, 0.5)
        args.eps = eps
        min_samples = st.slider('min_samples: ', 0, 100, 5)
        args.min_samples = min_samples

    args.show_top_n_words_per_cluster = st.slider('Number of words to show per cluster', 1, 100, 10)
    n_sents_to_print = st.number_input('Exemplary Sentences to Present', value=1, min_value=0)
    args.n_sents_to_print = n_sents_to_print

    clustering_load_state = st.text('Clustering...')
    jaccard_matrix, sorted_reps_to_instances = cached_jaccard(reps_to_instances)

    clustering = ClusterFactory.make(cluster_alg, args)
    clusters = clustering.fit_predict(jaccard_matrix)

    clustered_reps = clustering.reps_to_their_clusters(clusters, sorted_reps_to_instances)

    representative_sents = clustering.representative_sents(clusters, sorted_reps_to_instances, jaccard_matrix, args.n_sents_to_print)
    for i, (words_in_cluster, sents, msg) in enumerate(clustering.group_for_display(args, tokenizer, clustered_reps, representative_sents)):
        st.subheader(msg['header'])
        st.write(msg['found'])
        if words_in_cluster is not None:
            keys, values = zip(*words_in_cluster)
            source = pd.DataFrame({'words': keys, 'occurrences': values})

            chart = alt.Chart(source).mark_bar().encode(
                x='occurrences:Q',
                y=alt.Y('words:N', sort='-x')
            )
            st.altair_chart(chart, use_container_width=True)
            if n_sents_to_print > 0:
                st.write('**Exemplary Sentences**')
                for sent in sents:
                    st.write(f"*{tokenizer.decode(sent)}*")

    clustering_load_state.text('')

def display_apriori(tokenizer, reps_to_instances):
    min_support = st.slider('Min Support: Frequency of which the items in the rule appear together in the data set.', 0.0, 1.0, 0.5)
    itemsets, _ = run_apriori(reps_to_instances, min_support)
    st.subheader('Apriori Itemsets')
    if max(itemsets) != 1:
        itemset_sizes = st.slider("itemsets sizes", 1, max(itemsets), (1, max(itemsets)))
    else:
        itemset_sizes = (1, 1)
    for size in range(itemset_sizes[1], itemset_sizes[0]-1, -1):
        for i, (item, frequency) in enumerate(itemsets[size].items()):
            st.write(f"{tokenizer.decode(list(item))}: {frequency}")
            if i > 9:
                st.write(f"**{len(itemsets[size])-9} additional itemsets are available of size {size}**")
                break

if __name__ == "__main__":
    main()