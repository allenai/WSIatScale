# pylint: disable=no-member
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
from WSIatScale.create_inverted_index import index
from streamlit_utils.utils import StreamlitTqdm
import altair as alt

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def cached_read_files(token, replacements_dir, inverted_index, sample_n_files, seed):
    return read_files(token, replacements_dir, inverted_index, sample_n_files, seed, bar=StreamlitTqdm)

@st.cache(hash_funcs={tokenizers.Tokenizer: id}, suppress_st_warning=True, allow_output_mutation=True)
def cached_tokenizer(model_hg_path):
    tokenizer = AutoTokenizer.from_pretrained(model_hg_path, use_fast=True)
    return tokenizer

@st.cache(hash_funcs={RepsToInstances: id}, suppress_st_warning=True, allow_output_mutation=True)
def cached_jaccard_distances(reps_to_instances):
    sorted_reps_to_instances = [{'reps': k, 'examples': v} for k, v in sorted(reps_to_instances.data.items(), key=lambda kv: len(kv[1]), reverse=True)]
    jaccard_distances = Jaccard().pairwise_distance([x['reps'] for x in sorted_reps_to_instances])
    return jaccard_distances, sorted_reps_to_instances

def main():
    st.title('WSI at Scale')

    dataset = st.sidebar.selectbox('Dataset', ('Wikipedia', 'CORD-19'))
    args = prepare_arguments()
    if dataset == 'Wikipedia':
        args.model_hg_path = 'roberta-large'
        args.replacements_dir = '/mnt/disks/mnt1/datasets/processed_for_WSI/wiki/all/replacements'
        args.inverted_index = '/mnt/disks/mnt1/datasets/processed_for_WSI/wiki/all/inverted_index'
    else:
        args.model_hg_path = 'allenai/scibert_scivocab_uncased'
        args.replacements_dir = '/mnt/disks/mnt1/datasets/processed_for_WSI/CORD-19/replacements/done'
        args.inverted_index = '/mnt/disks/mnt1/datasets/processed_for_WSI/CORD-19/inverted_index'

    tokenizer = cached_tokenizer(args.model_hg_path)

    word = st.text_input('Word to disambiguate: (Split multiple words by `;` no space)', '')
    n_reps = st.slider(f"Number of replacements (Taken from {args.model_hg_path}'s masked LM)", 1, 100, 5)
    args.n_reps = n_reps

    sample_n_files = st.number_input("Number of files", min_value=1, max_value=10000, value=1000)
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
            # This should be in analyze.
            # similar_words = [w.lower().lstrip(), f" {w.lower().lstrip()}", w.lower().title().lstrip(), f" {w.title().lstrip()}"]
            # similar_tokens = []
            # for w in similar_words:
            #     t = tokenizer.encode(w, add_special_tokens=False)
            #     if len(t) == 1:
            #         similar_tokens.append(t[0])
            curr_word_reps_to_instances = read_files_from_cache(args, tokenizer, dataset, token, n_reps, sample_n_files)
            if reps_to_instances is None:
                reps_to_instances = curr_word_reps_to_instances
            else:
                reps_to_instances.merge(curr_word_reps_to_instances)

            # reps_to_instances.data = {tuple(x for x in k if x not in similar_tokens): v for k, v in reps_to_instances.data.items()}

    if reps_to_instances and st.checkbox('cluster'):
        display_clustering(args, tokenizer, reps_to_instances)

    if reps_to_instances and st.checkbox('apriori'):
        display_apriori(tokenizer, reps_to_instances)

def read_files_from_cache(args, tokenizer, dataset, token, n_reps, sample_n_files):
    try:
        all_reps_to_instances, msg = cached_read_files(token, args.replacements_dir, args.inverted_index, sample_n_files, args.seed)
        reps_to_instances = all_reps_to_instances.populate_specific_size(n_reps)
        st.write(msg)
        return reps_to_instances
    except ValueError as e:
        st.write(e)
        # if st.button('Index this word'):
        #     st.write('This might take a few minutes.')
        #     index(tokenizer, args.word, args.replacements_dir, args.inverted_index, dataset, bar=StreamlitTqdm)

def display_clustering(args, tokenizer, reps_to_instances):
    cluster_alg = st.selectbox(
        'Clustering Algorithm',
        ('KMedoids', 'agglomerative_clustering', 'dbscan', 'kmeans'),
        index=0)

    if cluster_alg == 'kmeans' or cluster_alg == 'KMedoids':
        n_clusters = st.slider('Num of Clusters', 1, 20, 2)
        args.n_clusters = n_clusters

    if cluster_alg == 'agglomerative_clustering':
        n_clusters = st.slider('Num of Clusters', 0, 20, 0)
        if n_clusters != 0:
            args.n_clusters = n_clusters
        distance_threshold = st.slider('Distance Threshold: The linkage distance threshold above which, clusters will not be merged.', 0.0, 1.00, 0.5)
        if distance_threshold != 0:
            distance_threshold += 0.0000001
            args.distance_threshold = distance_threshold
        affinity = st.selectbox('Affinity', ("euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"), index=5)
        args.affinity = affinity
        linkage = st.selectbox('Linkage', ("ward", "complete", "average", "single"), index=1)
        args.linkage = linkage

    if cluster_alg == 'dbscan':
        eps = st.slider('eps: max distance between two samples for one to be considered as in the neighborhood of the other', 0.0, 1.0, 0.5)
        args.eps = eps
        min_samples = st.slider('min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.', 0, 100, 5)
        args.min_samples = min_samples

    args.show_top_n_words_per_cluster = st.slider('Number of words to show per cluster', 1, 100, 10)
    n_sents_to_print = st.number_input('Exemplary Sentences to Present', value=1, min_value=0)
    args.n_sents_to_print = n_sents_to_print

    clustering_load_state = st.text('Clustering...')
    jaccard_distances, sorted_reps_to_instances = cached_jaccard_distances(reps_to_instances)

    model = ClusterFactory.make(cluster_alg, args)
    clusters = model.fit_predict(jaccard_distances)

    clustered_reps = model.reps_to_their_clusters(clusters, sorted_reps_to_instances)

    representative_sents = model.representative_sents(clusters, sorted_reps_to_instances, jaccard_distances, n_sents_to_print)
    st.header(f"Found {len(set(clusters))} clusters.")
    for words_in_cluster, sents, msg in model.group_for_display(args, tokenizer, clustered_reps, representative_sents):
        st.subheader(msg['header'])
        st.write(msg['found'])
        if words_in_cluster:
            keys, values = zip(*words_in_cluster)
            source = pd.DataFrame({'words': keys, 'occurrences': values})

            chart = alt.Chart(source).mark_bar().encode(
                x='occurrences:Q',
                y=alt.Y('words:N', sort='-x')
            ).configure_axis(
                labelFontSize=13,
            )
            chart.configure_axis(labelFontSize=0)
            st.altair_chart(chart, use_container_width=True)
            if n_sents_to_print > 0:
                st.write('**Exemplary Sentences**')
                for sent in sents:
                    st.write(f"* {tokenizer.decode(sent).lstrip()}")

    clustering_load_state.text('')

def display_apriori(tokenizer, reps_to_instances):
    min_support = st.slider('Min Support: Frequency of which the items in the rule appear together in the data set.', 0.0, 1.0, 0.5)
    itemsets = run_apriori(reps_to_instances, min_support)
    st.subheader('Apriori Itemsets')
    if max(itemsets) != 1:
        itemset_sizes = st.slider("itemsets sizes", 1, max(itemsets), (1, max(itemsets)))
    else:
        itemset_sizes = (1, 1)

    for size in range(itemset_sizes[1], itemset_sizes[0]-1, -1):
        itemset_of_size = itemsets[size]
        sorted_itemset_of_size = {k: v.itemset_count for k, v in sorted(itemset_of_size.items(), key=lambda item: item[1].itemset_count, reverse=True)}
        for i, (item, frequency) in enumerate(sorted_itemset_of_size.items()):
            st.write(f"{tokenizer.decode(list(item))}: {frequency}")
            if i > 9:
                st.write(f"**{len(itemsets[size])-9} additional itemsets are available of size {size}**")
                break

if __name__ == "__main__":
    main()
