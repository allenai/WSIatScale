# pylint: disable=no-member
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer
import tokenizers

from WSDatScale.analyze import prepare_arguments, read_files, bag_in_size, ClusterFactory, Jaccard, tokenize
from streamlit_utils.utils import StreamlitTqdm
import altair as alt

@st.cache(hash_funcs={tokenizers.Tokenizer: id}, suppress_st_warning=True, allow_output_mutation=True)
def cached_read_files(*args, **kwargs):
    return read_files(*args, **kwargs)

@st.cache(hash_funcs={tokenizers.Tokenizer: id}, suppress_st_warning=True, allow_output_mutation=True)
def cached_tokenizer():
    tokenizer_load_state = st.text('Importing tokenizer')
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', use_fast=True)
    tokenizer_load_state.text('')
    return tokenizer

def main():
    st.title('WSD at Scale - CORD19')
    args = prepare_arguments()

    tokenizer = cached_tokenizer()

    word = st.text_input('Word to disambiguate:', 'race')
    args.word = word
    n_reps = st.slider('Number of replacements', 1, 100, 5)

    token, bag_of_reps = None, None
    try:
        token = tokenize(tokenizer, word)
    except ValueError as e:
        st.write('Word given is more than a single wordpiece. Please choose a different word.')

    if token:
        bag_of_reps = read_files_from_cache(args, tokenizer, token, n_reps)

    if bag_of_reps:
        cluster(args, tokenizer, bag_of_reps)

def read_files_from_cache(args, tokenizer, token, n_reps):
    try:
        bag_of_all_reps, msg = cached_read_files(args, tokenizer, token, bar=StreamlitTqdm)
        bag_of_reps = bag_in_size(bag_of_all_reps, n_reps)
        st.write(msg)
        return bag_of_reps
    except ValueError as e:
        st.write('token is not in the inverted index. Dynamically indexing will be available soon.')


def cluster(args, tokenizer, bag_of_reps):
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

    if option == 'dbscan':
        eps = st.slider('eps: max distance between two samples for one to be considered as in the neighborhood of the other', 0.0, 1.0, 0.5)
        args.eps = eps
        min_samples = st.slider('min_samples: ', 0, 100, 5)
        args.min_samples = min_samples

    args.show_top_n_words_per_cluster = st.slider('Number of words to show per cluster', 1, 100, 10)
    n_sents_to_print = st.number_input('Exemplary Sentences to Present', value=1, min_value=0)
    args.n_sents_to_print = n_sents_to_print

    sorted_bag_of_reps = [{'reps': k, 'examples': v} for k, v in sorted(bag_of_reps.items(), key=lambda kv: len(kv[1]), reverse=True)]
    jaccard_matrix = Jaccard().pairwise_distance([x['reps'] for x in sorted_bag_of_reps])

    clustering = ClusterFactory.make(cluster_alg, args)
    clusters = clustering.fit_predict(jaccard_matrix)

    clustered_reps = clustering.reps_to_their_clusters(clusters, sorted_bag_of_reps)

    representative_sents = clustering.representative_sents(clusters, sorted_bag_of_reps, jaccard_matrix, args.n_sents_to_print)
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

if __name__ == "__main__":
    main()