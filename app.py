# pylint: disable=no-member
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer
import tokenizers

from WSIatScale.analyze import (read_files,
                                Jaccard,
                                RepsToInstances,
                                CommunityFinder,
                                prepare_arguments,
                                tokenize,
                                ClusterFactory,
                                run_apriori)

# from WSIatScale.create_inverted_index import index
from streamlit_utils.utils import StreamlitTqdm
import altair as alt

SEED = 111

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def cached_read_files(token, replacements_dir, inverted_index, sample_n_files):
    return read_files(token, replacements_dir, inverted_index, sample_n_files, bar=StreamlitTqdm)

@st.cache(hash_funcs={RepsToInstances: id}, suppress_st_warning=True, allow_output_mutation=True)
def cached_populate_specific_size(all_reps_to_instances, n_reps):
    return all_reps_to_instances.populate_specific_size(n_reps)

@st.cache(hash_funcs={tokenizers.Tokenizer: id}, suppress_st_warning=True, allow_output_mutation=True)
def cached_tokenizer(model_hg_path):
    tokenizer = AutoTokenizer.from_pretrained(model_hg_path, use_fast=True)
    return tokenizer

@st.cache(hash_funcs={RepsToInstances: id}, suppress_st_warning=True, allow_output_mutation=True)
def cached_jaccard_distances(reps_to_instances):
    sorted_reps_to_instances = [{'reps': k, 'examples': v} for k, v in sorted(reps_to_instances.data.items(), key=lambda kv: len(kv[1]), reverse=True)]
    jaccard_distances = Jaccard().pairwise_distance([x['reps'] for x in sorted_reps_to_instances])
    return jaccard_distances, sorted_reps_to_instances

@st.cache(hash_funcs={RepsToInstances: id}, suppress_st_warning=True, allow_output_mutation=True)
def cached_CommunityFinder(reps_to_instances):
    return CommunityFinder(reps_to_instances)

@st.cache(hash_funcs={CommunityFinder: id}, suppress_st_warning=True, allow_output_mutation=True)
def cached_find_communities(community_finder, method, top_n_nodes_to_keep=None):
    return community_finder.find(method, top_n_nodes_to_keep)

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

    word = st.sidebar.text_input('Word to disambiguate: (Split multiple words by `;` no space)', ' bass')
    n_reps = st.sidebar.slider(f"Number of replacements (Taken from {args.model_hg_path}'s masked LM)", 1, 100, 5)
    args.n_reps = n_reps

    sample_n_files = st.sidebar.number_input("Number of files", min_value=1, max_value=10000, value=1000)
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
            curr_word_reps_to_instances = read_files_from_cache(args, tokenizer, dataset, token, n_reps, sample_n_files)
            curr_word_reps_to_instances.remove_query_word(tokenizer, w, merge_same_keys=True)

            if reps_to_instances is None:
                reps_to_instances = curr_word_reps_to_instances
            else:
                reps_to_instances.merge(curr_word_reps_to_instances)


    action = st.sidebar.selectbox(
        'How to Group Instances',
        ('Select Action', 'Group by Communities', 'Cluster', 'Apriori'),
        index=1)

    if reps_to_instances and action == 'Cluster':
        cluster_alg, n_sents_to_print = prepare_choices(args)
        display_clusters(args, tokenizer, reps_to_instances, cluster_alg, n_sents_to_print)

    if reps_to_instances and action == 'Group by Communities':
        display_communities(args, tokenizer, reps_to_instances)

    if reps_to_instances and action == 'Apriori':
        display_apriori(tokenizer, reps_to_instances)

def read_files_from_cache(args, tokenizer, dataset, token, n_reps, sample_n_files):
    try:
        all_reps_to_instances, msg = cached_read_files(token, args.replacements_dir, args.inverted_index, sample_n_files)
        reps_to_instances = cached_populate_specific_size(all_reps_to_instances, n_reps)
        st.write(msg)
        return reps_to_instances
    except ValueError as e:
        st.write(e)
        # if st.button('Index this word'):
        #     st.write('This might take a few minutes.')
        #     index(tokenizer, args.word, args.replacements_dir, args.inverted_index, dataset, bar=StreamlitTqdm)

def display_clusters(args, tokenizer, reps_to_instances, cluster_alg, n_sents_to_print):
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

def display_communities(args, tokenizer, reps_to_instances):
    top_n_nodes_to_keep = None

    community_alg = st.selectbox(
        'Community Algorithm',
        ('Louvain', 'Weighted Girvan-Newman', 'Girvan-Newman', 'async LPA', 'Weighted async LPA', 'Clauset-Newman-Moore (no weights)'),
        index=0)
    n_sents_to_print = st.number_input('Exemplary Sentences to Present', value=3, min_value=0)
    at_least_n_matches = st.number_input('Number of Minimum Sentence Matches', value=5, min_value=1, max_value=100)
    
    community_finder = cached_CommunityFinder(reps_to_instances)
    communities = cached_find_communities(community_finder, community_alg, top_n_nodes_to_keep)
    communities_tokens, communities_sents = community_finder.community_tokens_with_sents(communities, reps_to_instances)

    print_communities_graph(tokenizer,
                            community_finder,
                            communities,
                            communities_tokens,
                            communities_sents,
                            n_sents_to_print,
                            at_least_n_matches)

def print_communities_graph(tokenizer,
                            community_finder,
                            communities,
                            communities_tokens,
                            communities_sents,
                            n_sents_to_print,
                            at_least_n_matches):
    num_skipped = 0
    for comm, sents in zip(communities_tokens, communities_sents):
        if len(sents) < at_least_n_matches:
            num_skipped += 1
            continue
        checkbox_text = tokenizer.decode(comm)
        if len(checkbox_text) > 500:
            checkbox_text = checkbox_text[:487]+" ..."
        display_sents = st.checkbox(checkbox_text + f" - ({len(sents)} sents)")
        if display_sents:
            for sent in sents[:n_sents_to_print]:
                st.write(f"* {tokenizer.decode(sent).lstrip()}")
    if num_skipped > 0:
        st.write(f"Skipped {num_skipped} communities with less than {at_least_n_matches} sentences.")

    import networkx as nx
    import matplotlib.pyplot as plt
    from PIL import Image

    G = nx.from_numpy_matrix(community_finder.cooccurrence_matrix)

    node2word = {n: tokenizer.decode([community_finder.node2token[n]]) for n in G.nodes()}
    words_i_care_about = ['guitar', 'violin', 'ass', 'fish', 'cod', 'shark', 'trout', 'fishing', 'tuna', 'shrimp', 'carp', 'salmon']
    extended_words_i_care_about = []
    for w in words_i_care_about:
        extended_words_i_care_about.extend([w, f" {w}", w.title(), f" {w.title()}"])
    node2word = {k: v for k, v in node2word.items() if v in extended_words_i_care_about}

    pos = nx.spring_layout(G, seed=SEED)  # compute graph layout
    plt.axis('off')
    partition = {n:c for c in range(len(communities)) for n in communities[c]}
    sorted_partition_values = [partition[n] for n in range(max(partition) + 1)]
    nx.draw_networkx_nodes(G, pos, node_size=100, cmap=plt.cm.RdYlBu, node_color=sorted_partition_values)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, node2word, font_size=8)

    plt.savefig("Graph.png", format="PNG")

    image = Image.open('Graph.png')
    st.image(image, use_column_width=True)

def prepare_choices(args):
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
    return cluster_alg, n_sents_to_print

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
