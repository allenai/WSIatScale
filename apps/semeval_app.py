# pylint: disable=no-member
from collections import Counter
import json
import numpy as np
import os
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer
import tokenizers
import random
import sys
sys.path.append('/home/matane/matan/dev/WSIatScale')

from WSIatScale.analyze import (read_files,
                                RepInstances,
                                prepare_arguments,
                                tokenize)
from WSIatScale.clustering import ClusterFactory
from WSIatScale.community_detection import find_communities_and_vote, CommunityFinder, label_by_comms
from WSIatScale.apriori import run_apriori
from SemEval.evaluate import evaluate_labeling_2010

from utils.utils import StreamlitTqdm, SpecialTokens
import altair as alt

import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image

SEED = 111

@st.cache(hash_funcs={tokenizers.Tokenizer: id}, suppress_st_warning=True, allow_output_mutation=True)
def cached_read_files_specific_n_reps(tokenizer, lemma, data_dir, special_tokens, remove_query_word, n_reps):
    rep_instances, msg  = read_files(lemma, data_dir, -1, special_tokens, instance_attributes=['doc_id', 'reps', 'tokens'], bar=StreamlitTqdm)
    if remove_query_word:
        rep_instances.remove_query_word(tokenizer, lemma)
    rep_instances.populate_specific_size(n_reps)
    return rep_instances, msg

@st.cache(hash_funcs={tokenizers.Tokenizer: id}, suppress_st_warning=True, allow_output_mutation=True)
def cached_tokenizer(model_hf_path):
    tokenizer = AutoTokenizer.from_pretrained(model_hf_path, use_fast=True)
    return tokenizer

# @st.cache(hash_funcs={RepInstances: id}, suppress_st_warning=True, allow_output_mutation=True)
# def cached_CommunityFinder(rep_instances, n_reps):
#     return CommunityFinder(rep_instances, n_reps)

# @st.cache(hash_funcs={CommunityFinder: id}, suppress_st_warning=True, allow_output_mutation=True)
# def cached_find_communities(community_finder, method):
#     communities = community_finder.find(method)
#     return communities

@st.cache(hash_funcs={RepInstances: id}, suppress_st_warning=True, allow_output_mutation=True)
def cached_find_communities_and_vote(rep_instances, query_n_reps):
    return find_communities_and_vote(rep_instances, query_n_reps)

@st.cache(hash_funcs={CommunityFinder: id, tokenizers.Tokenizer: id}, suppress_st_warning=True, allow_output_mutation=True)
def cached_create_graph(community_finder, communities, tokenizer):
    create_graph(community_finder, communities, tokenizer)

def main():
    st.title('WSI at Scale')

    dataset = st.sidebar.selectbox('Dataset', ('Wikipedia', 'CORD-19', 'SemEval2010', 'SemEval2013'), 2)
    args = prepare_arguments()
    example_word, special_tokens = dataset_configs(dataset, args)

    tokenizer = cached_tokenizer(args.model_hf_path)

    word = st.sidebar.text_input('Word to disambiguate: (Split multiple words by `;` no space)', example_word)
    n_reps = st.sidebar.slider(f"Number of replacements (Taken from {args.model_hf_path}'s masked LM)", 1, 100, 45)
    args.n_reps = n_reps

    if word == '': return

    rep_instances = None

    remove_query_word = st.checkbox('remove_query_word', value=True)

    for w in word.split(';'):
        args.word = w

        curr_word_rep_instances, msg = cached_read_files_specific_n_reps(tokenizer, w, args.data_dir, special_tokens, remove_query_word, n_reps)
        st.write(msg)

        if rep_instances is None:
            rep_instances = curr_word_rep_instances
        else:
            rep_instances.merge(curr_word_rep_instances)


    action = st.sidebar.selectbox(
        'How to Group Instances',
        ('Select Action', 'Group by Communities', 'Cluster', 'Apriori'),
        index=1)

    if rep_instances and action == 'Cluster':
        cluster_alg, n_sents_to_print = prepare_choices(args)
        display_clusters(args, tokenizer, rep_instances, cluster_alg, n_sents_to_print)

    if rep_instances and action == 'Group by Communities':
        display_communities(args, tokenizer, rep_instances)

    if rep_instances and action == 'Apriori':
        display_apriori(tokenizer, rep_instances)

def dataset_configs(dataset, args):
    if dataset == 'SemEval2010':
        args.model_hf_path = 'bert-large-uncased'
        # args.data_dir = '/home/matane/matan/dev/WSIatScale/write_mask_preds/out/SemEval2010/bert-large-uncased'
        # args.data_dir = '/home/matane/matan/dev/WSIatScale/write_mask_preds/out/SemEval2010/bert-large-uncased-no-double-instances'
        args.data_dir = '/mnt/disks/mnt2/datasets/SemEval/SemEval2010/bert-large-uncased-redoing-semeval'
        example_word = 'market'
        special_tokens = SpecialTokens(args.model_hf_path)
    elif dataset == 'SemEval2013':
        args.model_hf_path = 'bert-large-uncased'
        args.data_dir = '/home/matane/matan/dev/WSIatScale/write_mask_preds/out/SemEval2013/bert-large-uncased'
        example_word = 'become'
        special_tokens = SpecialTokens(args.model_hf_path)
    return example_word, special_tokens

def display_clusters(args, tokenizer, rep_instances, cluster_alg, n_sents_to_print):
    clustering_load_state = st.text('Clustering...')

    model = ClusterFactory.make(cluster_alg, args)
    clusters = model.fit_predict(rep_instances)
    clustered_reps = model.reps_to_their_clusters(clusters, rep_instances)
    representative_sents = model.representative_sents(clustered_reps, -1)

    st.header(f"Found {len(set(clusters))} clusters.")
    instance_id_to_doc_id = json.load(open(os.path.join(args.data_dir, 'instance_id_to_doc_id.json'), 'r'))
    doc_id_to_inst_id = {v:k for k,v in instance_id_to_doc_id.items()}
    gold_file = open("/home/matane/matan/dev/SemEval/resources/SemEval-2010/evaluation/unsup_eval/keys/all.key", 'r')
    gold_inst_id_to_sense = {}
    for gold in gold_file:
        if args.word in gold:
            line = gold.rstrip().split(' ')
            gold_inst_id_to_sense[line[1]] = line[2]
    st.write(Counter([cluster for row_id, cluster in gold_inst_id_to_sense.items()]))
    for words_in_cluster, sents_data, msg in model.group_for_display(args, tokenizer, clustered_reps, representative_sents):
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
            clusters_doc_ids = [rep.doc_id for rep in sents_data]
            counter = Counter([cluster for row_id, cluster in gold_inst_id_to_sense.items() if instance_id_to_doc_id[row_id] in clusters_doc_ids])
            from collections import OrderedDict
            st.write(OrderedDict(counter.most_common()))
            for rep_inst in sents_data[:n_sents_to_print]:
                text = f"{tokenizer.decode(rep_inst.sent).lstrip()}"
                reps_text = " ".join([tokenizer.decode([rep]).lstrip() for rep in rep_inst.reps])
                sense = gold_inst_id_to_sense[doc_id_to_inst_id[rep_inst.doc_id]]
                st.write(f"* {sense} - **{reps_text}:** ", text)

    clustering_load_state.text('')

def display_communities(args, tokenizer, rep_instances):
    community_alg = st.selectbox(
        'Community Algorithm',
        ('louvain', 'Weighted Girvan-Newman', 'Girvan-Newman', 'async LPA', 'Weighted async LPA', 'Clauset-Newman-Moore (no weights)'),
        index=0)

    n_sents_to_print = st.number_input('Exemplary Sentences to Present', value=3, min_value=0)
    at_least_n_matches = st.number_input('Number of Minimum Sentence Matches', value=1, min_value=0, max_value=100)
    query_n_reps = st.number_input('query_n_reps', value=15)

    communities_sents_data, (community_finder, communities, communities_tokens) = cached_find_communities_and_vote(rep_instances, query_n_reps)
    instance_id_to_doc_id = json.load(open(os.path.join(args.data_dir, 'instance_id_to_doc_id.json'), 'r'))
    doc_id_to_inst_id = {v:k for k,v in instance_id_to_doc_id.items()}
    lemma_labeling = label_by_comms(communities_sents_data, doc_id_to_inst_id)

    scores = evaluate_labeling_2010("/home/matane/matan/dev/SemEval/resources/SemEval-2010/evaluation/", lemma_labeling)
    for lemma in scores:
        if args.word in lemma:
            fscore = scores[lemma]['FScore']
            vmeasure = scores[lemma]['V-Measure']
            st.write(fscore, vmeasure, round(((fscore*vmeasure)**0.5)*100, 2))

    print_communities(args,
                      tokenizer,
                      community_finder,
                      communities,
                      communities_tokens,
                      communities_sents_data,
                      n_sents_to_print,
                      at_least_n_matches,
                      instance_id_to_doc_id,
                      doc_id_to_inst_id)

def print_communities(args,
                      tokenizer,
                      community_finder,
                      communities,
                      communities_tokens,
                      communities_sents_data,
                      n_sents_to_print,
                      at_least_n_matches,
                      instance_id_to_doc_id,
                      doc_id_to_inst_id):
    num_skipped = 0
    random.seed(SEED)
    gold_file = open("/home/matane/matan/dev/SemEval/resources/SemEval-2010/evaluation/unsup_eval/keys/all.key", 'r')
    gold_inst_id_to_sense = {}
    for gold in gold_file:
        if args.word in gold:
            line = gold.rstrip().split(' ')
            gold_inst_id_to_sense[line[1]] = line[2]
    st.write(Counter([cluster for row_id, cluster in gold_inst_id_to_sense.items()]))
    for i, (comm, rep_instances) in enumerate(zip(communities_tokens, communities_sents_data)):
        if len(rep_instances) < at_least_n_matches:
            num_skipped += 1
            continue
        random.shuffle(rep_instances)
        checkbox_text = get_checkbox_text(comm, rep_instances, tokenizer)
        display_sents = st.checkbox(checkbox_text + f" - ({len(rep_instances)} sents)", value=True)
        if display_sents:
            map_gold_sense_to_sent = {sense: [] for sense in gold_inst_id_to_sense.values()}
            for rep_inst in rep_instances:
                gold_sense = gold_inst_id_to_sense[doc_id_to_inst_id[rep_inst.doc_id]]
                map_gold_sense_to_sent[gold_sense].append(rep_inst)
            for gold_sense, rep_instances in map_gold_sense_to_sent.items():
                if len(rep_instances) > 0 and st.checkbox(f"{gold_sense} ({len(rep_instances)})", key=f"{gold_sense}, {i}, {len(rep_instances)}"):
                    for rep_inst in rep_instances[:n_sents_to_print]:
                        text = f"{tokenizer.decode(rep_inst.sent).lstrip()}"
                        reps_text = " ".join([tokenizer.decode([rep]).lstrip() for rep in rep_inst.reps])
                        st.write(f"* **{reps_text}:** ", text)
    if num_skipped > 0:
        st.write(f"Skipped {num_skipped} communities with less than {at_least_n_matches} sentences.")

    # cached_create_graph(community_finder, communities, tokenizer)
    # print_graph()

def get_checkbox_text(comm, rep_instances, tokenizer):
    max_words_to_display = 10
    community_tokens_counter = {t: 0 for t in comm}
    for rep_inst in rep_instances:
        for token in comm:
            if token in rep_inst.reps:
                community_tokens_counter[token] += 1
    community_tokens_counter = {k: v for k, v in sorted(community_tokens_counter.items(), key=lambda item: item[1], reverse=True)}

    checkbox_text = " ".join([tokenizer.decode([t]) for t in list(community_tokens_counter.keys())[:max_words_to_display]])
    if len(community_tokens_counter) > max_words_to_display:
        checkbox_text = checkbox_text+" ..."
    return checkbox_text

def create_graph(community_finder, communities, tokenizer):

    original_G = nx.from_numpy_matrix(community_finder.cooccurrence_matrix)
    components = list(nx.connected_components(original_G))
    nodes_of_largest_component  = max(components, key = len)
    G = original_G.subgraph(nodes_of_largest_component)
    degrees = dict(G.degree)
    largest_degree = max(degrees.values())

    pos = nx.spring_layout(G, k=0.1, iterations=50, seed=SEED)
    # pos = nx.spring_layout(G, k=0.9, iterations=20, seed=SEED)
    plt.axis('off')
    partition = {n:c for c in range(len(communities)) for n in communities[c]}
    sorted_partition_values = [partition[n] for n in G.nodes()]
    nx.draw_networkx_nodes(G,
                           pos,
                           node_size=[float(v)/largest_degree*500 for v in degrees.values()],
                           cmap=plt.cm.Set3,
                           node_color=sorted_partition_values)
    nx.draw_networkx_edges(G,
                           pos,
                           alpha=0.3)
    if G.number_of_edges() < 50:
        weights = nx.get_edge_attributes(G, 'weight')
        weights = {k: int(v) for k, v in weights.items()}
        nx.draw_networkx_edge_labels(G, pos, font_size=6, edge_labels=weights)

    if len(G.nodes()) <= 50:
        node2word = {n: tokenizer.decode([community_finder.node2token[n]]) for n in G.nodes()}
    else:
        most_common_nodes, _ = zip(*Counter(degrees).most_common(20))
        node2word = {n: tokenizer.decode([community_finder.node2token[n]]) if n in most_common_nodes else '' for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, node2word, font_size=8)

    plt.savefig("/tmp/Graph.png", format="PNG")
    if len(components) > 1:
        st.write(f"Additional {len(components) - 1} components not presented.")

def print_graph():
    image = Image.open('/tmp/Graph.png')
    st.image(image, use_column_width=True)

def prepare_choices(args):
    cluster_alg = st.selectbox(
        'Clustering Algorithm',
        ('KMedoids', 'agglomerative_clustering', 'dbscan', 'kmeans', 'BOW Hierarchical'),
        index=4)

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

def display_apriori(tokenizer, rep_instances):
    min_support = st.slider('Min Support: Frequency of which the items in the rule appear together in the data set.', 0.0, 1.0, 0.5)
    itemsets = run_apriori(rep_instances, min_support)
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