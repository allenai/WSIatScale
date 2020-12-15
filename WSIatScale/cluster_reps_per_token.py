# pylint: disable=no-name-in-module
# pylint: disable=import-error
import argparse
from collections import Counter
from copy import deepcopy
from functools import partial
import json
from multiprocessing import Pool, cpu_count
import os
import numpy as np
from tqdm import tqdm

from WSIatScale.analyze import read_files
from utils.utils import tokenizer_params

from WSIatScale.clustering import MyBOWHierarchicalLinkage
from WSIatScale.community_detection import CommunityFinder
from utils.special_tokens import SpecialTokens

from transformers import AutoTokenizer

SAMPLE_N_INSTANCES = 1000
MOST_COMMON_CLUSTER_REPS = 100
# WORD_CLUSTERS_DIR = 'word_clusters_resolution1.2' #TODO pick
WORD_CLUSTERS_DIR = 'word_clusters'

def main(args):
    model_hf_path = tokenizer_params[args.dataset]
    tokenizer = AutoTokenizer.from_pretrained(model_hf_path, use_fast=True)
    special_tokens = SpecialTokens(model_hf_path)
    tokens_to_index = special_tokens.tokens_to_annotate()
    already_done = set([int(f.split('_')[0]) for f in os.listdir(out_dir)])
    tokens_to_index -= already_done
    print(f"{len(tokens_to_index)} tokens to index")

    partial_write_communities_to_disk = partial(write_communities_to_disk, tokenizer=tokenizer, model_hf_path=model_hf_path)
    with Pool(cpu_count()) as p:
        list(tqdm(p.imap(partial_write_communities_to_disk, tokens_to_index), total=len(tokens_to_index)))

def write_communities_to_disk(token, tokenizer, model_hf_path):
    rep_instances, _ = read_files(token, args.data_dir,
        SAMPLE_N_INSTANCES,
        SpecialTokens(model_hf_path),
        should_lemmatize=True,
        instance_attributes=['doc_id', 'reps'],
        bar=lambda x: x,
    )

    clustering_data_by_n_reps = {'agglomerative_clustering': {}, 'community_detection': {}}
    for n_reps in [5, 20, 50]:
        curr_rep_instances = deepcopy(rep_instances)
        curr_rep_instances.remove_query_word(tokenizer, tokenizer.decode([token]))
        curr_rep_instances.populate_specific_size(n_reps)
        curr_rep_instances.remove_empty_replacements()

        # clustering_data_by_n_reps['agglomerative_clustering'][n_reps] = agglomerative_clustering(curr_rep_instances)
        clustering_data_by_n_reps['community_detection'][n_reps] = community_detection_clustering(curr_rep_instances)
    json.dump(clustering_data_by_n_reps, open(os.path.join(args.out_dir, f"{token}_clustering.json"), 'w'))

def agglomerative_clustering(rep_instances):
    model = MyBOWHierarchicalLinkage()
    clusters = model.fit_predict(rep_instances)
    clustered_reps = model.reps_to_their_clusters(clusters, rep_instances)
    clustering_data = []
    for _, intances in clustered_reps.items():
        cluster_reps = [inst.reps for inst in intances]
        reps_counter = Counter(int(r) for reps in cluster_reps for r in reps)
        most_common_tokens = reps_counter.most_common(MOST_COMMON_CLUSTER_REPS)
        if community_big_enough_heuristics(most_common_tokens):
            clustering_data.append(most_common_tokens)

    return clustering_data

def community_detection_clustering(rep_instances, query_n_reps=10):
    community_finder = CommunityFinder(rep_instances, query_n_reps)
    communities = community_finder.find(resolution=1.)

    community_tokens, communities_sents_data, _ = community_finder.argmax_voting(communities, rep_instances)

    community_tokens = sort_community_tokens_by_popularity(rep_instances, community_tokens)
    clustering_data = []
    for com_tokens, _ in zip(community_tokens, communities_sents_data):
        most_common_tokens = [(int(t), v) for t, v in com_tokens[:MOST_COMMON_CLUSTER_REPS]]
        if community_big_enough_heuristics(most_common_tokens):
            clustering_data.append(most_common_tokens)
        
    return clustering_data

def community_big_enough_heuristics(most_common_tokens):
    minimum_second_word_instances = 10
    return len(most_common_tokens) > 1 and most_common_tokens[1][1] > minimum_second_word_instances

def sort_community_tokens_by_popularity(rep_instances, community_tokens):
    ret = []
    for comm in community_tokens:
        community_tokens_by_popularity = {t: 0 for t in comm}
        for rep_inst in rep_instances.data:
            for token in comm:
                if token in rep_inst.reps:
                    community_tokens_by_popularity[token] += 1
        community_tokens_by_popularity = [(k, v) for k, v in sorted(community_tokens_by_popularity.items(), key=lambda item: item[1], reverse=True)]
        ret.append(community_tokens_by_popularity)

    return ret

def read_clustering_data(data_dir, token):
    cluster_file = os.path.join(data_dir, WORD_CLUSTERS_DIR, f"{token}_clustering.json")
    return json.load(open(cluster_file, 'r'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="replacements")
    parser.add_argument("--dataset", type=str, choices=['CORD-19', 'Wikipedia-roberta', 'Wikipedia-BERT'])
    args = parser.parse_args()

    out_dir = os.path.join(args.data_dir, WORD_CLUSTERS_DIR)

    args.out_dir = out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # assert len(os.listdir(out_dir)) == 0, f"{out_dir} already exists."

    main(args)