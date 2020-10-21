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
from WSIatScale.create_inverted_index import tokenizer_params, full_words_tokens

from WSIatScale.clustering import MyBOWHierarchicalLinkage
from WSIatScale.community_detection import CommunityFinder

from transformers import AutoTokenizer

FULL_STOP_INDEX = 119
SAMPLE_N_INSTANCES = 1000
MOST_COMMON_CLUSTER_REPS = 100
OUT_FOLDER = 'word_clusters_lemmatized'

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_params[args.dataset], use_fast=True)
    tokens_to_index = full_words_tokens(args.dataset, tokenizer)
    half_words_list = np.load(f"non-full-words/non-full-words-{args.model_hf_path}.npy")
    partial_write_communities_to_disk = partial(write_communities_to_disk, tokenizer=tokenizer, half_words_list=half_words_list)
    with Pool(cpu_count()) as p:
        list(tqdm(p.imap(partial_write_communities_to_disk, tokens_to_index), total=len(tokens_to_index)))

def write_communities_to_disk(token, tokenizer, half_words_list):
    try:
        rep_instances, _ = read_files(token, args.data_dir,
            SAMPLE_N_INSTANCES,
            FULL_STOP_INDEX,
            should_lemmatize=True,
            instance_attributes=['doc_id', 'reps'],
            bar=lambda x: x,
        )
        #TODO lemmatize
        rep_instances.remove_certain_words(tokenizer=tokenizer,
                                            word=tokenizer.decode([token]),
                                            remove_query_word=True,
                                            half_words_list=half_words_list)
        clustering_data_by_n_reps = {'agglomerative_clustering': {}, 'community_detection': {}}
        for n_reps in [5, 20, 50]:
            curr_rep_instances = deepcopy(rep_instances)
            curr_rep_instances.populate_specific_size(n_reps)

            clustering_data_by_n_reps['agglomerative_clustering'][n_reps] = agglomerative_clustering(curr_rep_instances)
            clustering_data_by_n_reps['community_detection'][n_reps] = community_detection_clustering(curr_rep_instances)
        json.dump(clustering_data_by_n_reps, open(os.path.join(args.out_dir, f"{token}_clustering.json"), 'w'))
    except ValueError as e:
        print(e) # if token is in vocab but not in database.
        pass

def agglomerative_clustering(rep_instances):
    model = MyBOWHierarchicalLinkage()
    clusters = model.fit_predict(rep_instances)
    clustered_reps = model.reps_to_their_clusters(clusters, rep_instances)
    clustering_data = []
    for _, intances in clustered_reps.items():
        cluster_reps = [inst.reps for inst in intances]
        reps_counter = Counter(int(r) for reps in cluster_reps for r in reps)
        most_common_tokens = reps_counter.most_common(MOST_COMMON_CLUSTER_REPS)
        clustering_data.append(most_common_tokens)

    return clustering_data

def community_detection_clustering(rep_instances, query_n_reps=10):
    community_finder = CommunityFinder(rep_instances, query_n_reps)
    communities = community_finder.find()

    community_tokens, communities_sents_data = community_finder.argmax_voting(communities, rep_instances)

    community_tokens = sort_community_tokens_by_popularity(rep_instances, community_tokens)
    clustering_data = []
    for com_tokens, _ in zip(community_tokens, communities_sents_data):
        clustering_data.append([(int(t), v) for t, v in com_tokens[:MOST_COMMON_CLUSTER_REPS]])

    return clustering_data

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
    cluster_file = os.path.join(data_dir, OUT_FOLDER, f"{token}_clustering.json")
    return json.load(open(cluster_file, 'r'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="replacements")
    parser.add_argument("--dataset", type=str, choices=['CORD-19', 'Wikipedia-roberta', 'Wikipedia-BERT'])
    args = parser.parse_args()
    if args.dataset == 'Wikipedia-BERT':
        args.model_hf_path = 'bert-large-cased-whole-word-masking'
    else:
        raise NotImplementedError

    out_dir = os.path.join(args.data_dir, OUT_FOLDER)

    args.out_dir = out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    assert len(os.listdir(out_dir)) == 0, f"{out_dir} already exists."

    main(args)