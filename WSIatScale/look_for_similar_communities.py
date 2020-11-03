# pylint: disable=no-name-in-module
# # pylint: disable=import-error
import argparse
import heapq
from functools import partial
import numpy as np
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count

from transformers import AutoTokenizer

from utils.utils import tokenizer_params, jaccard_score_between_elements
from WSIatScale.cluster_reps_per_token import read_clustering_data
from WSIatScale.create_inverted_index import full_words_tokens

from sklearn.feature_extraction import DictVectorizer

USE_TOP_N_WORDS_FROM_COMM = 10
N_CLOSEST_COMMS_TO_SAVE_PER_COMM = 10
CLOSEST_COMMS_DIR = 'closest_communities'

def main(args):
    model_hf_path = tokenizer_params[args.dataset]
    tokenizer = AutoTokenizer.from_pretrained(model_hf_path, use_fast=True)
    for method in ['community_detection', 'agglomerative_clustering']:
        for n_reps in ['5', '20', '50']:
            outdir = os.path.join(args.data_dir, CLOSEST_COMMS_DIR, f"{method}-{n_reps}")
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            global ALL_COMMUNITY_TOKENS
            ALL_COMMUNITY_TOKENS = find_all_community_tokens(tokenizer, args.data_dir, args.dataset, method, n_reps)
            partial_find_and_write = partial(find_and_write, outdir=outdir)
    
            with Pool(cpu_count()) as p:
                list(tqdm(p.imap(partial_find_and_write, ALL_COMMUNITY_TOKENS), total=len(ALL_COMMUNITY_TOKENS)))
    
def find_and_write(target_comm, outdir):
    closest_communities = find_closest_communities(target_comm)
    write_closest(outdir, target_comm, closest_communities)

def write_closest(outdir, target_comm, closest_communities):
    with open(os.path.join(outdir, target_comm), 'w') as f:
        for sim, comm_name in closest_communities:
            f.write(f"{comm_name}\t{round(sim, 4)}\n")

def find_closest_communities(target_comm):
    jaccard_sims_heap = []
    target_comm_tokens = ALL_COMMUNITY_TOKENS[target_comm]
    for comm_name, comm_tokens in ALL_COMMUNITY_TOKENS.items():
        if comm_name == target_comm: continue
        similarity = jaccard_score_between_elements(target_comm_tokens, comm_tokens)
        sim_tuple = (similarity, comm_name)
        if len(jaccard_sims_heap) <= N_CLOSEST_COMMS_TO_SAVE_PER_COMM:
            heapq.heappush(jaccard_sims_heap, sim_tuple)
        else:
            heapq.heappushpop(jaccard_sims_heap, sim_tuple)
    sorted_jaccard_sims = [heapq.heappop(jaccard_sims_heap) for _ in range(len(jaccard_sims_heap))]
    sorted_jaccard_sims.reverse()
    return sorted_jaccard_sims

def find_all_community_tokens(tokenizer, data_dir, dataset, method, n_reps):
    # This is a bit wastful. But it's alright atm.
    tokens_to_index = full_words_tokens(dataset, tokenizer)
    all_community_tokens = {}
    for token in tqdm(tokens_to_index):
        word = tokenizer.decode([token])
        clustering_data = read_clustering_data(data_dir, token)

        curr_clustering_data = clustering_data[method][n_reps]
        curr_clustering_data = [community for community in curr_clustering_data if len(community) > 1 and community[1][1] > 10] #heuristics
        for cluster_id, community in enumerate(curr_clustering_data):
            community_tokens = set([token] + [t for t, _ in community[:USE_TOP_N_WORDS_FROM_COMM]])
            all_community_tokens[f"{word}-{cluster_id}"] = community_tokens
    return all_community_tokens

def read_close_communities(data_dir, token, cluster_idx, method, n_reps):
    outdir = os.path.join(data_dir, CLOSEST_COMMS_DIR, f"{method}-{n_reps}")
    with open(os.path.join(outdir, f"{token}-{cluster_idx}"), 'r') as f:
        lines = f.readlines()

    lines = [l.rstrip().split('\t') for l in lines]

    return lines

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="replacements")
    parser.add_argument("--dataset", type=str, choices=['CORD-19', 'Wikipedia-roberta', 'Wikipedia-BERT'])
    args = parser.parse_args()

    main(args)