import argparse
from collections import defaultdict
from enum import Enum
import numpy as np
import os
import termplotlib as tpl
from tqdm import tqdm

from sklearn.cluster import AgglomerativeClustering

from transformers import AutoTokenizer
NUM_PREDS = 100

class Color(Enum):
    green = '\x1b[32m'
    orange = '\x1b[33m'
    back_to_white = '\x1b[00m'

    def __str__(self):
        return f"{self.value}"

class Jaccard:
    def __init__(self):
        self.matrix = None

    def init_matrix(self, length):
        self.matrix = np.zeros((length, length), dtype=np.float16)

    def pairwise_distance(self, X):
        length = len(X)
        self.init_matrix(length)
        for i in range(length):
            for j in range(i+1, length):
                distance = self.distance(X[i], X[j])
                self.matrix[i, j] = distance
                self.matrix[j, i] = distance

        return self.matrix

    def distance(self, x, y):
        return 1 - self.similarity(x, y)

    def similarity(self, x, y):
        intersection = len(x.intersection(y))
        union = len(x) + len(y) - intersection
        return float(intersection) / union

def main(args):
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', use_fast=True)

    token = tokenizer.encode(args.word, add_special_tokens=False)
    if len(token) > 1:
        raise Exception('Word given is more than a single wordpiece.')

    cache_file = os.path.join(args.cache_dir, f"{args.word}_{args.n_alters}_{args.stop_after_n_matches}.npy")
    bag_of_alters = (
        np.load(cache_file, allow_pickle=True).item() if os.path.exists(cache_file)
        else read_files(args, tokenizer, token, cache_file)
    )

    if args.report_alters_diversity:
        print_bag_of_alters(args, bag_of_alters, tokenizer)

    if args.cluster:
        cluster(args, bag_of_alters, tokenizer)

def read_files(args, tokenizer, token, cache_file):
    bag_of_alters = defaultdict(list)

    with open(os.path.join(args.preds_dir, 'in_words.npy'), 'rb') as words_f, \
         open(os.path.join(args.preds_dir, 'sent_lengths.npy'), 'rb') as lengths_f, \
         open(os.path.join(args.preds_dir, 'pred_ids.npy'), 'rb') as preds_f, \
         open(os.path.join(args.preds_dir, 'probs.npy'), 'rb') as probs_f:

        fsz = os.fstat(words_f.fileno()).st_size
        pbar = tqdm(total=fsz+1)

        prev_words_offset = 0
        preds_offset = 0
        n_matches = 0

        while words_f.tell() < fsz:
            tokens = np.load(words_f)
            lengths = np.load(lengths_f)

            token_idx_in_row = find_token_idx_in_row(tokens, token)
            n_matches += len(token_idx_in_row)

            if len(token_idx_in_row) != 0:
                preds = seek_and_load(preds_f, preds_offset)
                probs = seek_and_load(probs_f, preds_offset)

                sent_and_positions = list(find_sent_and_positions(token_idx_in_row, tokens, lengths))

                if args.print:
                    print_sents_with_token(args, tokenizer, sent_and_positions, preds, probs)
                populate_bag_of_alters(args, bag_of_alters, sent_and_positions, preds)

            preds_offset += tokens.shape[0] * 2 * NUM_PREDS + 128

            pbar.update(words_f.tell()-prev_words_offset)
            prev_words_offset = words_f.tell()

            if args.stop_after_n_matches is not None and args.stop_after_n_matches < n_matches:
                break

        save_to_cache(bag_of_alters, cache_file)
    pbar.close()

    return bag_of_alters

def save_to_cache(bag_of_alters, cache_file):
    cache_dir = '/'.join(cache_file.split('/')[:-1])
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    np.save(cache_file, bag_of_alters)

def seek_and_load(file, preds_offset):
    file.seek(preds_offset)
    return np.load(file)

def cluster(args, bag_of_alters, tokenizer):
    top_n_to_cluster = args.top_n_to_cluster
    n_clusters = args.n_clusters
    distance_threshold = args.distance_threshold

    sorted_bag_of_alters = [k for k, _ in \
        sorted(bag_of_alters.items(), key=lambda kv: len(kv[1]), reverse=True)[:top_n_to_cluster]
    ]
    jaccard_matrix = Jaccard().pairwise_distance(sorted_bag_of_alters)

    clustering = AgglomerativeClustering(n_clusters=n_clusters,
                                         distance_threshold=distance_threshold,
                                         affinity='precomputed',
                                         linkage='average')
    clusters = clustering.fit_predict(jaccard_matrix)
    clustered_alters = defaultdict(list)
    for c, alter in zip(clusters, sorted_bag_of_alters):
        clustered_alters[c].append(alter)

    for i, cluster_alters in enumerate(clustered_alters.values()):
        print(f"Cluster {i}:")
        for alters in cluster_alters:
            print(tokenizer.decode(list(alters)))
        print()

def populate_bag_of_alters(args, bag_of_alters, sent_and_positions, preds):
    for sent, token_pos_in_sent, global_token_pos in sent_and_positions:
        for local_pos, global_pos in zip(token_pos_in_sent, global_token_pos):
            single_sent = find_single_sent_around_token(sent, local_pos)
            key = frozenset(preds[global_pos][:args.n_alters])
            value = single_sent
            bag_of_alters[key].append(value)

def find_single_sent_around_token(concated_sents, local_pos):
    full_stop_token = 205
    full_stops_indices = np.where(concated_sents == full_stop_token)[0]
    if len(full_stops_indices) == 0:
        return concated_sents
    end_index = full_stops_indices.searchsorted(local_pos)
    start = full_stops_indices[end_index-1]+1 if end_index != 0 else 0
    if end_index == len(full_stops_indices):
        return concated_sents[start:]
    end = full_stops_indices[end_index]+1
    return concated_sents[start:end]

def print_bag_of_alters(args, bag_of_alters, tokenizer):
    n_sents_to_print = args.n_sents_to_print
    n_bow_alters_to_report = args.n_bow_alters_to_report

    for k, sent_and_positions in sorted(bag_of_alters.items(), key=lambda kv: len(kv[1]), reverse=True)[:n_bow_alters_to_report]:
        print(f"{Color.orange}{tokenizer.decode(list(k))} - {len(sent_and_positions)}{Color.back_to_white}")
        for sent in sent_and_positions[:n_sents_to_print]:
            print(tokenizer.decode(sent))
        print()

    alters_to_num = defaultdict(int)
    for _, v in bag_of_alters.items():
        alters_to_num[len(v)] += 1

    keys, values = zip(*sorted(alters_to_num.items(), key=lambda kv: kv[0], reverse=True))
    if len(keys) > n_bow_alters_to_report:
        keys = keys[:int(n_bow_alters_to_report/2)] + keys[-int(n_bow_alters_to_report/2):]
        values = values[:int(n_bow_alters_to_report/2)] + values[-int(n_bow_alters_to_report/2):]

    print(f"Top {int(n_bow_alters_to_report/2)} and bottom {int(n_bow_alters_to_report/2)}")
    print(f"{values[0]} alternative/s appear/s {keys[0]} time/s.")
    fig = tpl.figure()
    fig.barh(list(values), keys)
    fig.show()

def print_sents_with_token(args, tokenizer, sent_and_positions, preds, probs):
    for sent, token_pos_in_sent, global_token_pos in sent_and_positions:
        splits = np.split(sent, token_pos_in_sent)
        out = tokenizer.decode(splits[0])
        for i, split in enumerate(splits[1:]):
            if out:
                out += ' '
            out += f"{Color.green}(#{i}) {args.word}{Color.back_to_white} " + tokenizer.decode(split[1:])
        for i, global_pos in enumerate(global_token_pos):
            out += f"\n{Color.orange}(#{i}):"
            for pred, prob in zip(preds[global_pos, :args.n_alters], probs[global_pos, :args.n_alters]):
                out += f" {tokenizer.decode([pred])}-{prob:.5f}"
            out += str(Color.back_to_white)
        print(out)

def find_token_idx_in_row(tokens, token):
    return np.where(tokens == token)[0]

def find_sent_and_positions(token_idx_in_row, tokens, lengths):
    length_sum = 0
    for length in lengths:
        token_location = token_idx_in_row[np.where(np.logical_and(token_idx_in_row >= length_sum, token_idx_in_row < length_sum + length))[0]]
        if len(token_location) > 0:
            yield tokens[length_sum:length_sum + length], token_location-length_sum, token_location
        length_sum += length

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--preds_dir", type=str, default="preds")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--word", type=str, required=True)
    parser.add_argument("--n_alters", type=int, required=True)
    parser.add_argument("--stop_after_n_matches", type=int, default=None)
    parser.add_argument("--print", action='store_true')
    parser.add_argument("--report_alters_diversity", action='store_true')
    parser.add_argument("--n_bow_alters_to_report", type=int, default=10, help="How many different alternatives to report")
    parser.add_argument("--n_sents_to_print", type=int, default=2, help="Num sents to prin for report_alters_diversity")
    parser.add_argument("--cluster", action='store_true')
    parser.add_argument("--top_n_to_cluster", type=int, default=100)
    parser.add_argument("--n_clusters", type=int)
    parser.add_argument("--distance_threshold", type=float)

    args = parser.parse_args()
    assert args.print or args.report_alters_diversity or args.cluster, \
        "At least one of `print`, `report_alters_diversity` `cluster` should be available"

    if args.cluster:
        assert (args.n_clusters is not None or args.distance_threshold is not None) \
            and (args.n_clusters is None or args.distance_threshold is None), \
            "Pass one of `n_clusters` and `distance_threshold`"

    if args.print:
        print("Warning:\n"\
              "This is deprecated. I don't think there's use for this anymore.\n"\
              "`report_alters_diversity` is more powerful.\n"
              "Additionaly, this will not work if the data is already cached.")

    main(args)