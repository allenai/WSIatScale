# pylint: disable=import-error
import argparse
import json
from pathlib import Path
import numpy as np
from operator import itemgetter

from transformers import AutoTokenizer

from utils.utils import jaccard_score_between_elements
from utils.special_tokens import SpecialTokens
from WSIatScale.analyze import RepInstances, Instance as RepInstance

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_hf_path, use_fast=True)
    special_tokens = SpecialTokens(args.model_hf_path)

    examples = list(read_examples(args.wictsv_dataset_path))
    hypernyms = list(read_hypernyms(args.wictsv_dataset_path, tokenizer))
    if args.split == 'Development':
        labels = list(read_labels(args.wictsv_dataset_path))
    else:
        labels = None

    # for precomputed_n_reps in ['5', '20', '50']:
    for precomputed_n_reps in ['5']:
        args.precomputed_n_reps = precomputed_n_reps
        hypernyms_reps = list(get_hypernyms_reps_of_all_senses(args, tokenizer, hypernyms))
        all_precomputed_clustered_reps = list(get_precomputed_clustered_reps(args, tokenizer, examples, special_tokens))

        average_acc = 0
        # n_reps_range = range(5, 101, 5)
        n_reps_range = [55]
        for n_reps in n_reps_range:
            args.n_reps = n_reps
            tokens_reps = get_wic_tsv_reps(args, special_tokens, [f"{i}-reps.npy" for i in range(len(examples))])

            text_reps_preds = list(predict_cluster_by_precomputed(all_precomputed_clustered_reps, tokens_reps))
            hypernyms_preds = list(predict_cluster_by_precomputed(all_precomputed_clustered_reps, hypernyms_reps))

            preds_agreement = [p1 == p2 for p1, p2 in zip(text_reps_preds, hypernyms_preds)]

            if labels:
                accuracy = run_evaluation(args, preds_agreement, labels, len(examples))
                average_acc += accuracy/len(n_reps_range)
            else:
                print_preds_to_file(args, preds_agreement)
        print(average_acc*100)

def run_evaluation(args, preds_agreement, labels, len_examples):
    correct = sum([p == l for p, l in zip(preds_agreement, labels)])
    acc = correct/len_examples
    print(f"Accuracy (n_reps = {args.n_reps}): {acc}")
    return acc

def print_preds_to_file(args, preds_agreement):
    preds = ['T\n' if p else 'F\n' for p in preds_agreement]
    with open(f'word_sense_linking/test_preds/hypernyms_output-{args.precomputed_n_reps}-{args.n_reps}.txt', 'w') as f:
        f.writelines(preds)

def predict_cluster_by_precomputed(all_precomputed_clustered_reps, refs_reps, print_reps=False, tokenizer=None):
    best_jaccard_is_zero = 0
    for precomputed_clustered_reps, ref_reps in zip(all_precomputed_clustered_reps, refs_reps):
        jaccard_scores = []
        for pre_computed_cluster in precomputed_clustered_reps:
            if print_reps:
                print('precomputed_cluster:', tokenizer.decode(list(pre_computed_cluster)))
            similarity = jaccard_score_between_elements(set(pre_computed_cluster), ref_reps)
            jaccard_scores.append(similarity)

        if jaccard_scores:
            cluster_id, best_jaccard_score = max(enumerate(jaccard_scores), key=itemgetter(1))
            if best_jaccard_score == 0:
                best_jaccard_is_zero+=1
        else:
            cluster_id = -1
        yield cluster_id


def get_hypernyms_reps_of_all_senses(args, tokenizer, hypernyms):
    for curr_hypernyms in hypernyms:
        any_sense_rep_of_the_hypernyms = set(curr_hypernyms)
        for i, hypernym in enumerate(curr_hypernyms):
            clusters_file = args.background_data_dir/'word_clusters'/f"{hypernym}_clustering.json"
            if clusters_file.is_file():
                clusters = json.load(open(clusters_file, 'r'))

                any_sense_rep_of_the_hypernyms.update([t for cluster in clusters['community_detection'][args.precomputed_n_reps] for t, c in cluster[:10]])

        yield any_sense_rep_of_the_hypernyms

def get_precomputed_clustered_reps(args, tokenizer, examples, special_tokens):
    for ex in examples:
        tokens = tokenizer.encode(ex[0], add_special_tokens=False) # TODO Not dealing with multi token words

        precomputed_clusters = []
        for token in tokens:
            # token = special_tokens.lemmatize_if_possible(token)
            clusters_file = args.background_data_dir/'word_clusters'/f"{token}_clustering.json"
            if clusters_file.is_file():
                clusters = json.load(open(clusters_file, 'r'))
                precomputed_clusters += [[t for t, c in cluster] for cluster in clusters['community_detection'][args.precomputed_n_reps]]

        yield precomputed_clusters

def get_wic_tsv_reps(args, special_tokens, files):
    rep_instances = RepInstances() #not lemmatizing
    for file in files:
        reps = np.load(args.processed_wic_tsv/'replacements'/file)
        # rep_instances.clean_and_populate_reps(reps, special_tokens)
        rep_instances.data.append(RepInstance(reps=reps))
        rep_instances.populate_specific_size(args.n_reps)

    return [r.reps for r in rep_instances.data]

def read_examples(dir_path):
    file = [f for f in dir_path.iterdir() if 'examples' in f.name][0]
    with open(file, 'r') as f:
        for row in f:
            row = row.split('\t')
            row[2] = row[2].strip()
            yield row

def read_hypernyms(dir_path, tokenizer):
    file = [f for f in dir_path.iterdir() if 'hypernyms' in f.name][0]
    with open(file, 'r') as f:
        for row in f:
            row = row.strip().replace('_', ' ').split('\t')
            encoded_hypernyms = [tokenizer.encode(w, add_special_tokens=False) for w in row]
            # Dealing with wordpieces by taking all of them.
            yield [wordpiece for word in encoded_hypernyms for wordpiece in word]


def path_by_split(path, split):
    return Path(path.replace('__SPLIT__', split))

def read_labels(dir_path):
    with open(dir_path/'dev_labels.txt', 'r') as f:
        for row in f:
            row = row.strip()
            if row == 'T':
                yield True
            elif row == 'F':
                yield False

def check_wrong(tokenizer, examples, hypernyms, all_precomputed_clustered_reps, tokens_reps, hypernyms_reps, ex_num):
    tokenizer = AutoTokenizer.from_pretrained('bert-large-cased-whole-word-masking', use_fast=True)
    print("Example:", examples[ex_num])
    print("Hypernyms:", tokenizer.decode(hypernyms[ex_num]))
    print("Decoded:", [tokenizer.decode([t]) for t in tokenizer.encode(examples[ex_num][0], add_special_tokens=False)])
    print("token reps:", [tokenizer.decode([t]) for t in tokens_reps[ex_num]])
    print("hypernym reps:", [tokenizer.decode([t]) for t in hypernyms_reps[ex_num]])
    text_reps_pred = list(predict_cluster_by_precomputed([all_precomputed_clustered_reps[ex_num]], [tokens_reps[ex_num]], print_reps=True, tokenizer=tokenizer))
    print("text_reps_pred", text_reps_pred)
    hypernyms_pred = list(predict_cluster_by_precomputed([all_precomputed_clustered_reps[ex_num]], [hypernyms_reps[ex_num]], print_reps=False, tokenizer=tokenizer))
    print('hypernyms_pred', hypernyms_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--background_data_dir",
        type=Path,
        required=True)
    parser.add_argument("--wictsv_dataset_path",
        type=str,
        default='/mnt/disks/mnt2/datasets/WiC_TSV_Data/__SPLIT__/')
    parser.add_argument("--processed_wic_tsv",
        type=str,
        default='/mnt/disks/mnt2/datasets/WiC_TSV_Data/__SPLIT__/bert/')
    parser.add_argument("--n_reps",
        type=int)
    parser.add_argument("--precomputed_n_reps",
        type=str,
        choices=['5', '20', '50'])
    parser.add_argument("--split",
        type=str,
        required=True,
        choices=['Development', 'Test'])
    args = parser.parse_args()

    args.wictsv_dataset_path = path_by_split(args.wictsv_dataset_path, args.split)
    args.processed_wic_tsv = path_by_split(args.processed_wic_tsv, args.split)
    args.model_hf_path = 'bert-large-cased-whole-word-masking'

    main(args)