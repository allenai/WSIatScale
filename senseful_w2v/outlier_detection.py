import argparse
from itertools import product, combinations
from pathlib import Path
from gensim.models import KeyedVectors
from gensim import matutils
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, cpu_count, Manager

import numpy as np
from numpy import float32, vstack, dot
N_INLIERS = 8
N_OUTLIERS = 8

def main(args):
    dataset = read_dataset(args)
    dataset_size = len(dataset) * N_OUTLIERS
    print("Done reading dataset")

    if 'sense' in args.model_path.name:
        scorer = SensefulW2VSimilarityScorer(args.model_path)
    else:
        scorer = W2VSimilarityScorer(args.model_path)
    print(f"Done initlizing scorer")
    opp = scorer.score(dataset)

    print("OPP: ", sum(opp) / N_INLIERS / dataset_size * 100)
    print("Accuracy: ", sum([v == N_INLIERS for v in opp])/dataset_size * 100)

class SimilarityScorer:
    def __init__(self, model_path):
        self.embs = KeyedVectors.load(str(model_path), mmap='r')
        self.opp = []
        self.cached_similarities = None

    def score(self, dataset):
        for _, ex in tqdm(dataset.items()):
            self.score_example(ex)
        return self.opp

    def score_example(self, ex):
        for i in range(len(ex['outliers'])):
            curr_outlier = ex['outliers'][i]
            candidates = [*ex['inliers'], ex['distractor'], curr_outlier]
            inv_compactness_scores = self.inv_candidates_compactness_scores(candidates)

            _, sorted_candidates = zip(*sorted(zip(inv_compactness_scores, candidates), reverse=True))
            curr_opp = sorted_candidates.index(curr_outlier)

            self.opp.append(curr_opp)

            # if curr_opp != 8:
            #     print("inliers:", ex['inliers'], "distractor", ex['distractor'])
            #     print(f"outlier position {curr_outlier}: {curr_opp}")
            #     print("Positions:", sorted_candidates)
            #     print()

    def similarity(self, w_i, w_j):
        return dot(self.get_vector_with_fallback(w_i, True), self.get_vector_with_fallback(w_j, True))

    def inv_candidates_compactness_scores(self, candidates):
        compactness_scores = []
        for i in range(len(candidates)):
            word = candidates[i]
            W_minus_w = candidates[:i] + candidates[i+1:]
            pw = self.inv_compactness_score(word, W_minus_w)

            compactness_scores.append(pw)

        return compactness_scores

    def inv_compactness_score(self, word, W):
        raise NotImplementedError

    def get_vector_with_fallback(self, key, norm):
        if key in self.embs:
            return self.embs.get_vector(key, norm=norm)
        else:
            return self.embs.get_vector('UNK')

class W2VSimilarityScorer(SimilarityScorer):
    def __init__(self, model_path):
        super(W2VSimilarityScorer, self).__init__(model_path)

    def inv_compactness_score(self, word, W_minus_w):
        return sum([self.similarity(word, w_tag) for w_tag in W_minus_w])

class SensefulW2VSimilarityScorer(SimilarityScorer):
    def __init__(self, model_path):
        super(SensefulW2VSimilarityScorer, self).__init__(model_path)

    def inv_compactness_score(self, word, W_minus_w):
        word_with_senses = self.word_senses(word)
        W_minus_w_with_senses = [self.word_senses(word) for word in W_minus_w]
        W_minus_w_with_senses = [l for l in W_minus_w_with_senses if len(l) > 0]
        inv_compactnesses = []
        for word_with_sense in word_with_senses:
            inv_compactness_sum = 0
            for w_tags in W_minus_w_with_senses:
                inv_compactness_sum += max([self.similarity(word_with_sense, w_tag) for w_tag in w_tags])
            inv_compactnesses.append(inv_compactness_sum)

        if inv_compactnesses:
            return max(inv_compactnesses)
        return 0

    def word_senses(self, word):
        def all_senses(w):
            return [w, *[f"{w}_{i}" for i in range(3)]]
        candidates = [*all_senses(word), *all_senses(word.title()), *all_senses(word.upper())]
        ret = [c for c in candidates if c in self.embs.key_to_index]
        return ret

def read_dataset(args):
    ret = {}
    for file in args.dataset.iterdir():
        if file.suffix == '.txt':
            with open(file, 'r') as f:
                terms = f.readlines()
                split_index = terms.index('\n')
                ret[file.stem] = {
                    'inliers': [w.strip() for w in terms[:split_index-1]],
                    'distractor': terms[split_index-1].strip(),
                    'outliers': [w.strip() for w in terms[split_index+1:]],
                }
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
        type=Path,
        default='/home/matane/matan/dev/datasets/OutlierDetectionDataset/groups')
    parser.add_argument("--model_path",
        type=Path,
        required=True)

    args = parser.parse_args()
    main(args)