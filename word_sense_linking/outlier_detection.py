import argparse
from itertools import product
from pathlib import Path
from gensim.models import KeyedVectors
from gensim import matutils
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, cpu_count

import numpy as np
from numpy import float32, vstack, dot

def main(args):
    dataset = read_dataset(args)
    print("Done reading dataset")

    if 'sense' in args.model_path.name:
        scorer = SensefulW2VSimilarityScorer(args.model_path)
    else:
        scorer = W2VSimilarityScorer(args.model_path)
    print(f"Done initlizing scorer")
    scores_by_difficulty = scorer.score(dataset)

    print('difficulty', 'predicted_distractor', 'predicted_outlier')
    for difficulty, preds in scores_by_difficulty.items():
        print(difficulty+1, preds['predicted_distractor'], preds['predicted_outlier'])

    n_successes = sum([v['predicted_outlier'] for v in scores_by_difficulty.values()])

    print(f"\noverall {n_successes/(len(dataset)*8)}")

EMBS = None

class SimilarityScorer:
    def __init__(self, model_path):
        global EMBS
        EMBS = KeyedVectors.load(str(model_path), mmap='r')
        self.scores_by_difficulty = {}

    def score(self, dataset):
        for _, ex in tqdm(dataset.items()):
            self.score_example(ex)
        return self.scores_by_difficulty

    def score_example(self, ex):
        mean = self.find_mean(ex['inlier'])
        distractor = self.find_distractor(mean, ex['distractor'])
        for i in range(len(ex['outlier'])):
            pred_outlier = self.furthest_from_mean(mean, distractor, ex['outlier'][i])
            if i not in self.scores_by_difficulty:
                self.scores_by_difficulty[i] = {'predicted_distractor': 0, 'predicted_outlier': 0}
            if pred_outlier == ex['distractor']:
                self.scores_by_difficulty[i]['predicted_distractor'] += 1
            elif pred_outlier == ex['outlier'][i]:
                self.scores_by_difficulty[i]['predicted_outlier'] += 1
            else:
                raise Exception("Processing fault")

    def find_mean(self, words):
        raise NotImplementedError

    def furthest_from_mean(self, mean, distractor, outlier):
        raise NotImplementedError

    def find_distractor(self, mean, ex_distractor):
        raise NotImplementedError

    # From Gensim model.doesnt_match
    def _find_mean(self, words):
        used_words = []
        for word in words:
            if word in EMBS:
                used_words.append(word)
            elif word.title() in EMBS:
                used_words.append(word.title())
        if len(used_words) != len(words):
            ignored_words = set(words) - set(used_words)
            print(ignored_words)
        if not used_words:
            raise ValueError("cannot select a word from an empty list")
        vectors = vstack([EMBS.get_vector(word, norm=True) for word in used_words]).astype(float32)
        mean = matutils.unitvec(vectors.mean(axis=0)).astype(float32)
        similarities = dot(vectors, mean)
        similarities_mean = similarities.mean()

        return mean, similarities_mean

    def _furthest_from_mean(self, mean, distractor, outlier):
        possible_outlier = [distractor, outlier]
        for x in possible_outlier:
            if x not in EMBS:
                return x

        ranks = self.rank_dists(mean, possible_outlier)
        return ranks[-1][1]

    def rank_dists(self, mean, possible_outlier):
        vectors = vstack([EMBS.get_vector(word, norm=True) for word in possible_outlier]).astype(float32)
        similarities = dot(vectors, mean)

        ranks = sorted(zip(similarities, possible_outlier), reverse=True)
        return ranks

class W2VSimilarityScorer(SimilarityScorer):
    def __init__(self, model_path):
        super(W2VSimilarityScorer, self).__init__(model_path)

    def find_mean(self, words):
        mean, _ = self._find_mean(words)

        return mean

    def furthest_from_mean(self, mean, distractor, outlier):
        return self._furthest_from_mean(mean, distractor, outlier)

    def find_distractor(self, mean, ex_distractor):
        return ex_distractor

class SensefulW2VSimilarityScorer(SimilarityScorer):
    def __init__(self, model_path):
        super(SensefulW2VSimilarityScorer, self).__init__(model_path)

    def find_mean(self, words):
        words_with_senses = [self.word_senses(word) for word in words if word in EMBS]
        words_with_senses_product = (list(product(*words_with_senses)))
        with Pool(cpu_count()) as p:
            means, similarities_means = zip(*p.map(self._find_mean, words_with_senses_product))
        closest = np.argmax(similarities_means)
        print(words_with_senses_product[closest])

        return means[closest]

    def word_senses(self, word):
        def all_senses(w):
            return [w, *[f"{w}_{i}" for i in range(10)]]
        candidates = [*all_senses(word), *all_senses(word.title()), *all_senses(word.upper())]
        ret = [c for c in candidates if c in EMBS.key_to_index]
        return ret

    def closest_sense_to_mean(self, mean, word):
        word_with_senses = self.word_senses(word)
        sense_ranks = self.rank_dists(mean, word_with_senses)
        print(sense_ranks)
        return sense_ranks[0]

    def find_distractor(self, mean, ex_distractor):
        return self.closest_sense_to_mean(mean, ex_distractor)

    def furthest_from_mean(self, mean, closest_distractor, outlier):
        closest_outlier = self.closest_sense_to_mean(mean, outlier)

        if closest_distractor[0] > closest_outlier[0]:
            # print("Success: Predicted outlier")
            return outlier
        print("Fail: Predicted distractor")
        return closest_distractor[1].split('_')[0].lower()

def read_dataset(args):
    ret = {}
    for file in args.dataset.iterdir():
        if file.suffix == '.txt':
            with open(file, 'r') as f:
                terms = f.readlines()
                split_index = terms.index('\n')
                ret[file.stem] = {
                    'inlier': [w.strip() for w in terms[:split_index-1]],
                    'distractor': terms[split_index-1].strip(),
                    'outlier': [w.strip() for w in terms[split_index+1:]],
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