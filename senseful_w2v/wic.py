import argparse
import csv
from itertools import product
import json
from pathlib import Path
from tqdm import tqdm

import numpy as np
from numpy import dot
from gensim.models import KeyedVectors

STOPWORDS = ['ourselves', 'hers', 'between', 'Between', 'yourself', 'Yourself', 'but', 'But', 'again', 'Again', 'there', 'There', 'about', 'About', 'once', 'Once', 'during', 'During', 'out', 'Out', 'very', 'Very', 'having', 'Having', 'with', 'With', 'they', 'They', 'own', 'Own', 'an', 'An', 'be', 'Be', 'some', 'Some', 'for', 'For', 'do', 'Do', 'its', 'Its', 'yours', 'Yours', 'such', 'Such', 'into', 'Into', 'of', 'Of', 'most', 'Most', 'itself', 'other', 'Other', 'off', 'Off', 'is', 'Is', 'am', 'Am', 'or', 'Or', 'who', 'Who', 'as', 'As', 'from', 'From', 'him', 'Him', 'each', 'Each', 'the', 'The', 'themselves', 'until', 'Until', 'below', 'Below', 'are', 'Are', 'we', 'We', 'these', 'These', 'your', 'Your', 'his', 'His', 'through', 'Through', 'don', 'Don', 'nor', 'Nor', 'me', 'Me', 'were', 'Were', 'her', 'Her', 'more', 'More', 'himself', 'Himself', 'this', 'This', 'down', 'Down', 'should', 'Should', 'our', 'Our', 'their', 'Their', 'while', 'While', 'above', 'Above', 'both', 'Both', 'up', 'Up', 'to', 'To', 'ours', 'had', 'Had', 'she', 'She', 'all', 'All', 'no', 'No', 'when', 'When', 'at', 'At', 'any', 'Any', 'before', 'Before', 'them', 'Them', 'same', 'Same', 'and', 'And', 'been', 'Been', 'have', 'Have', 'in', 'In', 'will', 'Will', 'on', 'On', 'does', 'Does', 'then', 'Then', 'that', 'That', 'because', 'Because', 'what', 'What', 'over', 'Over', 'why', 'Why', 'so', 'So', 'did', 'Did', 'not', 'Not', 'now', 'Now', 'under', 'Under', 'he', 'He', 'you', 'You', 'herself', 'has', 'Has', 'just', 'Just', 'where', 'Where', 'too', 'Too', 'only', 'Only', 'myself', 'which', 'Which', 'those', 'Those', 'after', 'After', 'few', 'Few', 'whom', 'being', 'Being', 'if', 'If', 'theirs', 'my', 'My', 'against', 'Against', 'by', 'By', 'doing', 'Doing', 'it', 'It', 'how', 'How', 'further', 'Further', 'was', 'Was', 'here', 'Here', 'than', 'Than']

def main(args):
    dataset = read_dataset(args)
    lemmatized_vocab = prepare_lemmatized_vocab(dataset, args.split)

    embs = KeyedVectors.load(str(args.word_embeddings), mmap='r')
    
    preds = {k: [] for k in [x / 100.0 for x in range(50, 80, 2)]}

    for ex in tqdm(dataset):
        all_word_senses = word_senses(embs, lemmatized_vocab, ex['word'])
        if len(all_word_senses) == 0:
            for k in preds:
                preds[k].append(True)
            continue

        if len(all_word_senses) == 1:
            for k in preds:
                preds[k].append(True)
            continue

        sense_per_sent1 = most_likely_sense_per_sent_words(embs, lemmatized_vocab, all_word_senses, ex['sent1'], ex['sent1_word_loc'])
        sense_per_sent2 = most_likely_sense_per_sent_words(embs, lemmatized_vocab, all_word_senses, ex['sent2'], ex['sent2_word_loc'])
        sim = similarity(np.array(embs[all_word_senses[sense_per_sent1]]), np.array(embs[all_word_senses[sense_per_sent2]]))
        
        for k in preds:
            if k < sim:
                preds[k].append(True)
            else:
                preds[k].append(False)


        # if pred != ex['gold']:
        #     import ipdb; ipdb.set_trace()
        #     most_likely_sense_per_sent_words(embs, lemmatized_vocab, all_word_senses, ex['sent1'], ex['sent1_word_loc'])
        #     most_likely_sense_per_sent_words(embs, lemmatized_vocab, all_word_senses, ex['sent2'], ex['sent2_word_loc'])

    if args.split != 'test':
        print('Need to re-write and add the threshold')

    all_accuracies = {}
    for k in preds:
        print(f"Threshold {k}")
        all_accuracies[k] = compare_preds_to_gold(args, preds[k], dataset)
        
    best = max(all_accuracies.values())
    best_threshold = max(all_accuracies, key=all_accuracies.get)

    print()
    print('best: ', best)
    print('best_threshold: ', best_threshold)



def most_likely_sense_per_sent_words(embs, lemmatized_vocab, all_word_senses, sent, word_position):
    all_context_senses, all_context_embeddings = find_context_embeddings(embs, lemmatized_vocab, sent, word_position)
    all_sense_word_embs = [np.array(embs[w]) for w in all_word_senses]

    most_likley_word_sense = 0
    most_likley_word_sense_similarity = -1
    for i, word_emb in enumerate(all_sense_word_embs):
        similarity_with_context = [similarity(word_emb, context_word_embs) for context_word_embs in all_context_embeddings]
        if len(similarity_with_context) == 0:
            return 0
        sim = max(similarity_with_context)
        # max_sim_index = similarity_with_context.index(sim)
        # all_context_senses[max_sim_index]
        
        if sim > most_likley_word_sense_similarity:
            most_likley_word_sense = i
            most_likley_word_sense_similarity = sim

    return most_likley_word_sense

def similarity(word_embs, sent_embs, norm=True):
    if norm:
        word_embs /= np.linalg.norm(word_embs)
        sent_embs /= np.linalg.norm(sent_embs)
    return dot(word_embs, sent_embs)

def find_context_embeddings(embs, lemmatized_vocab, sent, word_position):
    bow_without_word = sent[:word_position] + sent[word_position+1:]
    bow_without_word_without_punct = [x for x in bow_without_word if len(x) > 1 and x not in STOPWORDS]
    bow_with_senses = [word_senses(embs, lemmatized_vocab, w) for w in bow_without_word_without_punct]
    flat_bow_with_senses = [word_sense for word_senses in bow_with_senses for word_sense in word_senses]
    return flat_bow_with_senses, [np.array(embs[word_sense]) for word_sense in flat_bow_with_senses]

def word_senses(embs, lemmatized_vocab, word):
    def all_senses(w):
        cands = [w, *[f"{w}_{i}" for i in range(4)]] # TODO MORE/LESS?
        return [w for w in cands if w in embs]

    ret = all_senses(word)
    if word in lemmatized_vocab:
        for w in all_senses(lemmatized_vocab[word]):
            if w not in ret:
                ret.append(w)
    return ret

def read_dataset(args):
    dataset = []
    with open(args.wic_dataset/args.split/f'{args.split}.data.txt', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            locs = row[2].split('-')
            sent1 = row[3].split()
            sent2 = row[4].split()

            ex = {
                'word': row[0],
                'sent1': sent1,
                'sent1_word_loc': int(locs[0]),
                'sent2': sent2,
                'sent2_word_loc': int(locs[1]),
            }
            dataset.append(ex)

    if args.split != 'test':
        with open(args.wic_dataset/args.split/f'{args.split}.gold.txt', 'r') as f:
            for i, r in enumerate(f):
                gold = r.strip()
                gold_label = True if gold == 'T' else False
                dataset[i]['gold'] = gold_label

    return dataset

def prepare_lemmatized_vocab(dataset, split):
    lemmaitized_vocab_path = Path(f'lemmatized_vocabs/for_wic-{split}.json')
    if lemmaitized_vocab_path.exists():
        return json.load(open(lemmaitized_vocab_path, 'r'))

    import spacy

    lemmas_dict = {}
    distinct_words = {w for ex in dataset for w in [*ex['sent1'], *ex['sent2']]}
    nlp = spacy.load("en_core_web_lg", disable=['ner', 'parser'])
    for word in distinct_words:
        spacy_token = nlp(word)[0]
        lemma = spacy_token.lemma_
        if lemma == '-PRON-':
            lemma = word
        lemmas_dict[word] = lemma

    json.dump(lemmas_dict, open(lemmaitized_vocab_path, 'w'))

    return lemmas_dict

def compare_preds_to_gold(args, preds, dataset):
    if args.split != 'test':
        return confusion_matrix(preds, [ex['gold'] for ex in dataset])
    else:
        with open('wic-preds.txt', 'w') as f:
            for pred in preds:
                if pred:
                    f.write('T\n')
                else:
                    f.write('F\n')

def confusion_matrix(preds, gold):
    both_positives, both_negatives, false_positives, false_negatives = 0, 0, 0, 0
    for p, g in zip(preds, gold):
        if p and g:
            both_positives += 1
        elif not p and not g:
            both_negatives += 1
        elif p and not g:
            false_positives += 1
        elif not p and g:
            false_negatives += 1
        else:
            print("Shouldn't get here")
    all = both_positives + both_negatives + false_positives + false_negatives

    # print('both_positives: ', 100*both_positives/all)
    # print('both_negatives: ', 100*both_negatives/all)
    # print('false_positives: ', 100*false_positives/all)
    # print('false_negatives: ', 100*false_negatives/all)

    accuracy = 100*(both_positives + both_negatives)/all
    print("Accuracy:", accuracy)

    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wic_dataset",
        type=Path,
        default='/home/matane/matan/dev/datasets/WiC/')
    parser.add_argument("--split",
        type=str,
        required=True,
        choices=['train', 'dev', 'test'])
    parser.add_argument("--word_embeddings",
        type=Path,
        required=True)
    # parser.add_argument("--similarity_threshold",
    #     type=float,
    #     default=0.99)
    args = parser.parse_args()
    main(args)