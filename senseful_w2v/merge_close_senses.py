import argparse
from pathlib import Path
from gensim.models import KeyedVectors
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

def main(args):
    new_embs_names = []
    new_embs_vectors = []
    embs = KeyedVectors.load(str(args.model_path), mmap='r')
    word_to_sense_mapping = find_word_to_sense_mapping(embs)
    total_number_of_word_senses = 0
    new_number_of_word_senses = 0
    total_number_of_word_senses_with_few_senses = 0
    new_number_of_word_senses_with_few_senses = 0

    for _, word_with_senses in tqdm(word_to_sense_mapping.items()):
        merged_senses_names, merge_sense_vectors, _ = merge_close_senses(embs, word_with_senses, args.cutoff)
        
        total_number_of_word_senses += len(word_with_senses)
        new_number_of_word_senses += len(merged_senses_names)

        if len(word_with_senses) > 1:
            total_number_of_word_senses_with_few_senses += len(word_with_senses)
            new_number_of_word_senses_with_few_senses += len(merged_senses_names)
        new_embs_names += merged_senses_names
        new_embs_vectors += merge_sense_vectors

    print(f"Total number of word senses: ", total_number_of_word_senses)
    print(f"New number of word senses: ", new_number_of_word_senses)
    print(f"Shrinkage", (total_number_of_word_senses-new_number_of_word_senses)/total_number_of_word_senses)
    print(f"Shrinkage of few senses", (total_number_of_word_senses_with_few_senses-new_number_of_word_senses_with_few_senses)/total_number_of_word_senses_with_few_senses)

    kv = KeyedVectors(100)
    kv.add_vectors(new_embs_names, new_embs_vectors)
    kv.save(str(args.out_embs))

def find_word_to_sense_mapping(embs):
    mapping = {}
    for key_with_sense in embs.key_to_index.keys():
        if key_with_sense == '_': continue
        key = key_with_sense.split('_')[0]
        if key not in mapping:
            mapping[key] = []
        mapping[key].append(key_with_sense)

    for k in mapping.keys():
        mapping[k] = sorted(mapping[k], key=senses_comparator)

    return mapping

def senses_comparator(x):
    if '_' not in x:
        return 100
    return int(x.split('_')[1])

def merge_close_senses(embs, word_with_senses, cutoff):
    merged_sense_into = {}
    sense_vectors = [embs[s] for s in word_with_senses]
    
    merged_senses_names = [*word_with_senses]
    merge_sense_vectors = [*sense_vectors]
    
    similarities, closest_index, vectors_to_merge = find_closest_vectors(merge_sense_vectors)
    
    while similarities[closest_index] > cutoff:
        merged_sense_into[merged_senses_names[closest_index[1]]] = merged_senses_names[closest_index[0]]
        del merged_senses_names[closest_index[1]]
        del merge_sense_vectors[closest_index[1]]
        merge_sense_vectors[closest_index[0]] = np.mean(vectors_to_merge, 0)

        similarities, closest_index, vectors_to_merge = find_closest_vectors(merge_sense_vectors)

    return merged_senses_names, merge_sense_vectors, merged_sense_into

def find_closest_vectors(sense_vectors):
    similarities = cosine_pdist(sense_vectors)
    closest_index = np.unravel_index(np.argmax(similarities, axis=None), similarities.shape)
    vectors_to_merge = sense_vectors[closest_index[0]], sense_vectors[closest_index[1]]
    return similarities, closest_index, vectors_to_merge

def cosine_pdist(sense_vectors):
    similarities = 1 - pdist(sense_vectors, 'cosine')
    return squareform(similarities)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
        type=Path,
        required=True)
    parser.add_argument("--out_embs",
        type=Path,
        required=True)
    parser.add_argument("--cutoff",
        type=float)
    args = parser.parse_args()

    assert 'senseful' in str(args.out_embs), "You'd want the word senseful in the out embeddings"

    main(args)