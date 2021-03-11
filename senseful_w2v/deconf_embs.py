import bz2
import json
from tqdm import tqdm
from pathlib import Path


EMBS_DIR = Path('/home/matane/matan/dev/datasets/embeddings/DeConf/')
OUTLIER_DETECTION_WORDS = {'crocodile', 'elephant', 'gorilla', 'gazelle', 'lion', 'buffalo', 'leopard', 'python', 'lowland', 'albino', 'jungle', 'savanna', 'safari', 'animal', 'sandwich', 'coast', 'cow', 'sheep', 'goat', 'horse', 'pig', 'goose', 'donkey', 'turkey', 'tiger', 'hedgehog', 'sister', 'mineral', 'poultry', 'ghost', 'basket', 'pineapple', 'apple', 'banana', 'mango', 'fig', 'cherry', 'lemon', 'orange', 'potato', 'flower', 'branch', 'google', 'ireland', 'meat', 'adapter', 'acid', 'plane', 'car', 'truck', 'bicycle', 'van', 'bus', 'motorcycle', 'balloon', 'house', 'seat', 'journey', 'road', 'shore', 'tree', 'skyscraper', 'refrigerator', 'anarchist', 'communist', 'authoritarian', 'socialist', 'fascist', 'colonialist', 'libertarian', 'conservative', 'pessimist', 'buddhist', 'spectrum', 'economy', 'russia', 'onion', 'garage', 'hope', 'mother', 'daughter', 'father', 'son', 'grandmother', 'brother', 'aunt', 'mummy', 'family', 'adoption', 'relation', 'pregnancy', 'holiday', 'winter', 'country', 'phone', 'cannabis', 'heroin', 'opium', 'cocaine', 'morphine', 'lsd', 'mdma', 'ecstasy', 'methanol', 'urea', 'nitrogen', 'chlorine', 'rehabilitation', 'drugstore', 'mountain', 'teacher', 'zeus', 'hades', 'poseidon', 'aphrodite', 'ares', 'athena', 'artemis', 'nike', 'mercury', 'odysseus', 'jesus', 'sparta', 'delphi', 'rome', 'wrath', 'atlanta', 'parrot', 'hummingbird', 'owl', 'penguin', 'vulture', 'heron', 'stork', 'crane', 'bee', 'airplane', 'plumage', 'eggs', 'humans', 'academy', 'guitar', 'accordion', 'tuba', 'trumpet', 'saxophone', 'oboe', 'harp', 'keyboard', 'microphone', 'orchestra', 'festival', 'concert', 'ear', 'beer', 'industry', 'aluminum', 'arm', 'finger', 'leg', 'foot', 'eye', 'abdomen', 'head', 'blood', 'hormone', 'person', 'doctor', 'jewelry', 'injury', 'pants', 'skyscrapers', 'helium', 'argon', 'hydrogen', 'oxygen', 'fluorine', 'krypton', 'methane', 'silver', 'brass', 'atom', 'water', 'speak', 'asphalt', 'black', 'white', 'green', 'yellow', 'pink', 'purple', 'red', 'gold', 'hue', 'saturation', 'painter', 'canvas', 'pencil', 'pixel', 'complimentary', 'radioactive', 'magnesium', 'copper', 'iron', 'zinc', 'titanium', 'nickel', 'lead', 'wood', 'plastic', 'porcelain', 'sand', 'gasoline', 'colony', 'tyre', 'coffee', 'tea', 'wine', 'milk', 'whiskey', 'slush', 'alcohol', 'thirst', 'waiter', 'snack', 'fruit', 'waterfall', 'ocean', 'throne', 'shirt', 'hat', 'jacket', 'skirt', 'shorts', 'boot', 'cloth', 'pillow', 'cotton', 'paper', 'needlework', 'computer', 'germany', 'danube', 'rhine', 'volga', 'tiber', 'nile', 'mekong', 'zambezi', 'amazon', 'kilimanjaro', 'gobi', 'albania', 'sydney', 'airport', 'shark', 'bible', 'handball', 'badminton', 'canoeing', 'tennis', 'boxing', 'volleyball', 'basketball', 'squash', 'fitness', 'sweat', 'ball', 'team', 'deceleration', 'write', 'children', 'fear', 'anger', 'sadness', 'joy', 'disgust', 'boredom', 'nostalgia', 'spite', 'instinct', 'appearance', 'heart', 'funny', 'day', 'impulse', 'online', 'televisions', 'bean', 'lentils', 'tomatoes', 'garlic', 'chili', 'swede', 'peanuts', 'juice', 'glasses', 'nine', 'oman', 'iran', 'afghanistan', 'yemen', 'syria', 'lebanon', 'qatar', 'kabul', 'morocco', 'damascus', 'caucasus', 'religion', 'oil', 'meningitis', 'malaria', 'tuberculosis', 'smallpox', 'measles', 'diabetes', 'influenza', 'aids', 'headache', 'liver', 'hospital', 'bed', 'thin', 'outbreak', 'robot', 'estonia', 'france', 'italy', 'spain', 'sweden', 'finland', 'georgia', 'paris', 'scandinavia', 'africa', 'export', 'montgomery', 'honolulu', 'boston', 'austin', 'nashville', 'albany', 'phoenix', 'canada', 'oregon', 'fresno', 'denmark', 'europe', 'university', 'shrub', 'couch', 'armchair', 'sofa', 'desk', 'lamp', 'bathtub', 'bench', 'cabinet', 'bathroom', 'kitchen', 'garden', 'roof', 'wall', 'austria'}

should_have_been_lemmatized = {
    'lentil': 'lentils',
    'tomato': 'tomatoes',
    'child': 'children',
    'canoe': 'canoeing',
    'television': 'televisions',
    }

def main():
    word_senses_names = find_words_senses_names()
    embs = read_relevant_embs(word_senses_names)

    embs['skyscrapers'] = embs['skyscraper']

    json.dump(embs, open('senseful_w2v/word_vectors/deconf/deconf_embs.json', 'w'))

def find_words_senses_names():
    ret = {}
    sense_list = EMBS_DIR / 'sense_list.txt'
    with open(sense_list, 'r') as f:
        for row in tqdm(f):
            word, senses = row.split('\t')
            if word in OUTLIER_DETECTION_WORDS \
                or word.lower() in OUTLIER_DETECTION_WORDS \
                or word in should_have_been_lemmatized:
                sense_list = senses.strip().split()
                if word in should_have_been_lemmatized:
                    word = should_have_been_lemmatized[word]
                if word.lower() in ret:
                    ret[word.lower()] += sense_list
                else:
                    ret[word.lower()] = sense_list

    # for word in OUTLIER_DETECTION_WORDS:
    #     if word not in ret:
    #         print(f"Couldn't find word {word}")

    return ret
        

def read_relevant_embs(word_senses_names):
    embs = {word: {} for word in word_senses_names}
    all_sense_names = {name: word for word, senses in word_senses_names.items() for name in senses}
    with bz2.open(EMBS_DIR / 'sense_vectors.txt.bz2', 'r') as f:
        stats = f.readline().decode('utf-8').split()
        for row in tqdm(f, total=int(stats[0])):
            row = row.decode('utf-8')
            sense_name = row.split(maxsplit=1)[0]
            if sense_name in all_sense_names:
                embs[all_sense_names[sense_name]][sense_name] = [float(x) for x in row.split()[1:]]

    return embs

if __name__ == "__main__":
    main()