# pylint: disable=no-member
import streamlit as st
import time

tokenizer_params = {'CORD-19': 'allenai/scibert_scivocab_uncased',
                    'Wikipedia-roberta': 'roberta-large',
                    'Wikipedia-BERT': 'bert-large-cased-whole-word-masking',}

class StreamlitTqdm:
    def __init__(self, iterable):
        self.prog_bar = st.progress(0)
        self.iterable = iterable
        self.length = len(iterable)
        self.i = 0

    def __iter__(self):
        for obj in self.iterable:
            yield obj
            self.i += 1
            current_prog = self.i / self.length
            self.prog_bar.progress(current_prog)

def sort_two_lists_by_one(l1, l2, key, reverse):
    return zip(*sorted(zip(l1, l2), key=key, reverse=reverse))

def jaccard_score_between_elements(set1, set2):
    intersection_len = len(set1.intersection(set2))
    union_len = len(set1) + len(set2) - intersection_len
    return intersection_len / union_len

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed