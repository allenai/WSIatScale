import argparse
import json
import logging
import math
import os
import subprocess
import tempfile
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from WSIatScale.analyze import read_files
from WSIatScale.clustering import MyBOWHierarchicalLinkage
from WSIatScale.community_detection import CommunityFinder


def main(args):
    evaluate_2010(args)
    # evaluate_2013(args)

def evaluate_2010(args):
    gold_dir = "/home/matane/matan/dev/SemEval/resources/SemEval-2010/evaluation/"
    labeling = label_2010(args)
    scores = evaluate_labeling_2010(gold_dir, labeling)
    fscore = scores['all']['FScore']
    v_measure = scores['all']['V-Measure']
    msg = 'SemEval 2010 FScore %.2f V-Measure %.2f AVG %.2f' % (fscore * 100, v_measure * 100, math.sqrt(fscore * v_measure) * 100)
    msg += '\n' + get_score_by_pos(scores)
    print(msg)

def evaluate_labeling_2010(dir_path, labeling: Dict[str, Dict[str, int]], key_path: str = None) \
        -> Tuple[Dict[str, Dict[str, float]], Tuple]:
    """
    similar to 2013 eval code, but only use top sense for each instnace
    """
    unsup_key = os.path.join(dir_path, 'unsup_eval/keys/all.key')

    with tempfile.NamedTemporaryFile('wt') as fout:
        lines = []
        for instance_id, clusters_str in labeling.items():
            lemma_pos = instance_id.rsplit('.', 1)[0]
            lines.append('%s %s %s' % (lemma_pos, instance_id, clusters_str))
        fout.write('\n'.join(lines))
        fout.flush()
        scores = get_2010_scores(dir_path, unsup_key, fout.name)
        if key_path:
            logging.info('writing key to file %s' % key_path)
            with open(key_path, 'w', encoding="utf-8") as fout2:
                fout2.write('\n'.join(lines))
    return scores

def get_2010_scores(dir_path, unsup_key, eval_key):
    ret = defaultdict(dict)

    for metric, jar in [
        ('FScore', os.path.join(dir_path, 'unsup_eval/fscore.jar')),
        ('V-Measure', os.path.join(dir_path, 'unsup_eval/vmeasure.jar'))
    ]:

        logging.info('calculating metric %s' % metric)
        res = subprocess.Popen(['java', '-jar', jar, eval_key, unsup_key, 'all'],
                                stdout=subprocess.PIPE).stdout.readlines()
        for line in res:
            line = line.decode().strip()
            split = line.split()
            if len(split) == 4 and split[0][-2:] in ['.n', '.v', '.j']:

                lemma = split[0]
                score = float(split[1])
                ret[lemma][metric] = score
            elif metric + ':' in line:
                ret['all'][metric] = float(line.split(metric + ':')[1])
    return ret

def label_2010(args):
    if args.labeling_function == 'clustering':
        labeling_function = bow_hierarchical_linkage_labelling
    else:
        labeling_function = community_detection_labelling
    instance_id_to_doc_id = json.load(open(os.path.join(args.data_dir2010, "instance_id_to_doc_id.json"), 'r'))
    lemmas = sorted(set([k.split('.')[0] for k in instance_id_to_doc_id.keys()]))

    return labeling_function(args,
        args.data_dir2010,
        lemmas,
        instance_id_to_doc_id)

def get_score_by_pos(results):
    res_string = ''
    for pos, pos_title in [('v', 'VERB'), ('n', 'NOUN'), ('j', 'ADJ')]:
        aggregated = defaultdict(list)
        for lemmapos in results:
            if lemmapos[-2:] == '.' + pos:
                for metric, score in results[lemmapos].items():
                    aggregated[metric].append(score)
        if aggregated:
            avg = 1
            for metric, listscores in aggregated.items():
                avg *= np.mean(listscores)
            avg = np.sqrt(avg)
            res_string += (f'{pos_title} ' + ' '.join(
                [f'{metric}: {np.mean(listscores)*100:.2f}' for metric, listscores in aggregated.items()]))
            res_string += f' AVG: {avg*100:.2f}\n'
    return res_string

def community_detection_labelling(args, data_dir, lemmas, instance_id_to_doc_id):
    model_hg_path, n_reps = args.model_hg_path, args.n_reps
    doc_id_to_inst_id = {v:k for k,v in instance_id_to_doc_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_hg_path, use_fast=True)

    labeling = {}
    # lemmas = ['market']
    for lemma in tqdm(lemmas):
        rep_instances, _ = read_files(lemma,
                                      data_dir,
                                      sample_n_files=-1,
                                      full_stop_index=None,
                                      should_lemmatize=True,
                                      bar=lambda x: x)
        rep_instances.populate_specific_size(n_reps)
        if args.remove_query_word or args.remove_stop_words:
            rep_instances.remove_certain_words(args.remove_query_word, args.remove_stop_words, tokenizer, lemma)
        community_finder = CommunityFinder(rep_instances, args.query_n_reps)
        communities = community_finder.find()

        _, communities_sents_data = community_finder.argmax_voting(communities, rep_instances)
        # community_tokens, communities_sents_data = community_finder.merge_small_clusters(community_tokens, communities_sents_data)

        for cluster, rep_instances in enumerate(communities_sents_data):
            for rep_inst in rep_instances:
                lemma_inst_id = doc_id_to_inst_id[rep_inst.doc_id]
                labeling[lemma_inst_id] = cluster

    return labeling

def bow_hierarchical_linkage_labelling(args, data_dir, lemmas, instance_id_to_doc_id):
    model_hg_path, n_reps = args.model_hg_path, args.n_reps
    model = MyBOWHierarchicalLinkage()
    doc_id_to_inst_id = {v:k for k,v in instance_id_to_doc_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_hg_path, use_fast=True)

    labeling = {}
    for lemma in tqdm(lemmas):
        rep_instances, _ = read_files(lemma,
                                              data_dir,
                                              sample_n_files=-1,
                                              should_lemmatize=True,
                                              full_stop_index=None,
                                              bar=lambda x: x)
        rep_instances.populate_specific_size(n_reps)
        if args.remove_query_word or args.remove_stop_words:
            rep_instances.remove_certain_words(args.remove_query_word, args.remove_stop_words, tokenizer, lemma)
        clusters = model.fit_predict(rep_instances)
        for doc_id, cluster in clusters.items():
            lemma_inst_id = doc_id_to_inst_id[doc_id]
            labeling[lemma_inst_id] = {cluster: 1}

    return labeling

# def evaluate_2013(args):
#     gold_dir = '/home/matane/matan/dev/SemEval/resources/SemEval-2013-Task-13-test-data'
#     labeling = label_2013(args)
#     scores = evaluate_labeling_2013(gold_dir, labeling)
#     fnmi = scores['all']['FNMI']
#     fbc = scores['all']['FBC']
#     msg = 'SemEval 2013 FNMI %.2f FBC %.2f AVG %.2f' % (fnmi * 100, fbc * 100, math.sqrt(fnmi * fbc) * 100)
#     # FNMI:21.4(0.5) FBC:64.0(0.5) Geom. mean:37.0(0.5)
#     # (previous SOTA 11.3,57.5,25.4)
#     print(msg)

# def label_2013(args):
#     instance_id_to_doc_id = json.load(open(os.path.join(args.data_dir2010, "instance_id_to_doc_id.json"), 'r'))
#     lemmas = set([k.split('.')[0] for k in instance_id_to_doc_id.keys()])
#     return community_detection_labelling(args.model_hg_path,
#         args.data_dir2013,
#         args.n_reps,
#         lemmas,
#         instance_id_to_doc_id)

# from xml.etree import ElementTree
# def read_semeval_2013(dir_path: str):
#     logging.info('reading SemEval dataset from %s' % dir_path)
#     # nlp = spacy.load("en", disable=['ner', 'parser'])
#     in_xml_path = os.path.join(dir_path, 'contexts/senseval2-format/semeval-2013-task-13-test-data.senseval2.xml')
#     gold_key_path = os.path.join(dir_path, 'keys/gold/all.key')
#     with open(in_xml_path, encoding="utf-8") as fin_xml, open(gold_key_path, encoding="utf-8") as fin_key:
#         instid_in_key = set()
#         for line in fin_key:
#             lemma_pos, inst_id, _ = line.strip().split(maxsplit=2)
#             instid_in_key.add(inst_id)
#         et_xml = ElementTree.parse(fin_xml)
#         for word in et_xml.getroot():
#             for inst in word.getchildren():
#                 inst_id = inst.attrib['id']
#                 if inst_id not in instid_in_key:
#                     # discard unlabeled instances
#                     continue
#                 context = inst.find("context")
#                 before, target, after = list(context.itertext())
#                 # before = [x.text for x in nlp(before.strip(), disable=['parser', 'tagger', 'ner'])]
#                 # target = target.strip()
#                 # after = [x.text for x in nlp(after.strip(), disable=['parser', 'tagger', 'ner'])]
#                 # yield before + [target] + after, len(before), inst_id
#                 yield inst_id


# def evaluate_labeling_2013(dir_path, labeling: Dict[str, Dict[str, int]], key_path: str = None) \
#         -> Tuple[Dict[str, Dict[str, float]], Tuple]:
#     """
#     labeling example : {'become.v.3': {'become.sense.1':3,'become.sense.5':17} ... }
#     means instance become.v.3' is 17/20 in sense 'become.sense.5' and 3/20 in sense 'become.sense.1'
#     :param key_path: write produced key to this file
#     :param dir_path: SemEval dir
#     :param labeling: instance id labeling
#     :return: FNMI, FBC as calculated by SemEval provided code
#     """
#     logging.info('starting evaluation key_path: %s' % key_path)

#     with tempfile.NamedTemporaryFile('wt') as fout:
#         lines = []
#         for instance_id, clusters_dict in labeling.items():
#             clusters = sorted(clusters_dict.items(), key=lambda x: x[1])
#             clusters_str = ' '.join([('%s/%d' % (cluster_name, count)) for cluster_name, count in clusters])
#             lemma_pos = instance_id.rsplit('.', 1)[0]
#             lines.append('%s %s %s' % (lemma_pos, instance_id, clusters_str))
#         fout.write('\n'.join(lines))
#         fout.flush()
#         gold_key_path = os.path.join(dir_path, 'keys/gold/all.key')
#         scores = get_2013_scores(dir_path, gold_key_path, fout.name)
#         if key_path:
#             logging.info('writing key to file %s' % key_path)
#             with open(key_path, 'w', encoding="utf-8") as fout2:
#                 fout2.write('\n'.join(lines))

#         # correlation = get_n_senses_corr(gold_key_path, fout.name)

#     return scores

# def get_2013_scores(dir_path, gold_key, eval_key):
#     ret = {}
#     for metric, jar, column in [
#         ('FNMI', os.path.join(dir_path, 'scoring/fuzzy-nmi.jar'), 1),
#         ('FBC', os.path.join(dir_path, 'scoring/fuzzy-bcubed.jar'), 3),
#     ]:
#         logging.info('calculating metric %s' % metric)
#         res = subprocess.Popen(['java', '-jar', jar, gold_key, eval_key], stdout=subprocess.PIPE).stdout.readlines()
#         for line in res:
#             line = line.decode().strip()
#             if line.startswith('term'):
#                 pass
#             else:
#                 split = line.split('\t')
#                 if len(split) > column:
#                     word = split[0]
#                     result = split[column]
#                     if word not in ret:
#                         ret[word] = {}
#                     ret[word][metric] = float(result)

#     return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir2010", type=str)
    parser.add_argument("--data_dir2013", type=str, default='/home/matane/matan/dev/WSIatScale/write_mask_preds/out/SemEval2013/bert-large-uncased')
    parser.add_argument("--n_reps", type=int, default=20)
    parser.add_argument("--query_n_reps", type=int, default=10)
    parser.add_argument("--model_hg_path", type=str, default='bert-large-uncased')
    parser.add_argument("--labeling_function", type=str, choices=['clustering', 'communities'])
    parser.add_argument("--remove_query_word", action='store_true')
    parser.add_argument("--remove_stop_words", action='store_true')
    args = parser.parse_args()
    main(args)