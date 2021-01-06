# Train senseful word2vec

Start by creating sense files that are aligned with the input files.

```
python -m WSIatScale.assign_clusters_to_tokens --data_dir /mnt/disks/mnt2/datasets/processed_for_WSI/wiki/bert/v2/ --dataset Wikipedia-BERT --run_specific_method community_detection --run_specific_n_reps 5 --write_aligned_sense_idx
```

Then `train`
```
python -m senseful_w2v.train --data_dir /mnt/disks/mnt1/datasets/CORD-19/scibert --processed_sents_cache_dir /mnt/disks/mnt1/datasets/CORD-19/scibert/aligned_sense_idx/processed_sents/ --dataset CORD-19
```

You can check performance on the outlier detection dataset

```
python -u senseful_w2v/outlier_detection.py --model_path ...
```

possible modelpaths are:
```
senseful_w2v/word_vectors/senseful_w2v.word_vectors-10epochs
senseful_w2v/word_vectors/senseful_w2v.word_vectors-10epochs-100dim-SG
senseful_w2v/word_vectors/word2vec-google-news-300
senseful_w2v/word_vectors/glove-wiki-gigaword-300
/home/matane/matan/dev/datasets/NASARI_embeddings/babelnet_api/outlier_detection_nasari_embs.json
```