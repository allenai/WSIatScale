# Train senseful word2vec

Start by created sense files that are aligned with the input files.

```
python -m WSIatScale.assign_clusters_to_tokens --data_dir /mnt/disks/mnt2/datasets/processed_for_WSI/wiki/bert/v2/ --dataset Wikipedia-BERT --run_specific_method community_detection --run_specific_n_reps 5 --write_aligned_sense_idx
```