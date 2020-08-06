# QuickBERTInference

## write_lm_mask_preds.py Script

### For running the forward pass and saving replacements to file run the following command with the additional optional arguments:
```
python write_lm_mask_preds.py --data_dir ../datasets/CORD-19/processed/raw-splits --input_file raw-splits --max_tokens_per_batch 16384
```

`max_tokens_per_batch=16384` is the max power of 2 that fits in a single P-100 GPU.

#### Mentionable Optional Arguments

`--iterativly_go_over_matrices`: An alternative method for going over predicted tokens and writing to disk. This is less optimized for PyTorch. Deprecated.

`--cpu`: Run on CPU, requires only when running on a machine with GPUs.

`--simple_sampler`: For a basic sampler (If using this, remove `max_tokens_per_batch` and choose a `batch_size` that fits your GPU).

`--fp16`: Running with half precision. Using `opt_level=O2`.

## analyze.py Script

```
python analyze.py --replacements_dir replacements --word race --inverted_index inverted_index.json --n_reps 5 --sample_n_files 1000 --cluster --distance_threshold 0.8
```

#### Mentionable Optional Arguments

`--sample_n_files`: recommended, stop after going over `n` files from the inverted index.

##### Pass one of the following options
`--print`: print all matches of `word`.

`--report_reps_diversity`: groups bags of replacements, prints an example and prints (bottom and top) the histogram. `--n_bow_reps_to_report` and `--n_sents_to_print` have default values of 10 and 1, change these if wishes.

`--cluster`: Cluster bag of replacements using Agglomerative Clustering (Pass one of `--n_clusters` and `--distance_threshold`). `--top_n_to_cluster` has a default value of 100 as we usually don't want to cluster all possible bag of replacements.
