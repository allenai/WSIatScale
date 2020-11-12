# QuickBERTInference

## write_mask_preds/write_mask_preds.py Script

### For running the forward pass and saving replacements to file run the following command with the additional optional arguments:
This script needs to to be ran from inside the `write_mask_preds` folder, to be consistent with running using the Dockerfile on Beaker.

```
python write_mask_preds.py --data_dir ../../datasets/wiki-ann/wiki-ann/ --starts_with wiki --out_dir ../../datasets/processed_for_WSI/wiki/wiki000/replacements --dataset wiki --max_tokens_per_batch 16384 --fp16 --files_range 100-109
```

`max_tokens_per_batch=16384` is the max power of 2 that fits in a single P-100 GPU.

#### Mentionable Optional Arguments

`--iterativly_go_over_matrices`: An alternative method for going over predicted tokens and writing to disk. This is less optimized for PyTorch. Deprecated.

`--cpu`: Run on CPU, requires only when running on a machine with GPUs.

`--simple_sampler`: For a basic sampler (If using this, remove `max_tokens_per_batch` and choose a `batch_size` that fits your GPU).

`--fp16`: Running with half precision. Using `opt_level=O2`.

`--starts_with`: file starts with string.

`--files_range`: important for datasets with a lot of files where I want to split it between processes.

## analyze.py Script

This is accessed from `app.py`

## app.py Script

Run by `streamlit run app.py`

## at_scale_app.py

We first want to index all words in the LM vocab with something like this:

```
python WSIatScale/create_inverted_index.py --replacements_dir .../replacements/ --outdir .../inverted_index --dataset Wikipedia-BERT
```

Then we want to precompute clusters for all words:
```
python -m WSIatScale.cluster_reps_per_token --data_dir .../bert/ --dataset Wikipedia-BERT
```

And finally we want to find all instances of words by cluster:
```
python -m WSIatScale.assign_clusters_to_tokens --data_dir /mnt/disks/mnt2/datasets/processed_for_WSI/wiki/bert/ --dataset Wikipedia-BERT
```

After that, opionally we can find for each community its closest communities:
```
python -m WSIatScale.look_for_similar_communities --data_dir /mnt/disks/mnt2/datasets/processed_for_WSI/wiki/bert/ --dataset Wikipedia-BERT
```


# Quick Access For debugging
```
from transformers import AutoTokenizer; model_hf_path = 'bert-large-cased-whole-word-masking'; tok = AutoTokenizer.from_pretrained(model_hf_path, use_fast=True)
tok.encode('bass', add_special_tokens=False)[0] == 2753
```