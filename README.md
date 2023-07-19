# WSIatScale

## write_mask_preds/write_mask_preds.py Script

### For running the forward pass and saving replacements to file run the following command with the additional optional arguments:
This script needs to to be ran from inside the `write_mask_preds` folder, to be consistent with running using the Dockerfile on Beaker.

```
python write_mask_preds.py --data_dir ../../datasets/wiki-ann/wiki-ann/ --starts_with wiki --out_dir ../../datasets/processed_for_WSI/wiki/wiki000/replacements --dataset wiki --max_tokens_per_batch 16384 --fp16 --files_range 100-109 --model bert-large-cased-whole-word-masking
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

First thing, create lemmatized vocab (~1 minute):
```
python WSIatScale/create_lemmatized_vocab.py --model bert-large-cased-whole-word-masking --outdir lemmatized_vocabs/
```

We want to index all words in the LM vocab with something like this (~5 minutes with 96 cpu cores):

```
python -m WSIatScale.create_inverted_index --replacements_dir /mnt/disks/mnt2/datasets/processed_for_WSI/wiki/bert/replacements --outdir /mnt/disks/mnt2/datasets/processed_for_WSI/wiki/bert/v2/inverted_index/ --dataset Wikipedia-BERT
```

Then we want to precompute clusters for all words (~1:30 hours with 96 cpu cores):
```
python -m WSIatScale.cluster_reps_per_token --data_dir /mnt/disks/mnt2/datasets/processed_for_WSI/wiki/bert/v2/ --dataset Wikipedia-BERT > cluster_reps_per_token.log 2>&1 &
```
Now you can start viewing results in `at_scale_app.py`

We want to find all instances of words by cluster (~8:30 hours with 96 cpu cores):
```
python -m WSIatScale.assign_clusters_to_tokens --data_dir /mnt/disks/mnt2/datasets/processed_for_WSI/wiki/bert/v2 --dataset Wikipedia-BERT --write_index_by_word  --run_specific_method community_detection --run_specific_n_reps 5
```

Finaly, we can find for each community its closest communities: (per method, n_reps ~2 minutes. so max ~12 minutes.)
```
python -m WSIatScale.look_for_similar_communities --data_dir /mnt/disks/mnt2/datasets/processed_for_WSI/wiki/bert/v2 --dataset Wikipedia-BERT
```

Or for CORD-19:
```
python -m WSIatScale.create_inverted_index --replacements_dir /mnt/disks/mnt1/datasets/CORD-19/scibert/replacements --outdir /mnt/disks/mnt1/datasets/CORD-19/scibert/inverted_index/ --dataset CORD-19

python -m WSIatScale.cluster_reps_per_token --data_dir /mnt/disks/mnt1/datasets/CORD-19/scibert --dataset CORD-19 > cord_cluster_reps_per_token.log 2>&1 &

python -m WSIatScale.assign_clusters_to_tokens --data_dir /mnt/disks/mnt1/datasets/CORD-19/scibert --dataset CORD-19 --write_index_by_word  --run_specific_method community_detection --run_specific_n_reps 5 > cord_assign_clusters_to_tokens.log 2>&1 &

python -m WSIatScale.assign_clusters_to_tokens --data_dir /mnt/disks/mnt1/datasets/CORD-19/scibert --dataset CORD-19 --write_aligned_sense_idx --run_specific_method community_detection --run_specific_n_reps 5
```


# Datasets

Manual intrinsic evaluation can be found in `datasets` folder.

Our Outlier detection dataset can be found in its [designated project](https://github.com/mataney/OutlierDetectionDataset).

# Senseful embeddings

The senseful embeddings are available to download in this [link](https://drive.google.com/drive/folders/1377_9rC-II2SsbWQbSc9v915UB__6nGD?usp=sharing). For example, for the `senseful_w2v.word_vectors-10epochs` embeddings, please download both `senseful_w2v.word_vectors-10epochs` and `senseful_w2v.word_vectors-10epochs-100dim-SG.vectors.npy` files.

Then read the embeddings in the following way:

```
import gensim
print(gensim.__version__) # tested for gensim==4.3.1
embs = gensim.models.KeyedVectors.load("/Users/matane/Downloads/senseful_w2v.word_vectors-10epochs-100dim-SG")
print(embs["bass_0"])
# array([-0.23907337, 0.13821346, -0.17623161, 0.576322, -0.58257973, ...], dtype=float32)
```
