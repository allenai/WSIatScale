# QuickBERTInference

## For running the forward pass and saving alternatives to file run the following command with the additional optional arguments:
```
python write_lm_mask_preds.py --data_dir data/CORD-19 --max_tokens_per_batch 16384
```

`max_tokens_per_batch=16384` is the max power of 2 that fits in a single P-100 GPU.

### Optional Arguments

`--iterativly_go_over_matrices`: An alternative method for going over predicted tokens and writing to disk. This is less optimized for PyTorch. Deprecated.

`--cpu`: If we want to run on CPU.

`simple_sampler` For a basic sampler (If using this, remove `max_tokens_per_batch` and choose a `batch_size` that fits your GPU.)