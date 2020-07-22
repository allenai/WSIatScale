# QuickBERTInference

## For running the forward pass and saving alternatives to file run the following command with the additional optional arguments:
```
python benchmarking_forward_pass.py --data_dir data/CORD-19 --adaptive_sampler --batch_size 1 --max_seq_length 512 --max_tokens_per_batch 16384 
```

Notice here `batch_size=1` as the adaptive sampler builds it dynamically every iteration. `max_tokens_per_batch=16384` is the max power of 2 that fits in a P-100 GPU.

For basic sampler remove `adaptive_sampler` and `max_tokens_per_batch` arguments and choose a `batch_size` that fits your GPU.

### Optional Arguments

`--iterativly_go_over_matrices`: An alternative method for going over predicted tokens and writing to disk. This is less optimized for PyTorch. Deprecated.

`--cpu`: If we want to run on CPU.