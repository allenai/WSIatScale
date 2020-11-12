# SemEval

## SemEval 2010
```
python write_mask_preds.py --data_dir ../../SemEval/resources/SemEval-2010/test_data/ --out_dir ./out/SemEval2010/remove --dataset SemEval2010 --model bert-large-uncased --max_tokens_per_batch 256 --no_input_file #--fp16
```
Create Inverted Index
```
python -m SemEval.create_inverted_SemEval2010_index --data_dir /mnt/disks/mnt2/datasets/SemEval/SemEval2010/bert-large-uncased-redoing-semeval/ --outdir /mnt/disks/mnt2/datasets/SemEval/SemEval2010/bert-large-uncased-redoing-semeval/inverted_index --model bert-large-uncased
```

Also, need to create lemmatized vocab using
```
python WSIatScale/create_lemmatized_vocab.py --outdir /home/matane/matan/dev/WSIatScale/write_mask_preds/out/SemEval2010/bert-large-uncased --model bert-large-uncased
```

Run Evaluation Using
```
python -m SemEval.evaluate --data_dir201x ... --n_reps ...
```

## SemEval 2013
```
python write_mask_preds.py --data_dir ../../SemEval/resources/SemEval-2013-Task-13-test-data/contexts/senseval2-format/semeval-2013-task-13-test-data.senseval2.xml --out_dir /mnt/disks/mnt2/datasets/SemEval/SemEval2013/bert/ --dataset SemEval2013 --model bert-large-uncased --max_tokens_per_batch 512
```
Create Inverted Index
```
python -m SemEval.create_inverted_SemEval2013_index --data_dir /mnt/disks/mnt2/datasets/SemEval/SemEval2013/bert/ --outdir /mnt/disks/mnt2/datasets/SemEval/SemEval2013/bert/inverted_index --model bert-large-uncased --data_file /home/matane/matan/dev/SemEval/resources/SemEval-2013-Task-13-test-data/contexts/senseval2-format/semeval-2013-task-13-test-data.senseval2.xml
```

Also, need to create lemmatized vocab using
```
python WSIatScale/create_lemmatized_vocab.py --outdir /mnt/disks/mnt2/datasets/SemEval/SemEval2013/bert --model bert-large-uncased
```

## Quick Access For debugging
```
from transformers import AutoTokenizer; model_hf_path = 'bert-large-uncased'; tok = AutoTokenizer.from_pretrained(model_hf_path, use_fast=True)
```