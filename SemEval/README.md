# SemEval
```
python write_mask_preds.py --data_dir ../../SemEval/resources/SemEval-2010/test_data/ --out_dir ./out/SemEval2010/remove --dataset SemEval2010 --model bert-large-uncased --max_tokens_per_batch 256 --no_input_file #--fp16
```

Create Inverted Index 
```
python -m SemEval.create_inverted_SemEval2010_index.py --data_dir ... --outdir .../inverted_index --model bert-large-uncased/RoBERTa --data_file .../semeval-2013-task-13-test-data.senseval2.xml
```


Also, need to create lemmatized vocab using 
```
python WSIatScale/create_lemmatized_vocab.py --outdir /home/matane/matan/dev/WSIatScale/write_mask_preds/out/SemEval2010/bert-large-uncased
```

Run Evaluation Using
```
python -m SemEval.evaluate --data_dir201x ... --n_reps ...
```