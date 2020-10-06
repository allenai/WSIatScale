# TODO

Run `write_mask_preds.py` with `--dataset SemEval`.

Create Inverted Index 
```
python SemEval/create_inverted_index.py --replacements_dir ... --outdir .../inverted_index --model bert-large-uncased/RoBERTa --data_file .../semeval-2013-task-13-test-data.senseval2.xml
```


Also, need to create lemmatized vocab using 
```
python WSIatScale/create_lemmatized_vocab.py --outdir /home/matane/matan/dev/WSIatScale/write_mask_preds/out/SemEval2010/bert-large-uncased
```

Run Evaluation Using
```
python -m SemEval.evaluate --data_dir201x ... --n_reps ...
```