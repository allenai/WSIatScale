Rerun `write_mask_preds.py` on WiC-TSV data.

Then run unsupervised prediction using

```
python -m word_sense_linking.unsupervised_wic_tsv --background_data_dir /mnt/disks/mnt2/datasets/processed_for_WSI/wiki/bert/v2/ --processed_wic_tsv /mnt/disks/mnt2/datasets/WiC_TSV_Data/__SPLIT__/bert --split Development
```