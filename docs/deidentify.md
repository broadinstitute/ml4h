# Deidentify data

Some compute resources may not be allowed to store Protected Health Information (PHI). Therefore we sometimes need to deidentify data before using those resources.

The script at [scripts/deidentify.py](../scripts/deidentify.py) currently supports deidentification of ECG HD5s and STS CSV files (including both feature & outcome spreadsheets, and bootstrap lists of MRNs). Deidenfication of additional data sources can be implemented using the modular approach documented in the script itself.

To deidentify ECG and STS data:
```bash
./scripts/run.sh -c -t \
    $PWD/scripts/deidentify.py \
    --starting_id 1 \
    --ecg_dir $HOME/data/ecg/mgh \
    --sts_dir $HOME/data/sts-data \
    --mrn_map $HOME/data/deid/mgh_mrn_deid_map.csv \
    --new_ecg_dir $HOME/data/deid/ecg/mgh \
    --new_sts_dir $HOME/data/deid/sts-data
```
