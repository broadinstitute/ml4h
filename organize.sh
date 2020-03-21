#!/bin/sh
python /home/${USER}/repos/ml/ingest/partners_ecg/organize_xmls.py \
    --src /data/partners_ecg/xml/1993-09 \
    --dst /data/partners_ecg/dst/ \
    --bad /data/partners_ecg/bad 