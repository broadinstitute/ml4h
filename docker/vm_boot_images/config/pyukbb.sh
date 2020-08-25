#!/bin/bash

# Fetch the latest available version
wget https://storage.googleapis.com/ml4h/ml4h-master.zip
unzip ml4h-master.zip
cd ml4h-master/pyukbb

tools/run-after-git-clone
pip install -e .[dev]
