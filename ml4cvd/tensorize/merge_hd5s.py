import os
import h5py
import logging
import argparse

from ml4cvd.defines import TENSOR_EXT, HD5_GROUP_CHAR

""" 
This script copies the hd5 groups specified as 'groups' from all hd5 files within the 'sources'
directories to the same-named files within the 'destination' directory.

If the tensor files are not in a local filesystem, they can be downloaded via gsutil:
gsutil -m cp -r <gcs bucket with tensors> <local directory>

Each source directory in 'sources' must contain the group in 'groups', respectively.

If the destination directory and/or file(s) don't exist, it creates them.

If any of the destination files contains the specified group already, it errors out.

Example command line:
python .merge_hd5s.py \
    --groups continuous categorical \
    --sources /path/to/src/continuous/tensor/directory /path/to/src/categorical/tensor/directory \
    --dest /path/to/output/directory \
    --logging_level DEBUG
"""


def _copy_hd5_datasets(source_file, destination_file, group_path=HD5_GROUP_CHAR):
    for k in source_file[group_path]:
        if isinstance(source_file[k], h5py.Dataset):
            destination_file.create_dataset(group_path + k, data=source_file[k])
        else:
            _copy_hd5_datasets(source_file, destination_file, group_path=group_path + k + HD5_GROUP_CHAR)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', nargs='+', help='List of source directories with hd5 files')
    parser.add_argument('--destination', help='Destination directory to copy hd5 groups to')
    parser.add_argument('--groups', nargs='+')
    parser.add_argument("--logging_level", default='INFO', help="Logging level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.getLogger().setLevel(args.logging_level)

    if not os.path.exists(os.path.dirname(args.destination)):
        os.makedirs(os.path.dirname(args.destination))

    for source_folder in args.sources:
        for source_file in os.listdir(source_folder):
            if not source_file.endswith(TENSOR_EXT):
                continue
            with h5py.File(os.path.join(args.destination, source_file), 'a') as destination_hd5:
                with h5py.File(source_file, 'r') as source_hd5:
                    _copy_hd5_datasets(source_hd5, destination_hd5)
