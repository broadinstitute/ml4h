import os
import h5py
import logging
import argparse
import traceback

from ml4cvd.defines import TENSOR_EXT, HD5_GROUP_CHAR

""" 
This script copies all the hd5 datasets from all hd5 files within the 'sources'
directories to the same-named files within the 'destination' directory.

If the tensor files are not in a local filesystem, they can be downloaded via gsutil:
gsutil -m cp -r <gcs bucket with tensors> <local directory>

If the destination directory and/or file(s) don't exist, it creates them.
If any of the source files contain the same dataset at the same group path, it errors out.

Example command line:
python .merge_hd5s.py \
    --sources /path/to/src/continuous/tensor/directory/ /path/to/src/categorical/tensor/directory/ \
    --destination /path/to/output/directory/ 
"""


def merge_hd5s_into_destination(destination, sources, min_sample_id, max_sample_id, intersection):
    if not os.path.exists(os.path.dirname(destination)):
        os.makedirs(os.path.dirname(destination))

    sample_sets = [set(os.listdir(source_folder)) for source_folder in sources]
    sample_set = sample_sets[0].intersection(sample_sets[1:])
    for source_folder in sources:
        for source_file in os.listdir(source_folder):
            if not source_file.endswith(TENSOR_EXT) or (intersection and source_file not in sample_set):
                continue
            if not min_sample_id <= int(os.path.splitext(source_file)[0]) < max_sample_id:
                continue

            with h5py.File(os.path.join(destination, source_file), 'a') as destination_hd5:
                with h5py.File(os.path.join(source_folder, source_file), 'r') as source_hd5:
                    _copy_hd5_datasets(source_hd5, destination_hd5)


def _copy_hd5_datasets(source_hd5, destination_hd5, group_path=HD5_GROUP_CHAR):
    for k in source_hd5[group_path]:
        try:
            if isinstance(source_hd5[group_path][k], h5py.Dataset):
                destination_hd5.create_dataset(group_path + k, data=source_hd5[group_path][k])
            else:
                logging.debug(f"copying group {group_path + k}")
                _copy_hd5_datasets(source_hd5, destination_hd5, group_path=group_path + k + HD5_GROUP_CHAR)
        except:
            print(f"Error while attempting to merge tensor:\n{traceback.format_exc()}\n")
            print('in group', source_hd5[group_path], 'gp:', group_path, '  k', k, "/nsomething erroring at:", source_hd5.keys(), destination_hd5.keys())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', nargs='+', help='List of source directories with hd5 files')
    parser.add_argument('--destination', help='Destination directory to copy hd5 datasets to')
    parser.add_argument('--min_sample_id', default=0, type=int, help='Minimum sample id to write to tensor.')
    parser.add_argument('--max_sample_id', default=7000000, type=int, help='Maximum sample id to write to tensor.')
    parser.add_argument('--intersection', default=False, action='store_true', help='Only merge files if the sample id is in every source directory')
    parser.add_argument("--logging_level", default='INFO', help="Logging level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.getLogger().setLevel(args.logging_level)
    merge_hd5s_into_destination(args.destination, args.sources, args.min_sample_id, args.max_sample_id, args.intersection)
