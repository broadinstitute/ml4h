import os
import h5py
import logging
import argparse
import traceback
from collections import Counter
import numpy as np

from ml4h.defines import TENSOR_EXT, HD5_GROUP_CHAR
from ml4h.tensormap.ukb.ecg import _check_valid_ecg_rest_random_beats, _create_ecg_rest_random_beats

"""
This script copies all the hd5 datasets from all hd5 files within the 'sources'
directories to the same-named files within the 'destination' directory.

If the tensor files are not in a local filesystem, they can be downloaded via gsutil:
gsutil -m cp -r <gcs bucket with tensors> <local directory>

If the destination directory and/or file(s) don't exist, it creates them.
If any of the source files contain the same dataset at the same group path, it errors out.

You can optionally add --derived_data_fn_name, which specifies a function to use to derive
new data from the source hd5s and save them to the destination hd5s. In this case, only
the derived data will be created, and no other hd5 datasets will be copied from the sources.

Example command line:
python .merge_hd5s.py \
    --sources /path/to/src/continuous/tensor/directory/ /path/to/src/categorical/tensor/directory/ \
    --destination /path/to/output/directory/
"""


def merge_hd5s_into_destination(destination, sources, min_sample_id, max_sample_id, intersect, inplace, valid_fn, derived_data_fn):
    stats = Counter()
    if not os.path.exists(os.path.dirname(destination)):
        os.makedirs(os.path.dirname(destination))

    if inplace:
        sample_set = os.listdir(destination)
    elif intersect:
        sample_sets = [os.listdir(source_folder) for source_folder in sources]
        sample_set = set(sample_sets[0]).intersection(*sample_sets[1:])

    for source_folder in sources:
        for source_file in os.listdir(source_folder):
            if not source_file.endswith(TENSOR_EXT):
                continue
            if not (min_sample_id <= int(os.path.splitext(source_file)[0]) < max_sample_id):
                continue
            if (intersect or inplace) and source_file not in sample_set:
                continue

            with h5py.File(os.path.join(destination, source_file), 'a') as destination_hd5:
                with h5py.File(os.path.join(source_folder, source_file), 'r') as source_hd5:
                    _copy_hd5_datasets(source_hd5, destination_hd5, valid_fn=valid_fn, derived_data_fn=derived_data_fn, stats=stats)
        logging.info(f"Done copying source folder {source_folder}")

    for k in stats:
        logging.info(f"{k} has {stats[k]} tensors")


def _copy_hd5_datasets(source_hd5, destination_hd5, group_path=HD5_GROUP_CHAR, valid_fn=None, derived_data_fn=None, stats=None):
    for k in source_hd5[group_path]:
        data = source_hd5[group_path][k]
        name = group_path + k

        is_dataset = isinstance(data, h5py.Dataset)
        if valid_fn:
            is_valid = valid_fn(name)
        else:
            is_valid = is_dataset

        if is_valid:
            try:
                if derived_data_fn:
                    out_dict = derived_data_fn(name, source_hd5)
                    for out_name, out_data in out_dict.items():
                        logging.debug(f"creating derived {out_name}")
                        if isinstance(out_data, np.ndarray) or isinstance(out_data, list):
                            destination_hd5.create_dataset(out_name, data=out_data, compression='gzip')
                        else:
                            destination_hd5.create_dataset(out_name, data=out_data)
                        stats[out_name] += 1
                else:
                    logging.debug(f"copying {name}")
                    if data.chunks is None:
                        destination_hd5.create_dataset(name, data=data)
                    else:
                        destination_hd5.create_dataset(name, data=data, compression='gzip')
                    stats[name] += 1
            except (OSError, KeyError, RuntimeError, ValueError) as e:
                logging.debug(f"Error trying to write:{k} at group path:{group_path} error:{e}\n{traceback.format_exc()}\n")
        if not is_dataset:
            logging.debug(f"copying group {name}")
            _copy_hd5_datasets(source_hd5, destination_hd5, group_path=name + HD5_GROUP_CHAR, valid_fn=valid_fn, derived_data_fn=derived_data_fn, stats=stats)

VALID_MAP = {
    'ecg_rest_random_beats': _check_valid_ecg_rest_random_beats,
}

DEF_MAP = {
    'ecg_rest_random_beats': _create_ecg_rest_random_beats,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', nargs='+', help='List of source directories with hd5 files')
    parser.add_argument('--destination', help='Destination directory to copy hd5 datasets to')
    parser.add_argument('--min_sample_id', default=0, type=int, help='Minimum sample id to write to tensor.')
    parser.add_argument('--max_sample_id', default=7000000, type=int, help='Maximum sample id to write to tensor.')
    parser.add_argument("--logging_level", default='INFO', help="Logging level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument('--inplace', default=False, action='store_true', help='Merge into pre-existing destination tensors')
    parser.add_argument(
        '--intersect', default=False, action='store_true',
        help='Only merge files if the sample id is in every source directory (and if destination if destination is not empty)',
    )
    parser.add_argument('--derived_data_fn_name', default=None, help='Name of the function with which to derive data from the source directories, only the derived data is transferred')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logging.getLogger().setLevel(args.logging_level)
    if args.derived_data_fn_name:
        valid_fn = VALID_MAP[args.derived_data_fn_name]
        derived_data_fn = DEF_MAP[args.derived_data_fn_name]
    else:
        valid_fn = None
        derived_data_fn = None

    merge_hd5s_into_destination(
        args.destination, args.sources, args.min_sample_id, args.max_sample_id, args.intersect,
        args.inplace, valid_fn, derived_data_fn,
    )
