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


def copy_groups(group_name: str, source_folder: str, destination_folder: str):
    message = f"Attempting to copy hd5 files from '{source_folder}' to '{destination_folder}'... "
    logging.debug(message)

    if not os.path.exists(source_folder):
        raise ValueError('Source directory does not exist: ', source_folder)

    # If dest_dir doesn't exist, create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate over files in src_dir
    for source_file in os.listdir(source_folder):
        if not source_file.endswith(TENSOR_EXT):
            continue
        # Name the destination file with the same as the source's
        with h5py.File(os.path.join(destination_folder, source_file), 'a') as destination_file:
            _copy_group(group_name, os.path.join(source_folder, source_file), destination_file)

    msg_succeeded = f"Successfully copied the group '{group_name}' from hd5 files in '{source_folder}' to '{destination_folder}'... "
    logging.debug(msg_succeeded)


def _copy_group(group_name: str, source_file_path: str, destination_file: h5py.File):
    with h5py.File(source_file_path, 'r') as source_file:
        # Get the name of the parent for the group we want to copy
        group_path = source_file[f"{HD5_GROUP_CHAR}{group_name}"].parent.name

        # If the group doesn't exist in the destination, create it (along with parents, if any)
        group_id = destination_file.require_group(group_path)
        source_file.copy(f"{HD5_GROUP_CHAR}{group_name}", group_id, name=f"{HD5_GROUP_CHAR}{group_name}")
        logging.debug(f"Copied hd5 group '{group_name}' from source '{source_file_path}' to '{destination_file.filename}'...")


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
    for group, source in zip(args.groups, args.sources):
        copy_groups(group, source, args.destination)
