# Imports: standard library
import os
import shutil
import argparse
from timeit import default_timer as timer
from multiprocessing import Pool, cpu_count

# Imports: third party
import h5py
import numpy as np
import pandas as pd

# Imports: first party
from ml4cvd.definitions import CSV_EXT, ECG_PREFIX, TENSOR_EXT, MRN_COLUMNS
from ml4cvd.tensor_generators import _sample_csv_to_set

"""
To add a new data source to deidentify:
1. add the function to get MRNs from the data to _get_mrns()
2. add the function to deidentify the data to run()
"""

phi_keys = {
    "attendingmdfirstname",
    "attendingmdhisid",
    "attendingmdlastname",
    "consultingmdfirstname",
    "consultingmdhisid",
    "consultingmdlastname",
    "hisorderingmdfirstname",
    "hisorderingmdlastname",
    "orderingmdfirstname",
    "orderingmdhisid",
    "orderingmdid",
    "orderingmdlastname",
    "placersfirstname",
    "placershisid",
    "placerslastname",
    "patientfirstname",
    "patientlastname",
    "patientid",
    "patientid_clean",
    "diagnosis_md",
    "diagnosis_pc",
    "acquisitiontechfirstname",
    "acquisitiontechid",
    "acquisitiontechlastname",
    "admittingmdfirstname",
    "admittingmdhisid",
    "admittingmdlastname",
    "fellowfirstname",
    "fellowlastname",
    "fellowid",
    "hisaccountnumber",
    "referringmdid",
}


def _get_ecg_mrns(args):
    mrns = set()
    if args.new_ecg_dir is not None:
        for root, dirs, files in os.walk(args.ecg_dir):
            for file in files:
                split = os.path.splitext(file)
                if split[-1] != TENSOR_EXT:
                    continue
                mrn = int(split[0])
                mrns.add(mrn)
    return mrns


sts_skip = set()


def _get_sts_mrns(args):
    mrns = set()
    if args.new_sts_dir is not None:
        for root, dirs, files in os.walk(args.sts_dir):
            for file in files:
                split = os.path.splitext(file)
                if split[-1] != CSV_EXT:
                    continue
                path = os.path.join(root, file)
                try:
                    _mrns = _sample_csv_to_set(sample_csv=path)
                except ValueError:
                    print(
                        f"Could not get MRNs from csv, skipping de-identification: {path}",
                    )
                    global sts_skip
                    sts_skip.add(path)
                    continue
                _mrns = {int(mrn) for mrn in _mrns}
                mrns |= _mrns
    return mrns


def _get_mrns(args):
    """
    Get a list of unique MRNs from the data sources that are being remapped.
    """
    mrns = set()

    mrns |= _get_ecg_mrns(args)
    mrns |= _get_sts_mrns(args)

    return list(mrns)


def _remap_mrns(args):
    """
    Remap and save the MRNs from the data sources that are being remapped to new random IDs.
    """
    if os.path.isfile(args.mrn_map):
        # call to _get_sts_mrns to determine which files to skip for sts deidentification
        _get_sts_mrns(args)

        mrn_map = pd.read_csv(args.mrn_map, low_memory=False, usecols=["mrn", "new_id"])
        mrn_map = mrn_map.set_index("mrn")
        mrn_map = mrn_map["new_id"].to_dict()
        print(f"Existing MRN map loaded from {args.mrn_map}")
    else:
        mrns = _get_mrns(args)

        new_ids = list(range(args.starting_id, len(mrns) + args.starting_id))
        print(f"Last ID used in remapping MRNs was {new_ids[-1]}")
        np.random.shuffle(new_ids)

        mrn_map = dict(zip(mrns, new_ids))

        df = pd.DataFrame.from_dict(mrn_map, orient="index", columns=["new_id"])
        df.index.name = "mrn"
        os.makedirs(os.path.dirname(args.mrn_map), exist_ok=True)
        df.to_csv(args.mrn_map)
        print(f"MRN map saved to {args.mrn_map}")

    return mrn_map


def _swap_path_prefix(path, prefix, new_prefix):
    """
    Given:
        path = /foo/bar
        prefix = /foo
        new_prefix = /baz
    Returns:
        /baz/bar
    """
    path_relative_root = path.replace(prefix, "").lstrip("/")
    new_path = os.path.join(new_prefix, path_relative_root)
    if not os.path.isdir(new_path):
        os.makedirs(new_path, exist_ok=True)
    return new_path


def _deidentify_ecg(old_new_path):
    """
    Given a path to an existing HD5, copy it to a new path and delete all identifiable information.
    """
    old_path, new_path = old_new_path
    if os.path.exists(new_path):
        print(f"Path to new de-identified ECG already exists: {new_path}")
        print(f"Replaced {new_path}")
        os.remove(new_path)
    shutil.copyfile(old_path, new_path)

    with h5py.File(new_path, "r+") as hd5:
        # Only delete PHI keys from HD5s that lack 'deidentified' flag
        if "deidentified" not in hd5:
            for ecg_date in hd5[ECG_PREFIX]:
                for key in hd5[ECG_PREFIX][ecg_date]:
                    if key in phi_keys:
                        del hd5[ECG_PREFIX][ecg_date][key]

            # Add bool to hd5 indicating this file is de-identified
            hd5.create_dataset("deidentified", data=True, dtype=bool)


def _deidentify_ecgs(args, mrn_map):
    """
    Create de-identified ECGs in parallel.
    """
    if args.new_ecg_dir is None:
        return

    old_new_paths = []
    for root, dirs, files in os.walk(args.ecg_dir):
        new_root = _swap_path_prefix(root, args.ecg_dir, args.new_ecg_dir)
        for file in files:
            split = os.path.splitext(file)
            if split[-1] != TENSOR_EXT:
                continue
            mrn = int(split[0])
            new_id = mrn_map[mrn]
            old_path = os.path.join(root, file)
            new_path = os.path.join(new_root, f"{new_id}{TENSOR_EXT}")
            old_new_paths.append((old_path, new_path))

    with Pool(processes=args.num_workers) as pool:
        pool.map(_deidentify_ecg, old_new_paths)

    print(f"De-identified {len(old_new_paths)} ECGs at {args.new_ecg_dir}")


def _deidentify_sts_csv(path, mrn_map):
    """
    Given a path to an STS CSV, delete all identifiable information.
    """
    df = pd.read_csv(path, header=None, low_memory=False)

    # Infer csv header
    try:
        # If first cell is an int, it's likely a sample ID and there is no header
        int(df.iloc[0].values[0])
    except ValueError:
        df.columns = df.iloc[0]
        df = df[1:]

    matches = set(df.columns) & MRN_COLUMNS
    if len(matches) == 0:
        # If none of the known MRN columns are in the csv, assume it's the first column
        mrn_col = df.columns[0]
    else:
        mrn_col = next(iter(matches))

    # Remap MRNs and drop PHI columns
    df[mrn_col] = df[mrn_col].apply(lambda mrn: mrn_map[int(mrn)])
    phi_cols = set(df.columns) & phi_keys
    df = df.drop(phi_cols, axis=1)

    df.to_csv(path, index=False)


def _deidentify_sts_csvs(args, mrn_map):
    """
    Create de-identified STS data.
    """
    if args.new_sts_dir is None:
        return

    count = 0
    for root, dirs, files in os.walk(args.sts_dir):
        new_root = _swap_path_prefix(root, args.sts_dir, args.new_sts_dir)
        for file in files:
            split = os.path.splitext(file)
            if split[-1] != CSV_EXT:
                continue
            old_path = os.path.join(root, file)
            global sts_skip
            if old_path in sts_skip:
                continue
            new_path = os.path.join(new_root, file)
            if os.path.exists(new_path):
                print(f"Path to new de-identified STS CSV already exists: {new_path}")
                print(f"Replaced {new_path}")
                os.remove(new_path)
            shutil.copyfile(old_path, new_path)
            _deidentify_sts_csv(new_path, mrn_map)
            count += 1

    print(f"De-identified {count} STS CSVs at {args.new_sts_dir}")


def run(args):
    start_time = timer()

    mrn_map = _remap_mrns(args)

    _deidentify_ecgs(args, mrn_map)
    _deidentify_sts_csvs(args, mrn_map)

    end_time = timer()
    elapsed_time = end_time - start_time
    print(f"De-identification took {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mrn_map",
        default=os.path.expanduser("~/dropbox/ecg/mgh_mrn_deid_map.csv"),
        help="Path to CSV of MRN -> deidentified ID map.",
    )
    parser.add_argument(
        "--starting_id", default=1, type=int, help="Starting value for new IDs.",
    )
    parser.add_argument(
        "--ecg_dir", default="/data/ecg/mgh", help="Path to ECG HD5s.",
    )
    parser.add_argument(
        "--sts_dir",
        default=os.path.expanduser("~/dropbox/sts_data"),
        help="Path to STS CSVs.",
    )
    parser.add_argument(
        "--new_ecg_dir",
        help="Path to save de-identified ECG HD5s to. "
        "Skip this argument to skip de-identification of ECG data.",
    )
    parser.add_argument(
        "--new_sts_dir",
        help="Path to save de-identified STS CSVs to. "
        "Skip this argument to skip de-identification of STS data.",
    )
    parser.add_argument(
        "--num_workers",
        default=cpu_count(),
        type=int,
        help="Number of worker processes to use if processing in parallel.",
    )
    args = parser.parse_args()
    run(args)
