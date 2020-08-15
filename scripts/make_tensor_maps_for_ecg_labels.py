"""
Create a python file with TensorMaps for labeling ECGs, using c_task input files to define the label mappings.

c_task files are formatted as follows:

Term                      |     Label | SubLabel_1 | SubLabel_2 | ...
--------------------------+-----------+------------+------------+-----
myocardial infarcation    |        mi |            |            |
old myocardial infarction |        mi |        old |            |
acute mi                  |        mi |      acute |            |
pacemaker                 | pacemaker |            |            |
ventricular pacing        | pacemaker |            |   v_pacing |
...

Where:
- Term is the term found in the read from the ECG. These can be regex patterns.
- Label is the task being predicted. Each label gets a unique binary TensorMap
  that gives a positive label if term is present in the read, otherwise a negative label.
  For example:

    TensorMap(
      name = "mi",
      channel_map = {"no_mi": 0, "mi": 1},
    )

- SubLabels are sub tasks relative to the primary task. Each level of SubLabel gets a
  TensorMap whose channels are all the unique SubLabels of the primary Label at the current
  level. For example, the subclasses above result in:

    TensorMap(
      name = "mi_old_acute",
      channel_map = {"other": 0, "old": 1, "acute": 2},
    )

    TensorMap(
      name = "pacemaker_v_pacing",
      channel_map = {"no_v_pacing": 0, "v_pacing": 1},
    )

Notes:
- All Terms, Labels, and SubLabels are lower-cased.
"""

# Imports: standard library
import os
import argparse
from typing import Set, Dict, List, TextIO

# Imports: third party
import pandas as pd

JOIN_CHAR = "_"
TENSOR_FUNC_NAME = "make_ecg_label"
TENSOR_PATH_PREFIX = "partners_ecg_rest"
NEW_SCRIPT_NAME = "tensor_maps_ecg_labels.py"


def _clean_label(label: str) -> str:
    """Replace spaces and slashes with JOIN_CHAR,
    and remove parentheses and commas. Channels
    cannot start with numbers because of the way
    TensorMap infers metrics so prefix with JOIN_CHAR."""
    label = label.replace(" ", JOIN_CHAR)
    label = label.replace("/", JOIN_CHAR)
    label = label.replace("(", "")
    label = label.replace(")", "")
    label = label.replace(",", "")
    try:
        int(label[0])
        label = JOIN_CHAR + label
    except ValueError:
        pass
    return label


def _write_tmap(
    py_file: TextIO,
    task: str,
    channel_terms: Dict[str, Set[str]],
    keys: List[str],
    not_found_channel: str,
):
    """
    Writes a TensorMap named task to py_file that searches the data accessed
    at keys to match terms in channel_terms to the corresponding channel.
    """
    channel_map = {
        channel: idx
        for idx, channel in enumerate([not_found_channel] + list(channel_terms))
    }

    # fmt: off
    py_file.write(
        f"tmaps['{task}'] = TensorMap(\n"
        f"    '{task}',\n"
        f"    interpretation=Interpretation.CATEGORICAL,\n"
        f"    time_series_limit=0,\n"
        f"    path_prefix='{TENSOR_PATH_PREFIX}',\n"
        f"    channel_map={channel_map},\n"
        f"    tensor_from_file={TENSOR_FUNC_NAME}(\n"
        f"        keys={keys},\n"
        f"        channel_terms={channel_terms},\n"
        f"        not_found_channel='{not_found_channel}',\n"
        f"    ),\n"
        f")\n\n\n".replace("'", "\""),
    )
    # fmt: on


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_maps_dir",
        default=os.path.expanduser("~/dropbox/ecg/labeling"),
        help="Path to directory with c_task.csv or c_task.xlsx label maps. Files should have a header row.",
    )
    parser.add_argument(
        "--hd5_keys",
        default=["read_md_clean", "read_pc_clean"],
        nargs="+",
        help="Keys to reads in hd5s from which to extract labels.",
    )
    args = parser.parse_args()

    # Determine full path to new script; this approach generalizes regardless of where
    # users clone the ml repo on their machine
    this_script_name = os.path.split(__file__)[1]
    path_to_repo = os.path.abspath(__file__).replace(f"/scripts/{this_script_name}", "")
    path_to_new_script = os.path.join(path_to_repo, "ml4cvd", NEW_SCRIPT_NAME)

    with open(path_to_new_script, "w") as py_file:
        py_file.write(f"from typing import Dict\n")
        py_file.write(f"from ml4cvd.TensorMap import TensorMap, Interpretation\n")
        py_file.write(f"from ml4cvd.tensor_maps_ecg import {TENSOR_FUNC_NAME}\n\n\n")
        py_file.write("tmaps: Dict[str, TensorMap] = {}\n")

        for file in os.listdir(args.label_maps_dir):
            if not file.startswith("c_"):
                continue

            path = os.path.join(args.label_maps_dir, file)
            ext = os.path.splitext(file)[-1]
            if ext == ".csv":
                df = pd.read_csv(path).fillna("")
            elif ext == ".xlsx":
                df = pd.read_excel(path).fillna("")
            else:
                print(f"Creating labels from {ext} files not supported: {path}")
                continue

            if len(df.columns) < 2 or (df[df.columns[1]] == "").any():
                print(f"Label mapping table has empty labels: {path}")
                continue

            df = df.apply(lambda x: x.str.lower())
            term_col = df.columns[0]
            label_col = df.columns[1]
            for label, group in df.groupby(label_col):
                label = _clean_label(label)
                terms = set(group[term_col])
                channel_terms = {label: terms}
                _write_tmap(
                    py_file=py_file,
                    task=label,
                    channel_terms=channel_terms,
                    keys=args.hd5_keys,
                    not_found_channel=f"no_{label}",
                )

                # TODO subclasses
                # for sublevel in df.columns[2:]:
                #     for sublabel, subgroup in df.groupby(sublevel):
                #         pass

            print(f"Created tmaps from label map: {file}")

    print(f"ECG label tmaps saved to {path_to_new_script}")
