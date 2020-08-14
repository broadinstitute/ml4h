# Imports: standard library
import os
import csv
import argparse
from typing import Dict, List
from collections import defaultdict

# Imports: third party
import pandas as pd

JOIN_CHAR = "_"
TENSOR_FUNC_NAME = "make_ecg_label"
TENSOR_PATH_PREFIX = "partners_ecg_rest"
NEW_SCRIPT_NAME = "tensor_maps_ecg_labels.py"


def _clean_label_string(string):
    """Replace spaces and slashes with JOIN_CHAR,
    and remove parentheses and commas. Channels
    cannot start with numbers because of the way
    TensorMap infers metrics so prefix with JOIN_CHAR."""
    string = string.replace(" ", JOIN_CHAR)
    string = string.replace("/", JOIN_CHAR)
    string = string.replace("(", "")
    string = string.replace(")", "")
    string = string.replace(",", "")
    try:
        int(string[0])
        string = JOIN_CHAR + string
    except ValueError:
        pass
    return string


def _write_tmap_to_py(
    py_file, label_maps: str, channel_maps: Dict[str, str], hd5_keys: List[str],
):
    """Given label_maps (which associates labels with source phrases)
    and channel_maps (which associates labels with unique sublabels),
    define the tensormaps to associate source phrases with precise labels,
    and write these maps in a python script
    """

    for label in label_maps:
        cm = "{"

        for i, channel in enumerate(channel_maps[label]):
            cm += f"'{channel}': {i}, "

        # At this point, i = len(channel_maps[label])-1
        # If 'unspecified' is not a label, we need to add and index it
        if "unspecified" not in channel_maps[label]:
            cm += f"'unspecified': {i+1}"

        cm += "}"

        key_list_string = "['" + "', '".join(hd5_keys) + "']"
        py_file.write(
            f"tmaps['{label}'] = TensorMap('{label}',"
            " interpretation=Interpretation.CATEGORICAL, time_series_limit=0,"
            f" path_prefix='{TENSOR_PATH_PREFIX}', channel_map={cm},"
            f" tensor_from_file={TENSOR_FUNC_NAME}(keys={key_list_string}, dict_of_list ="
            f" {dict(label_maps[label])})) \n\n",
        )
        for key in hd5_keys:
            short_key = "md" if "_md" in key else "pc" if "_pc" in key else key
            py_file.write(
                f"tmaps['{label}_{short_key}'] = TensorMap('{label}_{short_key}',"
                " interpretation=Interpretation.CATEGORICAL, time_series_limit=0,"
                f" path_prefix='{TENSOR_PATH_PREFIX}', channel_map={cm},"
                f" tensor_from_file={TENSOR_FUNC_NAME}(keys='{key}', dict_of_list ="
                f" {dict(label_maps[label])})) \n\n",
            )


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
            if not file.startswith("c_") or not (
                file.endswith(".csv") or file.endswith(".xlsx")
            ):
                continue

            task = file.replace("c_", "").replace(".csv", "").replace(".xlsx", "")
            path = os.path.join(args.label_maps_dir, file)

            ext = os.path.splitext(file)[-1]
            if ext == ".csv":
                df = pd.read_csv(path).fillna("")
            elif ext == ".xlsx":
                df = pd.read_excel(path).fillna("")
            else:
                raise NotImplementedError(
                    f"Creating labels from {ext} files not supported.",
                )

            # Associate labels with source phrases in dict of dicts:
            #   keys   - task name and all oot-level labels in hierarchy
            #   values - dicts:
            #       keys   - labels (next level down in hierarchy)
            #       values - list of source phrases that map to a given label
            # Note: because the first key is the task name, keys of dicts in
            # label_map[task] are the remaining keys in label_map itself
            label_maps = defaultdict(lambda: defaultdict(list))

            # Associate labels with unique set of sublabels in dict of sets
            # keys   - every label in hierarchy with children
            # values - set of all child labels within a given label
            channel_maps = defaultdict(set)

            # Iterate through every source phrase in list of lists (label map)
            for idx in range(len(df)):
                row = df.iloc[idx]
                prefix = []

                # First element in row is source phrase, all other elements are label strings
                for label_str in row[1:]:
                    if label_str == "":
                        continue
                    label_str = _clean_label_string(label_str)

                    # Isolate source phrase
                    source_phrase = row[0].lower()

                    # Append source phrase to list of source phrases for this task and label string
                    if len(prefix) == 0:
                        channel_maps[task].add(label_str)
                        label_maps[task][label_str].append(source_phrase)
                    else:
                        prefix_merged = JOIN_CHAR.join(prefix)
                        channel_maps[prefix_merged].add(label_str)
                        label_maps[prefix_merged][label_str].append(source_phrase)

                    prefix.append(label_str)

            # Use assembled label and channel maps and write tmap to py file
            _write_tmap_to_py(
                py_file=py_file,
                label_maps=label_maps,
                channel_maps=channel_maps,
                hd5_keys=args.hd5_keys,
            )

            print(f"Created tmaps from label map: {file}")

    print(f"ECG label tmaps saved to {path_to_new_script}")
