"""Write EchoNext (PhysioNet) ECGs and labels to one hd5 per ECG for use with ml4h.tensormap.echonext

Example:
    python ml4h/tensorize/tensorize_echonext.py \
        --source /path/to/physionet.org/files/echonext/1.1.0 \
        --output /path/to/echonext/tensors/
"""
import os
import json
import argparse

import h5py
import numpy as np
import pandas as pd

LABEL_COLS = [
    "lvef_lte_45_flag",
    "lvwt_gte_13_flag",
    "aortic_stenosis_moderate_or_greater_flag",
    "aortic_regurgitation_moderate_or_greater_flag",
    "mitral_regurgitation_moderate_or_greater_flag",
    "tricuspid_regurgitation_moderate_or_greater_flag",
    "pulmonary_regurgitation_moderate_or_greater_flag",
    "rv_systolic_dysfunction_moderate_or_greater_flag",
    "pericardial_effusion_moderate_large_flag",
    "pasp_gte_45_flag",
    "tr_max_gte_32_flag",
    "shd_moderate_or_greater_flag",
]

METADATA_COLS = [
    "ecg_key",
    "patient_key",
    "age_at_ecg",
    "sex",
    "acquisition_year",
    "location_setting",
    "race_ethnicity",
    "most_recent_ecg",
    "split",
]

SPLIT_TO_FILES = {
    "train": ("EchoNext_train_waveforms.npy", "EchoNext_train_tabular_features.npy"),
    "val": ("EchoNext_val_waveforms.npy", "EchoNext_val_tabular_features.npy"),
    "test": ("EchoNext_test_waveforms.npy", "EchoNext_test_tabular_features.npy"),
}


def write_hd5(out_path, waveform, tabular, row):
    with h5py.File(out_path, "w") as h:
        h.create_dataset("ecg/waveform", data=waveform[0].astype("float32"), compression="gzip", compression_opts=4)
        h.create_dataset("ecg/waveform_1x2500x12", data=waveform.astype("float32"), compression="gzip", compression_opts=4)
        h.create_dataset("ecg/tabular_features", data=tabular.astype("float32"))

        labels_group = h.create_group("labels")
        for label in LABEL_COLS:
            labels_group.create_dataset(label, data=np.array([float(row[label])], dtype="float32"))

        meta_group = h.create_group("metadata")
        for col in METADATA_COLS:
            value = row[col]
            if pd.isna(value):
                value = ""
            meta_group.attrs[col] = str(value)


def tensorize(source, output, max_per_split=None):
    os.makedirs(output, exist_ok=True)
    df = pd.read_csv(os.path.join(source, "echonext_metadata_100k.csv"))

    summary = []
    counts = {}
    for split, (wave_file, tab_file) in SPLIT_TO_FILES.items():
        split_dir = os.path.join(output, split)
        os.makedirs(split_dir, exist_ok=True)

        # metadata rows for each split are in the same order as the split's npy arrays
        meta_split = df[df["split"] == split].reset_index(drop=True)
        waves = np.load(os.path.join(source, wave_file), mmap_mode="r")
        tabs = np.load(os.path.join(source, tab_file), mmap_mode="r")

        n = min(len(meta_split), waves.shape[0], tabs.shape[0])
        if max_per_split is not None:
            n = min(n, max_per_split)
        counts[split] = n
        print(f"{split}: writing {n} files")

        for i in range(n):
            row = meta_split.iloc[i]
            out_path = os.path.join(split_dir, f"echonext_{split}_{i:06d}.hd5")
            if not os.path.exists(out_path):
                write_hd5(out_path, waves[i], tabs[i], row)

            summary.append({
                "split": split,
                "index_in_split": i,
                "path": out_path,
                "ecg_key": str(row["ecg_key"]),
                "patient_key": str(row["patient_key"]),
                "shd_moderate_or_greater_flag": int(row["shd_moderate_or_greater_flag"]),
            })

    manifest = os.path.join(output, "tensorize_manifest.csv")
    pd.DataFrame(summary).to_csv(manifest, index=False)

    with open(os.path.join(output, "tensorize_info.json"), "w") as f:
        json.dump({
            "source": source,
            "output": output,
            "counts": counts,
            "label_cols": LABEL_COLS,
        }, f, indent=2)

    print("Wrote manifest:", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="EchoNext folder with echonext_metadata_100k.csv and the *_waveforms.npy files")
    parser.add_argument("--output", required=True, help="Folder to write hd5s into, one subfolder per split")
    parser.add_argument("--max_per_split", type=int, default=None, help="Only write the first N ECGs of each split, useful for smoke tests")
    args = parser.parse_args()
    tensorize(args.source, args.output, args.max_per_split)
