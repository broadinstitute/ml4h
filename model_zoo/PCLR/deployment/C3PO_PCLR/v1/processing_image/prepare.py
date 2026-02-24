import argparse

import h5py
import numpy as np
import pandas as pd
import smart_open

leads = [
    'I', 'II', 'III', 'aVR', 'aVL', 'aVF',
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
]

ECG_LENGTH = 2500
ECG_SHAPE = (ECG_LENGTH, 12)
ECG_HD5_PATH = 'ukb_ecg_rest'

def ecg_as_tensor(ecg_file):
    with smart_open.open(ecg_file, 'rb') as f:
        with h5py.File(f, 'r') as hd5:
            ecg = np.zeros(ECG_SHAPE, dtype=np.float32)
            for k,l in enumerate(leads):
                lead = np.array(hd5[f'{ECG_HD5_PATH}/strip_{l}/instance_0'])

                interpolated_lead = np.interp(
                    np.linspace(0, 1, ECG_LENGTH),
                    np.linspace(0, 1, lead.shape[0]),
                    lead,
                )
                ecg[:, k] = interpolated_lead / 1000

    return ecg

def prepare(input_csv, output_h5):
    """Processes ECG files into HDF5 tensor format from GCS/Azure/Local."""
    df = pd.read_csv(input_csv, dtype={"file": str})
    h5_file = h5py.File(output_h5, "w")
    tensors_group = h5_file.create_group("tensors")
    df = df.dropna(subset=["file"])
    df["file"] = df["file"].astype(str)
    for _, row in df.iterrows():
        sample_id, file_path = row["file_id"], row["file"]
        print(f"Processing: sample_id={sample_id}, file_path={file_path}, type={type(file_path)}")
        tensor = ecg_as_tensor(file_path)
        tensors_group.create_dataset(str(sample_id), data=tensor)

    h5_file.close()
    print(f"Processed ECG tensors saved to {output_h5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output HDF5 file")
    args = parser.parse_args()

    prepare(args.input, args.output)
