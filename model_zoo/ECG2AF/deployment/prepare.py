import argparse

import h5py
import numpy as np
import pandas as pd
import smart_open

leads = [
    'I', 'II', 'III', 'aVR', 'aVL', 'aVF',
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
]

ECG_LENGTH = 5000
ECG_SHAPE = (ECG_LENGTH, 12)
ECG_HD5_PATH = 'partners_ecg_rest'
CHUNK_SIZE = 100

def ecg_as_tensor(ecg_file):
    """Builds full ECG tensor (timestamps, 4096, 12) across all timestamps and leads."""
    with smart_open.open(ecg_file, 'rb') as f:
        with h5py.File(f, 'r') as hd5:
            timestamp_keys = list(hd5[ECG_HD5_PATH].keys())
            linspace_standard = np.linspace(0, 1, ECG_LENGTH)
            tensor_list = []

            for start in range(0, len(timestamp_keys), CHUNK_SIZE):

                batch_keys = timestamp_keys[start:start + CHUNK_SIZE]

                for timestamp in batch_keys:
                    paths = [f"{ECG_HD5_PATH}/{timestamp}/{lead}" for lead in leads]

                    if any(path not in hd5 for path in paths):
                        print(f"Missing one or more leads at timestamp '{timestamp}' — skipping.")
                        continue
                    ecg_matrix = np.empty((ECG_LENGTH, len(leads)), dtype=np.float32)

                    for k, path in enumerate(paths):

                        ds = hd5[path]
                        raw_bytes = ds[()].tobytes()
                        
                        signal = np.frombuffer(raw_bytes, dtype=np.uint8)
                        
                        interpolated = np.interp(
                            linspace_standard,
                            np.linspace(0, 1, signal.shape[0]),
                            signal
                        )
                        ecg_matrix[:, k] = interpolated / 1000.0 

                    else:
                        tensor_list.append(ecg_matrix)

            if not tensor_list:
                raise ValueError("No valid ECG data found across timestamps.")
            return np.stack(tensor_list)
        
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
        if tensor.ndim == 2:
            # Handle the single-segment case
            tensors_group.create_dataset(str(sample_id), data=tensor)
        else:
            for i, segment in enumerate(tensor):  # each segment is (5000, 12)
                tensors_group.create_dataset(f"{str(sample_id)}_{i}", data=segment)
        #tensors_group.create_dataset(str(sample_id), data=tensor)

    h5_file.close()
    print(f"Processed ECG tensors saved to {output_h5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output HDF5 file")
    args = parser.parse_args()

    prepare(args.input, args.output)
