import os
import xmltodict
import pandas as pd
from datetime import datetime
from dateutil import parser as date_parser
from multiprocessing import Pool, cpu_count
from functools import partial

import os
import sys
import base64
import struct
from collections import defaultdict

import h5py
import xmltodict
import numpy as np
import pandas as pd

def decode_ekg_muse_to_array(raw_wave, downsample=1):
    """
    Ingest the base64 encoded waveforms and transform to numeric

    downsample: 0.5 takes every other value in the array. Muse samples at 500/s and the sample model requires 250/s. So take every other.
    """
    try:
        dwnsmpl = int(1 // downsample)
    except ZeroDivisionError:
        print("You must downsample by more than 0")
    # covert the waveform from base64 to byte array
    arr = base64.b64decode(bytes(raw_wave, 'utf-8'))
    # unpack every 2 bytes, little endian (16 bit encoding)
    unpack_symbols = ''.join([char * int(len(arr) / 2) for char in 'h'])
    byte_array = struct.unpack(unpack_symbols, arr)
    return np.array(byte_array)[::dwnsmpl]

import torch
import os
import xmltodict
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

def resolve_ecg_path(basepath, finngenid, measid):
    basename = f"{finngenid}_{measid}"
    patient_dir = os.path.join(basepath, finngenid)

    for ext in [".xml", ".XML", ".ecg", ".ECG"]:
        candidate = os.path.join(patient_dir, basename + ext)
        if os.path.exists(candidate):
            return candidate

    return None


class LongitudinalECGFromMetadata(Dataset):
    def __init__(self, metadata_csv, data_path, decode_fn=decode_ekg_muse_to_array,
                 transform=None, max_timestamps=50):
        """
        Args:
            metadata_csv (str): Path to prebuilt metadata CSV.
            decode_fn (callable): Function to decode WaveFormData → np.array.
            transform (callable): Optional transform on ECG arrays.
            max_timestamps (int): Max # of ECGs per patient.
        """
        self.decode_fn = decode_fn
        self.transform = transform
        self.max_timestamps = max_timestamps

        '''
        # Load metadata
        self.df = pd.read_csv(metadata_csv, parse_dates=["timestamp"])
        # Sort globally for consistency
        print("Dataframe head is ,", self.df.head(5))
        self.df.sort_values(["patient_id", "timestamp"], inplace=True)
        self.groups = self.df.groupby("patient_id")
        self.patient_ids = list(self.groups.groups.keys())
        print(f"Loaded metadata for {len(self.patient_ids)} patients.")
        print(f"groups are ,", self.groups)
        '''

        self.df = pd.read_csv(metadata_csv, sep='\t')
        # columns are FINNGENID,MEASID,EVENT_AGE,APPROX_EVENT_DAY,TIME
        self.df["timestamp"] = self.df["APPROX_EVENT_DAY"] + "T" + self.df["TIME"]
        self.basepath = data_path
        paths = []
        for _, row in self.df.iterrows():
            path = resolve_ecg_path(
                self.basepath,
                row["FINNGENID"],
                row["MEASID"],
            )
            paths.append(path)
        
        self.df["path"] = paths
        missing = self.df["path"].isna().sum()
        if missing > 0:
            print(f"⚠️ Dropping {missing} rows with missing ECG files")

        self.df = self.df.dropna(subset=["path"])


        self.groups = self.df.groupby("FINNGENID")
        self.patient_ids = list(self.groups.groups.keys())

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        group = self.groups.get_group(pid).sort_values("timestamp")

        # Limit to most recent `max_timestamps`
        group = group.tail(self.max_timestamps)
        print("Processing patient ", pid, " with ", len(group), " timestamps.")
        arrays, timestamps, event_ages = [], [], []

        for _, row in group.iterrows():
            path = row["path"]
            try:
                with open(path, "rb") as fd:
                    dic = xmltodict.parse(fd.read().decode("utf-8"))
                
                #print("Parsed XML for path ", path, dic.keys(), dic["RestingECG"].keys())

                # Extract and decode leads
                lead_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
                lead_data = dict.fromkeys(lead_order)
                #print("Lead data initialized ", lead_data)

                for wave in dic["RestingECG"]["Waveform"][1]["LeadData"]:
                    lid = wave["LeadID"]
                    if lid in lead_order:
                        #print(f"We got waveform data {len(wave["WaveFormData"])} and type is {type(wave["WaveFormData"])} for lead {lid}")
                        lead_data[lid] = self.decode_fn(wave["WaveFormData"])  # downsample to 250 Hz
                        #print(f"Decoded lead {lid} with shape {lead_data[lid].shape}")
                

                # Compute derived leads
                lead_data['III'] = lead_data['II'] - lead_data['I']
                lead_data['aVR'] = -(lead_data['I'] + lead_data['II']) / 2
                lead_data['aVF'] = (lead_data['II'] + lead_data['III']) / 2
                lead_data['aVL'] = (lead_data['I'] - lead_data['III']) / 2

                ecg = np.stack([lead_data[l] for l in lead_order], axis=1)  # [5000,12]
                #print("ECG is ",ecg.shape)
                ecg -= ecg.mean(axis=0, keepdims=True)
                ecg /= (ecg.std(axis=0, keepdims=True) + 1e-6)

                if self.transform:
                    ecg = self.transform(ecg)
                #print("ECG after transform is ", ecg.shape)
                arrays.append(ecg)
                timestamps.append(row["timestamp"])
                event_ages.append(row["EVENT_AGE"])
                if len(arrays) == 0:
                    return {
                        "patient_id": pid,
                        "timestamps": [],
                        "event_ages": [],
                        "ecgs": [],
                    }


            except Exception as e:
                print(f"⚠️ Failed to parse {path}: {e}")
                continue

        return {
            "patient_id": pid,
            "timestamps": timestamps,
            "event_ages": event_ages,      
            "ecgs": arrays,   # list of np.arrays [5000,12]
        }


# -----------------------------
# Collate Function (Variable Length)
# -----------------------------
def collate_longitudinal(batch):
    """
    Collate function for variable # of ECGs per patient.
    Returns lists, not stacked tensors.
    """
    batch_size = len(batch)
    if batch_size == 0:
        return None
    # Find max number of ECGs in this batch
    max_seq_len = max(len(item["ecgs"]) for item in batch)
    seq_len = max_seq_len

    # Initialize padded tensors
    ecgs_padded = []
    masks = []
    patient_ids = []
    event_ages = []


    for item in batch:
        #print(item)
        ecgs = item["ecgs"]
        num_ecgs = len(ecgs)
        # pad missing ECGs with zeros
        if num_ecgs == 0:
            print(f"⚠️ Skipping patient {item['patient_id']} (no valid ECGs)")
            continue
        pad_count = seq_len - num_ecgs
        if pad_count > 0:
            pad_ecgs = [np.zeros_like(ecgs[0]) for _ in range(pad_count)]
            ecgs = ecgs + pad_ecgs

        ecgs_tensor = torch.tensor(np.stack(ecgs)).float()   # [T, 5000, 12]
        mask_tensor = torch.zeros(seq_len, dtype=torch.bool)
        mask_tensor[:num_ecgs] = True

        ecgs_padded.append(ecgs_tensor)
        masks.append(mask_tensor)
        patient_ids.append(item["patient_id"])
        event_ages.append(item["event_ages"])

    if len(ecgs_padded) == 0:
        return None

    # Stack into tensors
    ecgs_padded = torch.stack(ecgs_padded, dim=0)   # [B, T, 5000, 12]
    masks = torch.stack(masks, dim=0)               # [B, T]

    return ecgs_padded, masks, patient_ids, event_ages

#Inference

#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from tensorflow.keras.models import load_model

from Tranformer import ECGKerasEncoderWrapper

# --- Normalization constants ---
MEAN_AGE = 54.4974
STD_AGE = 20.7367

def denorm(x):
    return x * STD_AGE + MEAN_AGE


# --------- Single ECG (Keras) inference ---------
def predict_single(keras_model, ecgs, masks):
    latest_ecgs = []
    for i in range(ecgs.shape[0]):
        seq_len = masks[i].sum().item()
        latest_ecgs.append(ecgs[i, seq_len - 1].numpy())
    latest_ecgs = np.stack(latest_ecgs, axis=0)
    preds = keras_model.predict(latest_ecgs, verbose=0)
    preds = preds["output_age_continuous"].flatten()
    return preds


# --------- Longitudinal (Torch) inference ---------
def predict_longitudinal(model, ecgs, masks, device):
    ecgs, masks = ecgs.to(device), masks.to(device)
    with torch.no_grad():
        out = model(ecgs, mask=masks)
    return out.cpu().numpy().flatten()




import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="The path to the data file.")
    parser.add_argument("--metadata_path", type=str, required=True, help="The path to the data file.")
    args = parser.parse_args()
    '''First create a metadata file, it may take some time but it will be useful later for
       analysis as well
    '''
    print("Path is ,", args.path)
    #metadata_csv_path = build_metadata_parallel(args.path)
    metadata_csv_path = args.metadata_path
    print("Metadata csv path is ,", metadata_csv_path)


    '''
    Creates longitudinal dataset from metadata
    '''
    dataset = LongitudinalECGFromMetadata(metadata_csv_path, data_path=args.path)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn = collate_longitudinal)

    # --- Load models ---
    keras_model = load_model(
        "ecg2age_v2025_09_11.keras", compile=False
    )
    print("Keras single-ECG model loaded.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    long_model = ECGKerasEncoderWrapper(
        transformer_dim=64, num_layers=4, num_heads=4, dropout=0.2,
        keras_model_path="ecg2age_v2025_09_11.keras"
    ).to(device)

    checkpoint = torch.load(
        "longitudinal_ecg2age_performerage.pt", map_location=device
    )
    long_model.load_state_dict(checkpoint["model_state_dict"])
    long_model.eval()
    print("Longitudinal Transformer model loaded.")

    # --- Prepare CSV ---
    out_csv = "/output/finngen_inference_comparison_single_vs_longitudinal.csv"
    header_written = False
    buffer = []

    # --- Run inference both models ---

    for batch_idx, batch in enumerate(tqdm(loader, desc = "Inference for both models")):
        if batch is None:
            continue
        ecgs, masks, patient_ids, event_ages = batch
        pid = patient_ids[0]


        # --- Single ECG (Keras model) ---
        seq_len = masks[0].sum().item()
        latest_event_age = event_ages[0][seq_len - 1]

        latest_ecg = ecgs[0, seq_len - 1].cpu().numpy()[np.newaxis, ...]  # [1, 5000, 12]
        y_pred_single_norm = keras_model.predict(latest_ecg, verbose=0)["output_age_continuous"].flatten()[0]
        y_pred_single = denorm(y_pred_single_norm)

        # --- Longitudinal ---
        y_pred_long_norm = predict_longitudinal(long_model, ecgs, masks, device)[0]
        y_pred_long = denorm(y_pred_long_norm)

        # --- Compose identifier ---

        # --- Append result ---
        buffer.append({
            "patient_id": pid,
            "num_ecgs": masks[0].sum().item(),
            "true_age": latest_event_age,
            "pred_single_norm": y_pred_single_norm,
            "pred_single": y_pred_single,
            "pred_long_norm": y_pred_long_norm,
            "pred_long": y_pred_long,
        })

        # --- Flush every 100 patients ---
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(dataset):
            df = pd.DataFrame(buffer)
            df.to_csv(out_csv, mode="a", header=not header_written, index=False)
            header_written = True
            buffer = []
            print(f"Flushed {batch_idx + 1} patients to {out_csv}")

    print(f"✅ Finished. All predictions saved in {out_csv}")



    
