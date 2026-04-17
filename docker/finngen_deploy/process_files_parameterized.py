#!/usr/bin/env python3
"""
Longitudinal ECG inference — both models are Keras.

Two modes controlled by --mode:

  encode  →  run the Keras encoder on every visit and save to parquet.
             One row per ECG visit:
               patient_id, timestamp, event_age, visit_index,
               latent_0, latent_1, …, latent_{D-1}

  infer   →  run single-ECG + longitudinal age prediction and save to CSV.

──────────────────────────────────────────────────────────────────────────────
Encode mode:
    python inference.py \\
        --mode            encode \\
        --path            /data/ecg_files \\
        --metadata_path   metadata.tsv \\
        --encoder_path    ecg2age_v2025_09_11.keras \\
        --encoder_layer   activation_18 \\
        --output          embeddings.parquet \\
        --batch_size      8

Infer mode:
    python inference.py \\
        --mode              infer \\
        --path              /data/ecg_files \\
        --metadata_path     metadata.tsv \\
        --encoder_path      ecg2age_v2025_09_11.keras \\
        --encoder_layer     activation_18 \\
        --longitudinal_path longitudinal_ecg2age_standalone.keras \\
        --output            predictions.csv \\
        --batch_size        8 \\
        --delta_consider
──────────────────────────────────────────────────────────────────────────────
"""

import os
import base64
import struct
import argparse

import numpy as np
import pandas as pd
import xmltodict
from tqdm import tqdm
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader

import keras
import keras.ops as ops

# ─────────────────────────────────────────────────────────────────────────────
# Required custom layer for loading the standalone longitudinal model
# ─────────────────────────────────────────────────────────────────────────────

@keras.saving.register_keras_serializable(package="ECGStandalone")
class MaskedMeanPool(keras.layers.Layer):
    """Mean-pool over sequence dim using a float mask (1=real, 0=padded)."""
    def call(self, inputs):
        z, mask = inputs
        mask_exp = ops.expand_dims(mask, -1)
        z_sum    = ops.sum(z * mask_exp, axis=1)
        m_sum    = ops.maximum(ops.sum(mask_exp, axis=1), 1.0)
        return z_sum / m_sum

    def get_config(self):
        return super().get_config()

@keras.saving.register_keras_serializable(package="ml4h")
class PositionIndexLayer(keras.layers.Layer):
    """Custom layer to generate position indices for positional encoding."""

    def __init__(self, max_len, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len

    def call(self, inputs):
        """Generate position indices."""
        b = keras.ops.shape(inputs)[0]
        pos = keras.ops.arange(0, self.max_len)
        pos = keras.ops.tile(keras.ops.expand_dims(pos, 0), (b, 1))
        return pos

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len})
        return config

@keras.saving.register_keras_serializable(package="ml4h")
class ExpandDimsLayer(keras.layers.Layer):
    """Custom layer to expand dimensions with specified axis."""

    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        """Expand dimensions."""
        return keras.ops.expand_dims(inputs, self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

@keras.saving.register_keras_serializable(package="ml4h")
class LogicalAndLayer(keras.layers.Layer):
    """Custom layer to apply logical AND to two inputs."""

    def call(self, inputs):
        """Apply logical AND."""
        return keras.ops.logical_and(inputs[0], inputs[1])


@keras.saving.register_keras_serializable(package="ml4h")
class CastToFloatLayer(keras.layers.Layer):
    """Custom layer to cast boolean to float32."""

    def call(self, inputs):
        """Cast to float32."""
        return keras.ops.cast(inputs, "float32")


@keras.saving.register_keras_serializable(package="ml4h")
class ApplyVeryNegativeLayer(keras.layers.Layer):
    """Custom layer to apply very negative values to masked positions."""

    def call(self, inputs):
        """Apply very negative values."""
        return (1.0 - inputs) * (-1e9)


@keras.saving.register_keras_serializable(package="ml4h")
class SumOverTimeLayer(keras.layers.Layer):
    """Custom layer to sum over time dimension."""

    def call(self, inputs):
        """Sum over axis 1."""
        return keras.ops.sum(inputs, axis=1)


@keras.saving.register_keras_serializable(package="ml4h")
class DynamicPositionIndicesLayer(keras.layers.Layer):
    """Custom layer to generate dynamic position indices based on input shape."""

    def call(self, inputs):
        """Generate position indices matching input sequence length."""
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]
        positions = ops.tile(
            ops.expand_dims(ops.arange(seq_len), axis=0), [batch_size, 1]
        )
        return positions

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])


@keras.saving.register_keras_serializable(package="ml4h")
class AttentionMaskLayer(keras.layers.Layer):
    """Custom layer to create attention mask from padding mask."""

    def call(self, inputs):
        """Expand mask for attention: (B, T) -> (B, 1, T)."""
        return ops.cast(inputs[:, None, :], "bool")

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, input_shape[1])


# ─────────────────────────────────────────────────────────────────────────────
# ECG decoding helpers
# ─────────────────────────────────────────────────────────────────────────────

def decode_ekg_muse_to_array(raw_wave, downsample=1):
    dwnsmpl        = int(1 // downsample)
    arr            = base64.b64decode(bytes(raw_wave, "utf-8"))
    unpack_symbols = "".join([char * int(len(arr) / 2) for char in "h"])
    byte_array     = struct.unpack(unpack_symbols, arr)
    return np.array(byte_array)[::dwnsmpl]






def resolve_ecg_path(basepath, finngenid, measid):
    basename    = f"{finngenid}_{measid}"
    patient_dir = os.path.join(basepath, finngenid)
    for ext in [".xml", ".XML", ".ecg", ".ECG"]:
        candidate = os.path.join(patient_dir, basename + ext)
        if os.path.exists(candidate):
            return candidate
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class LongitudinalECGFromMetadata(Dataset):
    def __init__(self, metadata_csv, data_path,
                 decode_fn=decode_ekg_muse_to_array,
                 transform=None, max_timestamps=50):
        self.decode_fn       = decode_fn
        self.transform       = transform
        self.max_timestamps  = max_timestamps

        df              = pd.read_csv(metadata_csv, sep="\t")
        df["timestamp"] = df["APPROX_EVENT_DAY"] + "T" + df["TIME"]
        df["path"]      = [
            resolve_ecg_path(data_path, r["FINNGENID"], r["MEASID"])
            for _, r in df.iterrows()
        ]
        missing = df["path"].isna().sum()
        if missing:
            print(f"⚠️  Dropping {missing} rows with missing ECG files")
        df = df.dropna(subset=["path"])

        max_age_per_person = df.groupby("FINNGENID")["EVENT_AGE"].transform("max")
        df["age_delta"] = (max_age_per_person - df["EVENT_AGE"]) * 365.25  # days from most recent ECG
        age_delta_mean = 2135.022283470709#df["age_delta"].mean()
        age_delta_std = 2361.7968570069415#df["age_delta"].std()
        df["age_delta_norm"] = (df["age_delta"] - age_delta_mean) / age_delta_std


        self.groups      = df.groupby("FINNGENID")
        self.patient_ids = list(self.groups.groups.keys())
        print(f"Loaded {len(self.patient_ids)} patients.")

    def __len__(self):
        return len(self.patient_ids)



    def process_ecg_xml_format(self, row):
        with open(row["path"], "rb") as fd:
            dic = xmltodict.parse(fd.read().decode("utf-8"))

        lead_order = ["I","II","III","aVR","aVL","aVF",
                  "V1","V2","V3","V4","V5","V6"]
        lead_data = dict.fromkeys(lead_order)

        for wave in dic["RestingECG"]["Waveform"][1]["LeadData"]:
            lid = wave["LeadID"]
            if lid in lead_order:
                lead_data[lid] = self.decode_fn(wave["WaveFormData"])

        lead_data["III"] = lead_data["II"] - lead_data["I"]
        lead_data["aVR"] = -(lead_data["I"] + lead_data["II"]) / 2
        lead_data["aVF"] = (lead_data["II"] + lead_data["III"]) / 2
        lead_data["aVL"] = (lead_data["I"] - lead_data["III"]) / 2

        ecg = np.stack([lead_data[l] for l in lead_order], axis=1).astype(np.float32)
        ecg = ecg[:4096, :]
        ecg -= ecg.mean(axis=0, keepdims=True)
        ecg /= ecg.std(axis=0, keepdims=True) + 1e-6

        if self.transform:
            ecg = self.transform(ecg)

        return ecg


    def process_ecg_hl7_format(self, row):
        with open(row['path'], "rt") as ecg:
            line = ecg.readline()
            field4 = line.split("|")[3]

        # --- Step 1: Find CHN line ---
            while field4 != "CHN":
                line = ecg.readline()
                field4 = line.split("|")[3]

            raw_channels = line.split("|")[5].split("~")

            def fixed_length(ecg_, target_len=4096):
                if ecg_.shape[0] >= target_len:
                    return ecg_[:target_len, :]
                pad = np.zeros((target_len - ecg_.shape[0], ecg_.shape[1]), dtype=ecg_.dtype)
                return np.vstack([ecg_, pad])

        # Extract "I" from "1^I^..."
            def extract_lead(x):
                parts = x.strip().split("^")
                return parts[1].strip().upper()

            channels = [extract_lead(x) for x in raw_channels]

        # --- Step 2: Read OBX waveform rows ---
            rows = []

            while line.startswith("OBX"):
                line = ecg.readline()
                if not line:
                    break

                parts = line.split("|")
                if len(parts) < 6:
                    continue

                values = parts[5].split("^")

            # Ensure correct number of leads
                if len(values) != len(channels):
                    continue

                try:
                    row = [float(v) for v in values]
                    rows.append(row)
                except:
                    continue

        if len(rows) == 0:
            raise ValueError("No waveform data found")

        ecg = np.array(rows, dtype=np.float32)  # shape: (time, leads)


        lead_order = ["I","II","III","aVR","aVL","aVF",
                  "V1","V2","V3","V4","V5","V6"]

        lead_idx = {l: i for i, l in enumerate(channels)}

        if "III" not in lead_idx:
            ecg_III = ecg[:, lead_idx["II"]] - ecg[:, lead_idx["I"]]
            lead_idx["III"] = None
        if "aVR" not in lead_idx:
            ecg_aVR = -(ecg[:, lead_idx["I"]] + ecg[:, lead_idx["II"]]) / 2
            lead_idx["aVR"] = None
        if "aVF" not in lead_idx:
            ecg_aVF = (ecg[:, lead_idx["II"]] + ecg[:, lead_idx["III"]]) / 2
            lead_idx["aVF"] = None
        if "aVL" not in lead_idx:
            ecg_aVL = (ecg[:, lead_idx["I"]] - ecg[:, lead_idx["III"]]) / 2
            lead_idx["aVL"] = None

        final_leads = []
        for l in lead_order:
            if l in lead_idx and lead_idx[l] is not None:
                final_leads.append(ecg[:, lead_idx[l]])
            else:
                if l == "III":
                    final_leads.append(ecg_III)
                elif l == "aVR":
                    final_leads.append(ecg_aVR)
                elif l == "aVF":
                    final_leads.append(ecg_aVF)
                elif l == "aVL":
                    final_leads.append(ecg_aVL)

        ecg = np.stack(final_leads, axis=1)
        ecg = fixed_length(ecg, target_len = 4096)
        #ecg = ecg[:4096, :]
        ecg -= ecg.mean(axis=0, keepdims=True)
        ecg /= ecg.std(axis=0, keepdims=True) + 1e-6


        return ecg

    def __getitem__(self, idx):
        pid   = self.patient_ids[idx]
        group = self.groups.get_group(pid).sort_values("timestamp")
        group = group.tail(self.max_timestamps)


        arrays, timestamps, event_ages, age_deltas = [], [], [], []
        
        for _, row in group.iterrows():
            try:
                ecg = self.process_ecg_xml_format(row)

            except Exception as e:
                print(f"⚠️ Primary method failed, not xml, try hl7 parsing for {row['path']}: {e}")

                try:
                    ecg = self.process_ecg_hl7_format(row)

                except Exception as e2:
                    print(f"❌ Fallback also failed, not able to parse at all for {row['path']}: {e2}")
                    ecg = None

            if ecg is not None:
                arrays.append(ecg)
                timestamps.append(row["timestamp"])
                event_ages.append(row["EVENT_AGE"])
                age_deltas.append(row["age_delta_norm"])




        return {
            "patient_id":  pid,
            "timestamps":  timestamps,   # list[str], length = num valid ECGs
            "event_ages":  event_ages,   # list[float]
            "age_deltas":  age_deltas,   # list[float]
            "ecgs":        arrays,       # list of np.float32 [5000, 12]
        }


# ─────────────────────────────────────────────────────────────────────────────
# Collate — pads ECGs; forwards timestamps + event_ages as plain lists
# ─────────────────────────────────────────────────────────────────────────────

def collate_longitudinal(batch, fixed_max_seq_len=50):
    batch = [b for b in batch if len(b["ecgs"]) > 0]
    if not batch:
        return None

    max_seq = fixed_max_seq_len

    ecgs_padded      = []
    masks            = []
    patient_ids      = []
    timestamps_list  = []
    event_ages_list  = []
    age_deltas_list  = []

    for item in batch:
        ecgs = item["ecgs"][-max_seq:]
        timestamps = item["timestamps"][-max_seq:]
        event_ages = item["event_ages"][-max_seq:]
        age_deltas = item["age_deltas"][-max_seq:]



        n = len(ecgs)
        pad = max_seq - n

        #pad = max_seq - len(age_deltas)
        age_deltas = age_deltas + [0.0] * pad

        padded = ecgs + [np.zeros_like(ecgs[0]) for _ in range(pad)]
        ecgs_padded.append(torch.tensor(np.stack(padded)).float())

        mask = torch.zeros(max_seq, dtype=torch.bool)
        mask[:n] = True
        masks.append(mask)

        patient_ids.append(item["patient_id"])
        timestamps_list.append(timestamps)
        event_ages_list.append(event_ages)
        age_deltas_list.append(age_deltas)

    return (
        torch.stack(ecgs_padded),   # [B, 50, 5000, 12]
        torch.stack(masks),         # [B, 50]
        patient_ids,
        timestamps_list,
        event_ages_list,
        age_deltas_list
    )
# ─────────────────────────────────────────────────────────────────────────────
# Model loading helper
# ─────────────────────────────────────────────────────────────────────────────

def load_encoder(encoder_path, encoder_layer):
    """
    Load the full Keras model and extract a sub-model that outputs
    the specified layer (e.g. 'activation_18').
    """
    print(f"Loading encoder from {encoder_path}  (output layer: '{encoder_layer}') …")
    full_model   = keras.saving.load_model(encoder_path, compile=False)
    enc_output   = full_model.get_layer(encoder_layer).output
    encoder      = keras.Model(inputs=full_model.input,
                               outputs=enc_output,
                               name="ecg_encoder")
    encoder.trainable = False
    print(f"  ✓  Encoder output shape: {encoder.output_shape}")
    return full_model, encoder


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation — values passed in at runtime, not hardcoded
# ─────────────────────────────────────────────────────────────────────────────

def denorm(x, mean, std):
    return x * std + mean


# ─────────────────────────────────────────────────────────────────────────────
# Shared encoder helper — auto-detects output shape
# ─────────────────────────────────────────────────────────────────────────────

def encode_batch(encoder, ecgs_flat_np):
    """
    Run encoder on [N, ...] input and return [N, D] feature vectors.

    Auto-detects output rank:
      - rank 3 [N, PatchLen, D]  →  mean-pool over PatchLen  →  [N, D]
      - rank 2 [N, D]            →  used directly
    Any other rank raises a clear error.
    """
    out = encoder(ecgs_flat_np, training=False).numpy()
    if out.ndim == 3:
        # [N, PatchLen, D] → mean over patch dimension
        return out.mean(axis=1)
    elif out.ndim == 2:
        # [N, D] already flat
        return out
    else:
        raise ValueError(
            f"Unexpected encoder output shape {out.shape}. "
            f"Expected rank 2 [N, D] or rank 3 [N, PatchLen, D]."
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — encode: save encoder outputs to parquet
# ─────────────────────────────────────────────────────────────────────────────

def run_encode(encoder, loader, output_path, flush_every=100):
    """
    One row per ECG visit:
        patient_id | timestamp | event_age | visit_index | latent_0 … latent_{D-1}
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    all_rows    = []
    latent_cols = None
    D           = None
    flushed     = 0

    def _flush(rows, first):
        df = pd.DataFrame(rows)
        if first:
            df.to_parquet(output_path, index=False)
        else:
            existing = pd.read_parquet(output_path)
            pd.concat([existing, df], ignore_index=True).to_parquet(
                output_path, index=False
            )
        return True

    header_written = False

    for batch_idx, batch in enumerate(tqdm(loader, desc="Encoding visits")):
        if batch is None:
            continue

        ecgs, masks, patient_ids, timestamps_list, event_ages_list, age_deltas_list = batch
        B  = ecgs.shape[0]
        L  = ecgs.shape[1]
        flat_np = ecgs.reshape(B * L, *ecgs.shape[2:]).numpy().astype(np.float32)

        feat_flat = encode_batch(encoder, flat_np)      # [B*L, D]
        feat      = feat_flat.reshape(B, L, -1)         # [B,  L, D]

        D = feat.shape[2]
        if latent_cols is None:
            latent_cols = [f"latent_{i}" for i in range(D)]
            print(f"  Encoder output dim: {D}  →  latent_0 … latent_{D-1}")

        for i in range(B):
            seq_len    = int(masks[i].sum().item())
            pid        = patient_ids[i]
            timestamps = timestamps_list[i]
            ages       = event_ages_list[i]
            age_deltas = age_deltas_list[i]

            for v in range(seq_len):
                row = {
                    "patient_id":  pid,
                    "timestamp":   timestamps[v] if v < len(timestamps) else None,
                    "event_age":   float(ages[v]) if v < len(ages) else None,
                    "age_delta":   float(age_deltas[v]) if v < len(age_deltas) else None,
                    "visit_index": v,
                }
                row.update({latent_cols[d]: float(feat[i, v, d]) for d in range(D)})
                all_rows.append(row)

        if (batch_idx + 1) % flush_every == 0:
            header_written = _flush(all_rows, not header_written)
            flushed       += len(all_rows)
            all_rows       = []
            print(f"  Flushed {flushed} rows after batch {batch_idx + 1}")

    if all_rows:
        _flush(all_rows, not header_written)
        flushed += len(all_rows)

    total = pd.read_parquet(output_path).shape[0]
    print(f"\n✅  Encode done.  {total} rows → {output_path}")
    print(f"   Columns: patient_id, timestamp, event_age, age_delta, visit_index, "
          f"latent_0 … latent_{D-1}")


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — infer: single-ECG + longitudinal age prediction → CSV
# ─────────────────────────────────────────────────────────────────────────────

def extract_prediction(model_out, output_key):
    """
    Extract a scalar float from a model's output.

    output_key:
      "tensor"      →  model returns a plain tensor/array
      anything else →  model returns a dict; use output_key as the dict key
    """
    if output_key == "tensor":
        return float(np.array(model_out).flatten()[0])
    else:
        val = model_out[output_key]
        return float(np.array(val).flatten()[0])


def run_infer(full_model, encoder, long_model, loader,
              output_path, output_key, mean_norm, std_norm, use_delta, flush_every=100):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    header_written = False
    buffer         = []

    for batch_idx, batch in enumerate(tqdm(loader, desc="Inference")):
        if batch is None:
            continue

        ecgs, masks, patient_ids, timestamps_list, event_ages_list, age_deltas_list  = batch
        B  = ecgs.shape[0]
        L  = ecgs.shape[1]

        # Age delta
        age_delta_np = np.array(age_deltas_list, dtype=np.float32)
        age_delta_np = age_delta_np[..., np.newaxis]

        # Encode all visits — auto-detect pooling
        flat_np  = ecgs.reshape(B * L, *ecgs.shape[2:]).numpy().astype(np.float32)
        feat     = encode_batch(encoder, flat_np).reshape(B, L, -1).astype(np.float32)
        mask_np  = masks.numpy().astype(np.float32)

        if use_delta:
            feat = np.concatenate([age_delta_np, feat], axis=-1)  # (B, L, 513)



        # Longitudinal prediction for whole batch
        long_out        = long_model.predict(
            {"num": feat, "mask": mask_np}, verbose=0
        )
        long_out_np = {k: np.asarray(v) for k, v in long_out.items()}
        #long_preds_norm = np.array(long_out).flatten()

        # Single-ECG prediction per patient (latest valid visit)
        for i in range(B):
            seq_len       = int(masks[i].sum().item())
            latest_ecg_np = ecgs[i, seq_len - 1].numpy()[np.newaxis]
            #single_out    = full_model(latest_ecg_np, training=False)
            #single_norm   = extract_prediction(single_out, output_key)
            latest_age    = event_ages_list[i][seq_len - 1]
            latest_ts     = timestamps_list[i][seq_len - 1] if timestamps_list[i] else None
            latest_delta  = age_deltas_list[i][seq_len - 1] if age_deltas_list[i] else None

            row_long_out = {k: float(v[i].squeeze()) for k, v in long_out_np.items()}
            
            buffer.append({
                "patient_id":       patient_ids[i],
                "latest_timestamp": latest_ts,
                "num_ecgs":         seq_len,
                "true_age":         latest_age,
                "age_delta":        latest_delta,
                **row_long_out,
                #"pred_single_norm": single_norm,
                #"pred_single":      denorm(single_norm, mean_norm, std_norm),
                #"pred_long_norm":   float(long_preds_norm[i]),
                #"pred_long":        denorm(float(long_preds_norm[i]), mean_norm, std_norm),
            })
            

        if (batch_idx + 1) % flush_every == 0:
            pd.DataFrame(buffer).to_csv(
                output_path, mode="a", header=not header_written, index=False
            )
            header_written = True
            buffer         = []
            print(f"  Flushed after batch {batch_idx + 1}")

    if buffer:
        pd.DataFrame(buffer).to_csv(
            output_path, mode="a", header=not header_written, index=False
        )

    print(f"\n✅  Inference done. Predictions saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",               required=True,
                        choices=["encode", "infer"],
                        help="'encode' saves latents to parquet; "
                             "'infer' runs prediction to CSV")
    parser.add_argument("--path",               required=True,
                        help="Root directory containing patient ECG folders")
    parser.add_argument("--metadata_path",      required=True,
                        help="TSV file: FINNGENID, MEASID, EVENT_AGE, "
                             "APPROX_EVENT_DAY, TIME")
    parser.add_argument("--encoder_path",       required=True,
                        help="Path to single-ECG Keras model (.keras)")
    parser.add_argument("--encoder_layer",      required=True,
                        help="Layer name to tap as encoder output, e.g. activation_18")
    parser.add_argument("--longitudinal_path",  default=None,
                        help="Path to longitudinal Keras model (.keras) "
                             "[required for infer mode]")
    parser.add_argument("--output",             required=True,
                        help="Output path (.parquet for encode, .csv for infer)")
    # ── infer-mode options ────────────────────────────────────────────────────
    parser.add_argument("--output_key",         default="tensor",
                        help="How to read the single-ECG model output. "
                             "Use 'tensor' if the model returns a plain array, "
                             "or pass the dict key name (e.g. 'output_age_continuous') "
                             "if the model returns a dict.  [default: tensor]")
    parser.add_argument("--mean_norm",          type=float, default=54.4974,
                        help="Mean used to normalise the training target "
                             "(for de-normalising predictions).  [default: 0.0]")
    parser.add_argument("--std_norm",           type=float, default=20.7367,
                        help="Std used to normalise the training target.  "
                             "[default: 1.0  →  predictions are left as-is]")
    # ── shared options ────────────────────────────────────────────────────────
    parser.add_argument("--batch_size",         type=int, default=8)
    parser.add_argument("--max_seq_len",        type=int, default=50)
    parser.add_argument("--flush_every",        type=int, default=100)
    # ── flags ─────────────────────────────────────────────────────
    parser.add_argument("--delta_consider", action="store_true")
    args = parser.parse_args()

    if args.mode == "infer" and args.longitudinal_path is None:
        parser.error("--longitudinal_path is required for infer mode")

    # ── Load encoder ──────────────────────────────────────────────────────────
    full_model, encoder = load_encoder(args.encoder_path, args.encoder_layer)

    # ── Dataset / DataLoader ──────────────────────────────────────────────────
    dataset = LongitudinalECGFromMetadata(
        args.metadata_path,
        data_path=args.path,
        max_timestamps=args.max_seq_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_longitudinal, fixed_max_seq_len=args.max_seq_len),
        num_workers=0,
    )

    # ── Run selected mode ─────────────────────────────────────────────────────
    if args.mode == "encode":
        run_encode(encoder, loader, args.output, flush_every=args.flush_every)

    else:  # infer
        print(f"Loading longitudinal model from {args.longitudinal_path} …")
        long_model = keras.saving.load_model(
            args.longitudinal_path, compile=False
        )
        print("  ✓  Longitudinal model loaded")
        print(f"  output_key='{args.output_key}'  "
              f"mean_norm={args.mean_norm}  std_norm={args.std_norm}")
        run_infer(
            full_model, encoder, long_model, loader,
            args.output,
            output_key=args.output_key,
            mean_norm=args.mean_norm,
            std_norm=args.std_norm,
            use_delta=args.delta_consider,
            flush_every=args.flush_every
        )


if __name__ == "__main__":
    main()
