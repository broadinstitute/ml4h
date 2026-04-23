"""
Externe validatie van LVM-AI modellen op eigen dataset.

Drie pre-trained modellen worden ondersteund:
  1. ecg_rest_raw_lvm_symmetric_loss.h5        - ECG only, symmetric loss
  2. ecg_rest_raw_lvm_asymmetric_loss.h5       - ECG only, asymmetric loss
  3. ecg_rest_raw_age_sex_bmi_lvm_asymmetric_loss.h5 - ECG + leeftijd/geslacht/BMI

Invoer-opties voor ECG-data:
  A) XML-bestanden (GE MUSE-formaat)  → worden automatisch geconverteerd
  B) NumPy .npy bestanden  (shape: 5000 x 12)
  C) CSV-bestanden per patiënt        (kolommen: I, II, III, aVR, aVL, aVF, V1..V6)

Label-CSV (labels.csv) vereiste kolommen:
  - sample_id   : unieke ID die overeenkomt met bestandsnaam
  - lvm_g       : gemeten LV massa in gram (float, optioneel voor MAE)
  - lvh_label   : 1=LVH aanwezig, 0=geen LVH (int, optioneel voor AUC)
  - age         : leeftijd in jaren (float, vereist voor age_sex_bmi model)
  - sex         : 0=vrouw, 1=man (int, vereist voor age_sex_bmi model)
  - bmi         : BMI in kg/m² (float, vereist voor age_sex_bmi model)

Gebruik:
  python validate_lvm_external.py \
      --model ecg_rest_raw_age_sex_bmi_lvm_asymmetric_loss.h5 \
      --ecg_dir /pad/naar/ecg_bestanden \
      --ecg_format xml \
      --labels /pad/naar/labels.csv \
      --output_dir /pad/naar/resultaten
"""

import os
import sys
import argparse
import logging
import struct
import base64
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Volgorde van afleiders die het model verwacht ──────────────────────────
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ECG_SAMPLES = 5000          # 500 Hz × 10 s
ECG_NORMALIZATION = 2000.0  # deelfactor opgegeven in paper


# ══════════════════════════════════════════════════════════════════════════
#  ECG LADEN
# ══════════════════════════════════════════════════════════════════════════

def _decode_b64_lead(raw: str, scale: float = 1.0) -> np.ndarray:
    decoded = base64.b64decode(raw)
    samples = [
        struct.unpack("h", bytes([decoded[i], decoded[i + 1]]))[0]
        for i in range(0, len(decoded), 2)
    ]
    return np.array(samples, dtype=np.float32) * scale


def load_ecg_from_xml(xml_path: str) -> np.ndarray:
    """
    Leest een GE MUSE XML en geeft een (5000, 12) array terug.
    Alleen het Rhythm-waveform wordt gebruikt; Median-waveforms worden overgeslagen.
    De afleiders III, aVR, aVL en aVF worden afgeleid via vectorrekenkunde.
    """
    try:
        import bs4
    except ImportError:
        raise ImportError("Installeer beautifulsoup4 en lxml:  pip install beautifulsoup4 lxml")

    with open(xml_path, "r", errors="replace") as f:
        soup = bs4.BeautifulSoup(f, "lxml")

    voltage: dict[str, np.ndarray] = {}
    for waveform in soup.find_all("waveform"):
        wt = waveform.find("waveformtype")
        if wt is None or wt.text.strip() != "Rhythm":
            continue
        for lead_tag in waveform.find_all("leaddata"):
            lead_id = lead_tag.find("leadid").text.strip()
            scale_tag = lead_tag.find("leadamplitudeunitsperbit")
            scale = float(scale_tag.text) if scale_tag else 1.0
            raw = lead_tag.find("waveformdata").text.strip()
            voltage[lead_id] = _decode_b64_lead(raw, scale)
        break  # gebruik alleen het eerste Rhythm-blok

    if not voltage:
        raise ValueError(f"Geen Rhythm-waveform gevonden in {xml_path}")

    required = {"I", "II", "V1", "V2", "V3", "V4", "V5", "V6"}
    missing = required - set(voltage.keys())
    if missing:
        raise ValueError(f"Afleiders ontbreken: {missing} in {xml_path}")

    # Bereken ontbrekende afleiders
    voltage["III"] = voltage["II"] - voltage["I"]
    voltage["aVR"] = -0.5 * (voltage["I"] + voltage["II"])
    voltage["aVL"] = voltage["I"] - voltage["II"] / 2
    voltage["aVF"] = voltage["II"] - voltage["I"] / 2

    # Resample naar 5000 monsters als nodig
    def _resample(v: np.ndarray) -> np.ndarray:
        n = len(v)
        if n == ECG_SAMPLES:
            return v
        if n == 2500:
            return np.interp(np.linspace(0, n, ECG_SAMPLES), np.arange(n), v)
        if n > ECG_SAMPLES:
            idx = np.round(np.linspace(0, n - 1, ECG_SAMPLES)).astype(int)
            return v[idx]
        raise ValueError(f"Kan afleider van lengte {n} niet resamplen naar {ECG_SAMPLES}")

    matrix = np.column_stack([_resample(voltage[lead]) for lead in LEAD_ORDER])
    return matrix.astype(np.float32)


def load_ecg_from_npy(npy_path: str) -> np.ndarray:
    arr = np.load(npy_path).astype(np.float32)
    if arr.shape == (12, ECG_SAMPLES):
        arr = arr.T
    if arr.shape != (ECG_SAMPLES, 12):
        raise ValueError(f"Verwacht shape ({ECG_SAMPLES}, 12) maar kreeg {arr.shape} in {npy_path}")
    return arr


def load_ecg_from_csv(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    missing = [c for c in LEAD_ORDER if c not in df.columns]
    if missing:
        raise ValueError(f"Kolommen ontbreken in {csv_path}: {missing}")
    arr = df[LEAD_ORDER].values.astype(np.float32)
    if arr.shape[0] != ECG_SAMPLES:
        raise ValueError(f"Verwacht {ECG_SAMPLES} rijen maar kreeg {arr.shape[0]} in {csv_path}")
    return arr


ECG_LOADERS = {
    "xml": (load_ecg_from_xml, [".xml"]),
    "npy": (load_ecg_from_npy, [".npy"]),
    "csv": (load_ecg_from_csv, [".csv"]),
}


def collect_ecg_files(ecg_dir: str, fmt: str) -> dict[str, str]:
    """Geeft {sample_id: bestandspad} terug voor alle ECG-bestanden in ecg_dir."""
    _, extensions = ECG_LOADERS[fmt]
    result = {}
    for p in Path(ecg_dir).iterdir():
        if p.suffix.lower() in extensions:
            result[p.stem] = str(p)
    return result


# ══════════════════════════════════════════════════════════════════════════
#  MODEL LADEN EN INFERENTIE
# ══════════════════════════════════════════════════════════════════════════

def load_model(model_path: str):
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow niet gevonden. Installeer: pip install tensorflow==2.19.0")
    model = tf.keras.models.load_model(model_path, compile=False)
    log.info(f"Model geladen: {model_path}")
    log.info(f"  Inputs : {[i.name for i in model.inputs]}")
    log.info(f"  Outputs: {[o.name for o in model.outputs]}")
    return model


def needs_demographics(model) -> bool:
    return len(model.inputs) > 1


def run_inference(
    model,
    ecg_matrix: np.ndarray,
    age: float = None,
    sex: float = None,
    bmi: float = None,
) -> dict[str, float]:
    """
    Geeft {'lvm_pred': float, 'lvh_prob': float (optioneel)} terug.
    ecg_matrix: (5000, 12) niet-genormaliseerde waarden in μV of mV
    """
    ecg_norm = ecg_matrix / ECG_NORMALIZATION
    ecg_batch = ecg_norm[np.newaxis, ...]  # (1, 5000, 12)

    if needs_demographics(model):
        if any(v is None for v in [age, sex, bmi]):
            raise ValueError("Dit model vereist age, sex en bmi.")
        demo_batch = np.array([[age, sex, bmi]], dtype=np.float32)
        preds = model.predict([ecg_batch, demo_batch], verbose=0)
    else:
        preds = model.predict(ecg_batch, verbose=0)

    # Sommige modellen geven lijst van arrays terug, andere een enkele array
    if isinstance(preds, (list, tuple)):
        lvm_pred = float(preds[0][0])
        lvh_prob = float(preds[1][0]) if len(preds) > 1 else None
    else:
        lvm_pred = float(preds[0])
        lvh_prob = None

    result = {"lvm_pred": lvm_pred}
    if lvh_prob is not None:
        result["lvh_prob"] = lvh_prob
    return result


# ══════════════════════════════════════════════════════════════════════════
#  STATISTIEKEN
# ══════════════════════════════════════════════════════════════════════════

def compute_metrics(df: pd.DataFrame) -> dict:
    metrics = {}

    # ─ LV-massa regressie ─────────────────────────────────────────────────
    if "lvm_g" in df.columns and df["lvm_g"].notna().sum() > 0:
        mask = df["lvm_g"].notna() & df["lvm_pred"].notna()
        true = df.loc[mask, "lvm_g"].values
        pred = df.loc[mask, "lvm_pred"].values
        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        corr = float(np.corrcoef(true, pred)[0, 1])
        metrics["n_lvm"] = int(mask.sum())
        metrics["MAE_g"] = round(mae, 2)
        metrics["RMSE_g"] = round(rmse, 2)
        metrics["Pearson_r"] = round(corr, 4)
        log.info(f"LVM regressie  n={metrics['n_lvm']}  MAE={metrics['MAE_g']} g  "
                 f"RMSE={metrics['RMSE_g']} g  r={metrics['Pearson_r']}")

    # ─ LVH classificatie ──────────────────────────────────────────────────
    if "lvh_label" in df.columns and "lvh_prob" in df.columns:
        mask = df["lvh_label"].notna() & df["lvh_prob"].notna()
        y_true = df.loc[mask, "lvh_label"].values.astype(int)
        y_prob = df.loc[mask, "lvh_prob"].values.astype(float)

        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            auc = roc_auc_score(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            metrics["n_lvh"] = int(mask.sum())
            metrics["AUC_ROC"] = round(auc, 4)
            metrics["AUC_PR"] = round(ap, 4)
            log.info(f"LVH classificatie  n={metrics['n_lvh']}  "
                     f"AUC-ROC={metrics['AUC_ROC']}  AUC-PR={metrics['AUC_PR']}")

            # Bootstrap 95 % CI voor AUC
            rng = np.random.default_rng(42)
            boot_aucs = []
            for _ in range(2000):
                idx = rng.integers(0, len(y_true), len(y_true))
                if len(np.unique(y_true[idx])) < 2:
                    continue
                boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
            if boot_aucs:
                ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
                metrics["AUC_ROC_CI95"] = f"[{ci_lo:.4f}, {ci_hi:.4f}]"
                log.info(f"  95 %% CI AUC-ROC: {metrics['AUC_ROC_CI95']}")
        except ImportError:
            log.warning("scikit-learn niet beschikbaar; AUC wordt niet berekend. "
                        "Installeer: pip install scikit-learn")

    return metrics


# ══════════════════════════════════════════════════════════════════════════
#  HOOFD-LOOP
# ══════════════════════════════════════════════════════════════════════════

def run_validation(
    model_path: str,
    ecg_dir: str,
    ecg_format: str,
    labels_csv: str,
    output_dir: str,
) -> pd.DataFrame:

    os.makedirs(output_dir, exist_ok=True)

    # Laad labels
    labels = pd.read_csv(labels_csv, dtype={"sample_id": str})
    labels["sample_id"] = labels["sample_id"].str.strip()
    log.info(f"Labels geladen: {len(labels)} rijen")

    # Verzamel ECG-bestanden
    ecg_files = collect_ecg_files(ecg_dir, ecg_format)
    log.info(f"ECG-bestanden gevonden: {len(ecg_files)}")

    # Laad model
    model = load_model(model_path)
    use_demo = needs_demographics(model)

    # Inferentie per patiënt
    loader, _ = ECG_LOADERS[ecg_format]
    results = []

    for _, row in labels.iterrows():
        sid = str(row["sample_id"]).strip()
        if sid not in ecg_files:
            log.warning(f"Geen ECG gevonden voor sample_id={sid} – overgeslagen")
            continue

        try:
            ecg = loader(ecg_files[sid])
        except Exception as e:
            log.warning(f"Fout bij laden ECG voor {sid}: {e}")
            continue

        try:
            kwargs = {}
            if use_demo:
                kwargs = {
                    "age": float(row["age"]),
                    "sex": float(row["sex"]),
                    "bmi": float(row["bmi"]),
                }
            pred = run_inference(model, ecg, **kwargs)
        except Exception as e:
            log.warning(f"Inferentie mislukt voor {sid}: {e}")
            continue

        entry = {"sample_id": sid, **pred}
        for col in ["lvm_g", "lvh_label", "age", "sex", "bmi"]:
            if col in row.index:
                entry[col] = row[col]
        results.append(entry)

    if not results:
        raise RuntimeError("Geen voorspellingen gegenereerd. Controleer ECG-bestanden en labels.")

    df = pd.DataFrame(results)
    log.info(f"Voorspellingen gegenereerd: {len(df)} patiënten")

    # Statistieken
    metrics = compute_metrics(df)

    # Sla op
    out_preds = os.path.join(output_dir, "predictions.csv")
    df.to_csv(out_preds, index=False)
    log.info(f"Voorspellingen opgeslagen: {out_preds}")

    out_metrics = os.path.join(output_dir, "metrics.csv")
    pd.DataFrame([metrics]).to_csv(out_metrics, index=False)
    log.info(f"Statistieken opgeslagen:   {out_metrics}")

    return df


# ══════════════════════════════════════════════════════════════════════════
#  COMMAND-LINE INTERFACE
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Externe validatie LVM-AI modellen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        help="Pad naar het .h5 modelbestand",
    )
    p.add_argument(
        "--ecg_dir",
        required=True,
        help="Map met ECG-bestanden (XML, NPY of CSV)",
    )
    p.add_argument(
        "--ecg_format",
        choices=["xml", "npy", "csv"],
        default="xml",
        help="Formaat van de ECG-bestanden (standaard: xml)",
    )
    p.add_argument(
        "--labels",
        required=True,
        help="Pad naar labels.csv",
    )
    p.add_argument(
        "--output_dir",
        default="./lvm_validation_results",
        help="Map voor resultaten (standaard: ./lvm_validation_results)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_validation(
        model_path=args.model,
        ecg_dir=args.ecg_dir,
        ecg_format=args.ecg_format,
        labels_csv=args.labels,
        output_dir=args.output_dir,
    )
