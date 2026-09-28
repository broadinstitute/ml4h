import argparse
import json
import numpy as np
import pandas as pd


def convert_survival_curve_to_risk_score(curve):
    curve = np.array(curve)
    return 1 - np.cumprod(curve[:25])[-1]


def finalize(input_csv, predictions_json, output_csv):
    with open(predictions_json, "r") as f:
        prediction_data = json.load(f)

    df = pd.read_csv(input_csv, dtype={"file_id": str})

    age = prediction_data["output_age_from_wide_csv_continuous"]
    af = prediction_data["output_af_in_read_categorical"]
    sex = prediction_data["output_sex_from_wide_categorical"]
    curves = prediction_data["output_survival_curve_af_survival_curve"]

    if len(age) != len(df):
        raise ValueError(f"Mismatch: {len(age)} predictions but {len(df)} rows in input CSV!")

    df["output_age"] = [row[0] for row in age]
    df["output_af_0"] = [row[0] for row in af]
    df["output_af_1"] = [row[1] for row in af]
    df["output_sex_male"] = [row[0] for row in sex]
    df["output_sex_female"] = [row[1] for row in sex]
    df["af_risk_score"] = [convert_survival_curve_to_risk_score(row) for row in curves]

    df.to_csv(output_csv, index=False)
    print(f"✅ Predictions written to {output_csv} ({len(df)} rows).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to final CSV with predictions")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON")
    args = parser.parse_args()

    finalize(args.input, args.predictions, args.output)