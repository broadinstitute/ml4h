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

    df = pd.DataFrame()

    age = prediction_data["output_0"]
    af = prediction_data["output_1"]
    sex = prediction_data["output_2"]
    curves = prediction_data["output_3"]
    mortality = prediction_data["output_4"]

    df["output_0"] = [row[0] for row in age]
    df["output_1"] = [row[0] for row in af]
    df["output_2"] = [row[0] for row in sex]
    df["output_3"] = [row[0] for row in curves]
    df["output_4"] = [row[0] for row in mortality]
    df.to_csv(output_csv, index=False)
    print(f"✅ Predictions written to {output_csv} ({len(df)} rows).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to final CSV with predictions")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON")
    args = parser.parse_args()
    finalize(args.input, args.predictions, args.output)
