import argparse
import json
import pandas as pd

latent_dimensions = 320

def finalize(input_csv, predictions_json, output_csv):
    with open(predictions_json, "r") as f:
        prediction_data = json.load(f)

    df = pd.read_csv(input_csv, dtype={"file_id": str})

    embedding = prediction_data["embed"]

    if len(embedding) != len(df):
        raise ValueError(f"Mismatch: {len(embedding)} predictions but {len(df)} rows in input CSV!")

    new_frame = pd.DataFrame(embedding, columns=[f'pclr_{i}' for i in range(latent_dimensions)])
    df = pd.concat([df, new_frame], axis=1)

    df.to_csv(output_csv, index=False)
    print(f"✅ Predictions written to {output_csv} ({len(df)} rows).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to final CSV with predictions")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON")
    args = parser.parse_args()

    finalize(args.input, args.predictions, args.output)
