import sys
import os
from tensorflow.keras.models import load_model

# Load the model
model = load_model('ecg2af_quintuplet_v2024_01_13.h5')

def process_file(filepath):
    # Placeholder for file processing logic
    print(f"Processing file: {filepath}")
    # Example: Use the model to make a prediction (add real processing logic here)

def main(directory):
    # Iterate over all files in the specified directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            process_file(filepath)

if __name__ == "__main__":
    # Take directory path from command-line arguments
    directory = sys.argv[1] if len(sys.argv) > 1 else "/data"
    main(directory)