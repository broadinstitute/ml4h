import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from ml4h.TensorMap import TensorMap, Interpretation
from ml4h.models.model_factory import get_custom_objects

n_intervals = 25

af_tmap = TensorMap(
    'survival_curve_af',
    Interpretation.SURVIVAL_CURVE,
    shape=(n_intervals*2,),
)

death_tmap = TensorMap(
    'death_event',
    Interpretation.SURVIVAL_CURVE,
    shape=(n_intervals*2,),
)

sex_tmap = TensorMap(name='sex', interpretation=Interpretation.CATEGORICAL, channel_map={'Female': 0, 'Male':1})
age_tmap = TensorMap(name='age_in_days', interpretation=Interpretation.CONTINUOUS, channel_map={'age_in_days': 0})
af_in_read_tmap = TensorMap(name='af_in_read', interpretation=Interpretation.CATEGORICAL, channel_map={'no_af_in_read': 0, 'af_in_read':1})

output_tensormaps = {tm.output_name(): tm for tm in [af_tmap, death_tmap, sex_tmap, age_tmap, af_in_read_tmap]}
custom_dict = get_custom_objects(list(output_tensormaps.values()))
model = load_model('ecg2af_quintuplet_v2024_01_13.h5', custom_objects=custom_dict)

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