import os
import sys
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from ml4h.TensorMap import TensorMap, Interpretation
from ml4h.defines import ECG_REST_AMP_LEADS
from ml4h.models.model_factory import get_custom_objects

n_intervals = 25

ecg_tmap = TensorMap(
    'ecg_5000_std',
    Interpretation.CONTINUOUS,
    shape=(5000, 12),
    channel_map=ECG_REST_AMP_LEADS
)

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
space_dict = defaultdict(list)

def process_file(filepath, space_dict):
    # Placeholder for file processing logic
    print(f"Processing file: {filepath}")
    with h5py.File(filepath, 'r') as hd5:
        tensor = np.zeros(ecg_tmap.shape, dtype=np.float32)
        for lead in ecg_tmap.channel_map:
            tensor[:, ecg_tmap.channel_map[lead]] = hd5[f'/ukb_ecg_rest/strip_{lead}/instance_0']

        tensor -= tensor.mean()
        tensor /= (tensor.std() + 1e-6)
        #print(f"Got tensor: {tensor.mean():0.3f}")
        prediction = model.predict(np.expand_dims(tensor, axis=0), verbose=0)
        if len(model.output_names) == 1:
            prediction = [prediction]
        predictions_dict = {name: pred for name, pred in zip(model.output_names, prediction)}
        #print(f"Got predictions: {predictions_dict}")
        space_dict['sample_id'].append(os.path.basename(filepath).replace('.hd5', ''))
        space_dict['ecg_path'].append(filepath)
        for otm in output_tensormaps.values():
            y = predictions_dict[otm.output_name()]
            if otm.is_categorical():
                space_dict[f'{otm.name}_prediction'].append(y[0, 1])
            elif otm.is_continuous():
                space_dict[f'{otm.name}_prediction'].append(y[0, 0])
            elif otm.is_survival_curve():
                intervals = otm.shape[-1] // 2
                days_per_bin = 1 + (2 * otm.days_window) // intervals
                predicted_survivals = np.cumprod(y[:, :intervals], axis=1)
                space_dict[f'{otm.name}_prediction'].append(str(1 - predicted_survivals[0, -1]))
                # print(f' got target: {target[otm.output_name()].numpy().shape}')
                # sick = np.sum(target[otm.output_name()].numpy()[:, intervals:], axis=-1)
                # follow_up = np.cumsum(target[otm.output_name()].numpy()[:, :intervals], axis=-1)[:, -1] * days_per_bin
                # space_dict[f'{otm.name}_event'].append(str(sick[b]))
                # space_dict[f'{otm.name}_follow_up'].append(str(follow_up[b]))
    # Example: Use the model to make a prediction (add real processing logic here)

def main(directory):
    # Iterate over all files in the specified directory
    space_dict = defaultdict(list)
    for i,filename in enumerate(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            process_file(filepath, space_dict)
        if i > 100:
            break

    df = pd.DataFrame.from_dict(space_dict)
    df.to_csv('/output/ecg2af_quintuplet.csv', index=False)

if __name__ == "__main__":
    # Take directory path from command-line arguments
    directory = sys.argv[1] if len(sys.argv) > 1 else "/data"
    main(directory)