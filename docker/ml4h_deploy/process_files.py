import os
import sys
import base64
import struct
from collections import defaultdict

import h5py
import xmltodict
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

hf_nlp_tmap = TensorMap(
    'hf_nlp_event',
    Interpretation.SURVIVAL_CURVE,
    shape=(n_intervals*2,),
)

hf_primary_tmap = TensorMap(
    'hf_primary_event',
    Interpretation.SURVIVAL_CURVE,
    shape=(n_intervals*2,),
)


death_tmap = TensorMap(
    'death_event',
    Interpretation.SURVIVAL_CURVE,
    shape=(n_intervals*2,),
)
is_male_tmap = TensorMap(
    'is_male', Interpretation.CATEGORICAL, channel_map={'Female': 0, 'Male': 1},
)
sex_tmap = TensorMap(name='sex', interpretation=Interpretation.CATEGORICAL, channel_map={'Female': 0, 'Male':1})
age_tmap = TensorMap(name='age_in_days', interpretation=Interpretation.CONTINUOUS, channel_map={'age_in_days': 0})
af_in_read_tmap = TensorMap(name='af_in_read', interpretation=Interpretation.CATEGORICAL, channel_map={'no_af_in_read': 0, 'af_in_read':1})

output_tensormaps = {tm.output_name(): tm for tm in [hf_nlp_tmap, hf_primary_tmap, death_tmap, is_male_tmap, age_tmap]}
custom_dict = get_custom_objects(list(output_tensormaps.values()))
model = load_model('ecg_5000_hf_quintuplet_dropout_v2023_04_17.keras')
output_file = '/output/ecg2hf_quintuplet.csv'
space_dict = defaultdict(list)

def process_ukb_hd5(filepath, space_dict):
    # Placeholder for file processing logic
    print(f"Processing file: {filepath}")
    with h5py.File(filepath, 'r') as hd5:
        ecg_array = np.zeros(ecg_tmap.shape, dtype=np.float32)
        for lead in ecg_tmap.channel_map:
            ecg_array[:, ecg_tmap.channel_map[lead]] = hd5[f'/ukb_ecg_rest/strip_{lead}/instance_0']

        ecg_array -= ecg_array.mean()
        ecg_array /= (ecg_array.std() + 1e-6)
        #print(f"Got tensor: {tensor.mean():0.3f}")
        prediction = model.predict(np.expand_dims(ecg_array, axis=0), verbose=0)
        if len(model.output_names) == 1:
            prediction = [prediction]
        predictions_dict = {name: pred for name, pred in zip(model.output_names, prediction)}
        #print(f"Got predictions: {predictions_dict}")
        space_dict['sample_id'].append(os.path.basename(filepath).replace('.hd5', ''))
        space_dict['ecg_path'].append(filepath)
        if '/dates/atrial_fibrillation_or_flutter_date' in hd5:
            space_dict['has_af'].append(1)
        else:
            space_dict['has_af'].append(0)

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

def decode_ekg_muse(raw_wave):
    """
    Ingest the base64 encoded waveforms and transform to numeric
    """
    # covert the waveform from base64 to byte array
    arr = base64.b64decode(bytes(raw_wave, 'utf-8'))

    # unpack every 2 bytes, little endian (16 bit encoding)
    unpack_symbols = ''.join([char * int(len(arr) / 2) for char in 'h'])
    byte_array = struct.unpack(unpack_symbols, arr)
    return byte_array

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

def process_ge_muse_xml(filepath, space_dict):
    """

    Upload the ECG as numpy array with shape=[2500,12,1] ([time, leads, 1]).

    The voltage unit should be in 1 mv/unit and the sampling rate should be 250/second (total 10 second).

    The leads should be ordered as follow I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.

    """
    try:
        with open(filepath, 'rb') as fd:
            content = fd.read()
            if not content.strip():
                print(f"Skipping empty file: {filepath}")
                return
            try:
                decoded = content.decode('utf-8')
            except UnicodeDecodeError:
                print(f"Skipping non-UTF8 file: {filepath}")
                return

            dic = xmltodict.parse(decoded)
            print(f"Successfully parsed XML from: {filepath}")
            # continue processing dic...

    except xmltodict.expat.ExpatError as e:
        print(f"XML parsing error in file {filepath}: {e}")
        return
    except Exception as e:
        print(f"Unexpected error processing {filepath}: {e}")
        return

    try:
        patient_id = dic['RestingECG']['PatientDemographics']['PatientID']
    except:
        print("no PatientID")
        patient_id = "none"
    try:
        pharma_unique_ecg_id = dic['RestingECG']['PharmaData']['PharmaUniqueECGID']
    except:
        print("no PharmaUniqueECGID")
        pharma_unique_ecg_id = "none"
    try:
        acquisition_date_time = dic['RestingECG']['TestDemographics']['AcquisitionDate'] + "_" + \
                              dic['RestingECG']['TestDemographics']['AcquisitionTime'].replace(":", "-")
    except:
        print("no AcquisitionDateTime")
        acquisition_date_time = "none"

        # try:
    #     requisition_number = dic['RestingECG']['Order']['RequisitionNumber']
    # except:
    #     print("no requisition_number")
    #     requisition_number = "none"

    # need to instantiate leads in the proper order for the model
    lead_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    """
    Each EKG will have this data structure:
    lead_data = {
        'I': np.array
    }
    """

    lead_data = dict.fromkeys(lead_order)
    # lead_data = {leadid: None for k in lead_order}

    #     for all_lead_data in dic['RestingECG']['Waveform']:
    #         for single_lead_data in lead['LeadData']:
    #             leadname =  single_lead_data['LeadID']
    #             if leadname in (lead_order):
    if 'RestingECG' not in dic or 'Waveform' not in dic['RestingECG']:
        print(f"Missing 'RestingECG' or 'Waveform' in file {filepath}, returning.")
        return
    for lead in dic['RestingECG']['Waveform']:
        if not isinstance(lead, dict) or 'LeadData' not in lead:
            print(f"Lead data is not a dictionary with LeadData key at {filepath}, returning.")
            return
        for leadid in range(len(lead['LeadData'])):
            try:
                sample_length = len(decode_ekg_muse_to_array(lead['LeadData'][leadid]['WaveFormData']))
            except:
                print("Failed to decode lead data to array, returning.")
                return
            # sample_length is equivalent to dic['RestingECG']['Waveform']['LeadData']['LeadSampleCountTotal']
            if sample_length == 5000:
                lead_data[lead['LeadData'][leadid]['LeadID']] = decode_ekg_muse_to_array(
                    lead['LeadData'][leadid]['WaveFormData'], downsample=1)
            elif sample_length == 2500:
                lead_data[lead['LeadData'][leadid]['LeadID']] = decode_ekg_muse_to_array(
                    lead['LeadData'][leadid]['WaveFormData'], downsample=2)
            else:
                continue
        # ensures all leads have 2500 samples and also passes over the 3 second waveform

    lead_data['III'] = (np.array(lead_data["II"]) - np.array(lead_data["I"]))
    lead_data['aVR'] = -(np.array(lead_data["I"]) + np.array(lead_data["II"])) / 2
    lead_data['aVF'] = (np.array(lead_data["II"]) + np.array(lead_data["III"])) / 2
    lead_data['aVL'] = (np.array(lead_data["I"]) - np.array(lead_data["III"])) / 2

    lead_data = {k: lead_data[k] for k in lead_order}
    # drops V3R, V4R, and V7 if it was a 15-lead ECG

    # now construct and reshape the array
    # converting the dictionary to an np.array
    temp = []
    for key, value in lead_data.items():
        temp.append(value)

    # transpose to be [time, leads, ]
    ecg_array = np.array(temp).T

    print(f'Writing row of ECG2AF predictions for ECG {patient_id}, at {acquisition_date_time}')
    ecg_array -= ecg_array.mean()
    ecg_array /= (ecg_array.std() + 1e-6)
    #print(f"Got tensor: {tensor.mean():0.3f}")
    prediction = model.predict(np.expand_dims(ecg_array, axis=0), verbose=0)
    if len(model.output_names) == 1:
        prediction = [prediction]
    predictions_dict = {name: pred for name, pred in zip(model.output_names, prediction)}
    #print(f"Got predictions: {predictions_dict}")
    space_dict['filepath'].append(os.path.basename(filepath))
    space_dict['patient_id'].append(patient_id)
    space_dict['acquisition_datetime'].append(acquisition_date_time)
    space_dict['pharma_unique_ecg_id'].append(pharma_unique_ecg_id)

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
    for root, _, files in os.walk(directory):
        for i, filename in enumerate(files):
            filepath = os.path.join(root, filename)
            if os.path.isfile(filepath):
                process_ge_muse_xml(filepath, space_dict)
            # if i > 10000:
            #     break

    df = pd.DataFrame.from_dict(space_dict)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    # Take directory path from command-line arguments
    directory = sys.argv[1] if len(sys.argv) > 1 else "/data"
    main(directory)