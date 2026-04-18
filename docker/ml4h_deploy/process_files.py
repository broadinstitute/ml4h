import os
import sys
import base64
import struct
from collections import defaultdict

import argparse
import h5py
import xmltodict
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from ml4h.TensorMap import TensorMap, Interpretation
from ml4h.defines import ECG_REST_AMP_LEADS
from ml4h.models.model_factory import get_custom_objects
from ml4h.metrics import weighted_crossentropy

n_intervals = 25

ecg_tmap = TensorMap(
    'ecg_4096_std',
    Interpretation.CONTINUOUS,
    shape=(4096, 12),
    channel_map=ECG_REST_AMP_LEADS,
)

lvef_tmap = TensorMap('lvef', Interpretation.CONTINUOUS, channel_map={'lvef': 0})
nlp_as_tmap = TensorMap('nlp_as_label', Interpretation.CATEGORICAL, 
                        channel_map={'no_as': 0, 'severe_as':1, 'prosthetic_valve':2, 'no_data':3})
ecg_age_tmap = TensorMap('ecg_age', Interpretation.CONTINUOUS, channel_map={'ecg_age': 0})
echo_age_tmap = TensorMap('echo_age', Interpretation.CONTINUOUS, channel_map={'echo_age': 0})
lbbb_tmap = TensorMap(name='lbbb', interpretation=Interpretation.CATEGORICAL, 
                            loss=weighted_crossentropy([1.0, 10.0], 'lbbb'),
                            channel_map={'no_lbbb': 0, 'lbbb':1})
rbbb_tmap = TensorMap(name='rbbb', interpretation=Interpretation.CATEGORICAL, 
                            loss=weighted_crossentropy([1.0, 10.0], 'rbbb'),
                            channel_map={'no_rbbb': 0, 'rbbb':1})
avb_tmap = TensorMap(name='avb', interpretation=Interpretation.CATEGORICAL, 
                            loss=weighted_crossentropy([1.0, 10.0], 'avb'),
                            channel_map={'no_avb': 0, 'avb':1})
af_in_read_tmap = TensorMap(name='af_in_read', interpretation=Interpretation.CATEGORICAL, 
                            loss=weighted_crossentropy([1.0, 10.0], 'af_in_read'),
                            channel_map={'no_af_in_read': 0, 'af_in_read':1})

input_tmaps = [ecg_tmap]
output_tmaps = [lvef_tmap, nlp_as_tmap, echo_age_tmap, ecg_age_tmap, lbbb_tmap, rbbb_tmap, avb_tmap, af_in_read_tmap]
ecg_label_tmaps = [ecg_age_tmap, lbbb_tmap, rbbb_tmap, avb_tmap, af_in_read_tmap]
ecg_labels = ['is_male', 'lvh', 'aortic_stenosis', 'dm', 'cad', 'mi', 'htn', 'valve_dz', 'hypertension_med', 'afib', 'obesity', 'ckd']
for d in ecg_labels:
    d_tmap = TensorMap(d, Interpretation.CATEGORICAL, channel_map={f'no_{d}': 0, f'{d}':1})
    output_tmaps.append(d_tmap)
    ecg_label_tmaps.append(d_tmap)
                        
cutpoints = [30, 35, 40, 45, 50, 55]
for cutpoint in cutpoints:
    output_tmaps.append(
        TensorMap(name=f'lvef_lt_{cutpoint}', interpretation=Interpretation.CATEGORICAL, 
                  channel_map={f'no_lvef_lt_{cutpoint}':0, f'lvef_lt_{cutpoint}': 1})
    )







output_tensormaps = {tm.output_name(): tm for tm in output_tmaps}
custom_dict = get_custom_objects(list(output_tensormaps.values()))
#model = load_model('ecg_5000_hf_quintuplet_dropout_v2023_04_17.keras')
#output_file = '/output/ecg2hf_quintuplet.csv'
#space_dict = defaultdict(list)

def process_ukb_hd5(filepath, space_dict, model):
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

def process_ge_muse_xml(filepath, space_dict, model):
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
        raise#return
    except Exception as e:
        print(f"Unexpected error processing {filepath}: {e}")
        raise#return

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
    ecg_array = ecg_array[:4096, :]

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



def process_ge_muse_hl7(filepath, space_dict, model):

    def fix_length(ecg_, target_len=4096):
        if ecg_.shape[0] >= target_len:
            return ecg_[:target_len, :]
        pad = np.zeros((target_len - ecg_.shape[0], ecg_.shape[1]), dtype=ecg_.dtype)
        return np.vstack([ecg_, pad])

    with open(filepath, "rt") as ecg:
        line = ecg.readline()
        field4 = line.split("|")[3]

        # --- Step 1: Find CHN line ---
        while field4 != "CHN":
            line = ecg.readline()
            field4 = line.split("|")[3]

        raw_channels = line.split("|")[5].split("~")

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

    # --- Step 3: Convert to array ---
    ecg = np.array(rows, dtype=np.float32)  # shape: (time, leads)

    print("Raw ECG shape:", ecg.shape)  # should be (~4000, 12)

    # --- Step 4: Reorder leads ---
    lead_order = ["I","II","III","aVR","aVL","aVF",
                  "V1","V2","V3","V4","V5","V6"]

    lead_idx = {l: i for i, l in enumerate(channels)}

    # Derive missing leads if needed
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

    # Build final matrix
    final_leads = []
    for l in lead_order:
        if l in lead_idx and lead_idx[l] is not None:
            final_leads.append(ecg[:, lead_idx[l]])
        else:
            # use derived leads
            if l == "III":
                final_leads.append(ecg_III)
            elif l == "aVR":
                final_leads.append(ecg_aVR)
            elif l == "aVF":
                final_leads.append(ecg_aVF)
            elif l == "aVL":
                final_leads.append(ecg_aVL)

    ecg = np.stack(final_leads, axis=1)
    ecg_array = fixed_length(ecg, target_len = 4096)
    ecg_array = ecg_array[:4096, :]





    print(f'Writing row of ECG2AF predictions for hl7 ECG {patient_id}, at {acquisition_date_time}')
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

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--directory",               required=True,
                        help="Root directory containing patient ECG folders")
    parser.add_argument("--model_path",       required=True,
                        help="Path to  Keras model (.keras)")
    parser.add_argument("--output_file",             required=True,
                        help="Output path")
    args = parser.parse_args()

    model = load_model(args.model_path)
    # Iterate over all files in the specified directory
    space_dict = defaultdict(list)
    for root, _, files in os.walk(args.directory):
        for i, filename in enumerate(files):
            filepath = os.path.join(root, filename)
            if os.path.isfile(filepath):
                try:
                    process_ge_muse_xml(filepath, space_dict, model)
                except:
                    try:
                        print(f'trying hl7 for {filepath}')
                        process_ge_muse_hl7(filepath, space_dict, model)
                    except:
                        print(f'skipped altogether for {filepath}')
            # if i > 10000:
            #     break

    df = pd.DataFrame.from_dict(space_dict)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    main()
