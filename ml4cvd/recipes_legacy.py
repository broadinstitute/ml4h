import logging
import datetime
from typing import Any
from dateutil import relativedelta

import numpy as np
from keras.utils import to_categorical

from ml4cvd.defines import DataSetType, CODING_VALUES_MISSING, TENSOR_MAP_GROUP_MISSING_CONTINUOUS, TENSOR_MAP_GROUP_CONTINUOUS
from ml4cvd.defines import EPS, JOIN_CHAR, IMPUTATION_RANDOM, IMPUTATION_MEAN, CODING_VALUES_LESS_THAN_ONE, MRI_SEGMENTED_CHANNEL_MAP
from ml4cvd.defines import MRI_FRAMES, MRI_SEGMENTED, MRI_TO_SEGMENT, MRI_ZOOM_INPUT, MRI_ZOOM_MASK, MRI_ANNOTATION_NAME, MRI_ANNOTATION_CHANNEL_MAP
import ml4cvd.TensorMap
from ml4cvd.utility import str2date

## Legacy
np.set_printoptions(threshold=np.inf)

CONTINUOUS_NEVER_ZERO = ['bike_max_hr', 'bike_resting_hr', 'ecg-bike-max-pred-hr-no0',
                         '25006_Volume-of-grey-matter_2', '25021_Volume-of-amygdala-left_2',
                         '25737_Discrepancy-between-dMRI-brain-image-and-T1-brain-image_2', '25738_Discrepancy-between-SWI-brain-image-and-T1-brain-image_2',
                         '25739_Discrepancy-between-rfMRI-brain-image-and-T1-brain-image_2', '25740_Discrepancy-between-tfMRI-brain-image-and-T1-brain-image_2',
                         '25736_Discrepancy-between-T2-FLAIR-brain-image-and-T1-brain-image_2',
                         ]

CONTINUOUS_WITH_CATEGORICAL_ANSWERS = ['92_Operation-yearage-first-occurred_0_0', '1807_Fathers-age-at-death_0_0',
                                       '130_Place-of-birth-in-UK--east-coordinate_0_0',
                                       '87_Noncancer-illness-yearage-first-occurred_0_0',
                                       '1883_Number-of-full-sisters_0_0', '2966_Age-high-blood-pressure-diagnosed_0_0',
                                       '129_Place-of-birth-in-UK--north-coordinate_0_0',
                                       '1070_Time-spent-watching-television-TV_0_0', '1438_Bread-intake_0_0',
                                       '3526_Mothers-age-at-death_0_0',
                                       '2217_Age-started-wearing-glasses-or-contact-lenses_0_0', '1488_Tea-intake_0_0',
                                       '1060_Time-spent-outdoors-in-winter_0_0', '1528_Water-intake_0_0',
                                       '874_Duration-of-walks_0_0', '894_Duration-of-moderate-activity_0_0',
                                       '1458_Cereal-intake_0_0',
                                       '884_Number-of-daysweek-of-moderate-physical-activity-10-minutes_0_0',
                                       '1873_Number-of-full-brothers_0_0', '1845_Mothers-age_0_0',
                                       '1090_Time-spent-driving_0_0', '1289_Cooked-vegetable-intake_0_0',
                                       '3809_Time-since-last-prostate-specific-antigen-PSA-test_0_0',
                                       '1568_Average-weekly-red-wine-intake_0_0', '2897_Age-stopped-smoking_0_0',
                                       '864_Number-of-daysweek-walked-10-minutes_0_0',
                                       '1588_Average-weekly-beer-plus-cider-intake_0_0',
                                       '2355_Most-recent-bowel-cancer-screening_0_0', '2976_Age-diabetes-diagnosed_0_0',
                                       '3761_Age-hay-fever-rhinitis-or-eczema-diagnosed_0_0',
                                       '3786_Age-asthma-diagnosed_0_0',
                                       '1578_Average-weekly-champagne-plus-white-wine-intake_0_0',
                                       '1598_Average-weekly-spirits-intake_0_0',
                                       '1608_Average-weekly-fortified-wine-intake_0_0',
                                       '1299_Salad--raw-vegetable-intake_0_0', '1309_Fresh-fruit-intake_0_0',
                                       '1319_Dried-fruit-intake_0_0', '3680_Age-when-last-ate-meat_0_0',
                                       '914_Duration-of-vigorous-activity_0_0',
                                       '1050_Time-spend-outdoors-in-summer_0_0', '1737_Childhood-sunburn-occasions_0_0',
                                       '1269_Exposure-to-tobacco-smoke-at-home_0_0',
                                       '2867_Age-started-smoking-in-former-smokers_0_0',
                                       '2887_Number-of-cigarettes-previously-smoked-daily_0_0',
                                       '2926_Number-of-unsuccessful-stopsmoking-attempts_0_0',
                                       '2684_Years-since-last-breast-cancer-screening--mammogram_0_0',
                                       '2734_Number-of-live-births_0_0',
                                       '2804_Age-when-last-used-oral-contraceptive-pill_0_0',
                                       '2824_Age-at-hysterectomy_0_0',
                                       '3536_Age-started-hormonereplacement-therapy-HRT_0_0',
                                       '3546_Age-last-used-hormonereplacement-therapy-HRT_0_0',
                                       '3581_Age-at-menopause-last-menstrual-period_0_0',
                                       '3839_Number-of-spontaneous-miscarriages_0_0',
                                       '2405_Number-of-children-fathered_0_0',
                                            '4022_Age-pulmonary-embolism-blood-clot-in-lung-diagnosed_0_0',
                                  '3992_Age-emphysemachronic-bronchitis-diagnosed_0_0',
                                       '4429_Average-monthly-beer-plus-cider-intake_0_0'
                                       ]

MRI_ANNOTATION_GOOD_NEEDED = ['corrected_extracted_lvesv', 'corrected_extracted_lvedv', 'corrected_extracted_lvef', 'sax_inlinevf_zoom_mask',
                              'cine_segmented_sax_inlinevf_segmented', 'mri_systole_diastole_8_segmented', 'mri_systole_diastole_segmented',
                              'mri_slice_segmented'] # , 'sax_all_diastole_segmented'

MERGED_MAPS = ['mothers_age_0', 'fathers_age_0',]
NOT_MISSING = 'not-missing'

MEAN_IDX = 0
STD_IDX = 1

########################################
# HDF5 load functions for tensors
########################################
def _hdf5_load_categorical_index(tm: "TensorMap", hdf5, dependents={}):
    categorical_data = np.zeros(tm.shape, dtype=np.float32)
    if tm.name in hdf5:
        index = int(hdf5[tm.name][0])
        categorical_data[index] = 1.0
    elif tm.name in hdf5['categorical']:
        index = int(hdf5['categorical'][tm.name][0])
        categorical_data[index] = 1.0
    else:
        raise ValueError(f"No categorical index found for tensor map: {tm.name}.")
    return categorical_data

def _hdf5_load_categorical_flag(tm: "TensorMap", hdf5, dependents={}):
    categorical_data = np.zeros(tm.shape, dtype=np.float32)
    index = 0
    if tm.name in hdf5 and int(hdf5[tm.name][0]) != 0:
        index = 1
    elif tm.name in hdf5['categorical'] and int(hdf5['categorical'][tm.name][0]) != 0:
        index = 1
    categorical_data[index] = 1.0
    return categorical_data

def _hdf5_load_categorical_date(tm: "TensorMap", hdf5, dependents={}):
    categorical_data = np.zeros(tm.shape, dtype=np.float32)
    if tm.name in hdf5:
        index = int(hdf5[tm.name][0])
    elif tm.name in hdf5['categorical']:
        index = int(hdf5['categorical'][tm.name][0])
    else:
        index = 0  # Assume no disease if the tensor does not have the dataset
    if index != 0:
        if tm.name + '_date' in hdf5:
            disease_date = str2date(str(hdf5[tm.name + '_date'][0]))
            assess_date = str2date(str(hdf5['assessment-date_0_0'][0]))
        elif tm.name + '_date' in hdf5['dates']:
            disease_date = str2date(str(hdf5['dates'][tm.name + '_date'][0]))
            assess_date = str2date(str(hdf5['dates']['enroll_date'][0]))
        else:
            raise ValueError(f"No date found for tensor map: {tm.name}.")
        index = 1 if disease_date < assess_date else 2
    categorical_data[index] = 1.0
    return categorical_data

def _hdf5_load_diagnosis_time(tm: "TensorMap", hdf5, dependents={}):
    time_data = np.zeros((1,), dtype=np.float32)
    disease_status = int(hdf5[tm.name][0])
    assess_date = str2date(str(hdf5['assessment-date_0_0'][0]))
    disease_date = str2date(str(hdf5[tm.name + '_date'][0]))
    delta = relativedelta.relativedelta(disease_date, assess_date)
    difference = (delta.years * 12) + delta.months
    if disease_status == 0 or difference == 0:
        raise ValueError('Ignoring healthy people in diagnosis time.')
    time_data[0] = difference
    return time_data

def _hdf5_load_mri_segment(tm: "TensorMap", hdf5, dependents={}):
    mask_group = MRI_SEGMENTED if tm.name == MRI_TO_SEGMENT else MRI_ZOOM_MASK
    slice_idx = np.random.choice(list(hdf5[tm.name].keys()))
    angle_idx = int(slice_idx) // MRI_FRAMES
    tensor = np.zeros(tm.shape, dtype=np.float32)
    dependents[tm.dependent_map] = np.zeros(tm.dependent_map.shape, dtype=np.float32)
    for i in range(tm.shape[-2]):
        cur_slice = str((angle_idx * MRI_FRAMES) + i + 1)  # Instance Number off by 1
        tensor[:, :, i, 0] = np.array(hdf5[tm.name].get(cur_slice), dtype=np.float32)
        label_tensor = np.array(hdf5[mask_group].get(cur_slice), dtype=np.float32)
        dependents[tm.dependent_map][:, :, i, :] = to_categorical(label_tensor, tm.dependent_map.shape[-1])
    return tm.normalize_and_validate(tensor)
    
def _hdf5_load_aligned_distance(tm: "TensorMap", hdf5, dependents={}):
    return np.zeros((1,), dtype=np.float32)

def _hdf5_load_mri_slice(tm: "TensorMap", hdf5, dependents={}):
    cur_slice = np.random.choice(list(hdf5[MRI_TO_SEGMENT].keys()))
    tensor = np.zeros(tm.shape, dtype=np.float32)
    dependents[tm.dependent_map] = np.zeros(tm.dependent_map.shape, dtype=np.float32)
    tensor[:, :, 0] = np.array(hdf5[MRI_TO_SEGMENT].get(cur_slice), dtype=np.float32)
    label_tensor = np.array(hdf5[MRI_SEGMENTED].get(cur_slice), dtype=np.float32)
    dependents[tm.dependent_map][:, :, :] = to_categorical(label_tensor, tm.dependent_map.shape[-1])
    return tm.normalize_and_validate(tensor)
    
def _hdf5_load_sax_zoom_blackout(tm: "TensorMap", hdf5, dependents={}):
    mask_group = MRI_ZOOM_MASK
    slice_idx = np.random.choice(list(hdf5[MRI_ZOOM_INPUT].keys()))
    angle_idx = int(slice_idx) // MRI_FRAMES
    tensor = np.zeros(tm.shape, dtype=np.float32)
    dependents[tm.dependent_map] = np.zeros(tm.dependent_map.shape, dtype=np.float32)
    for i in range(tm.shape[-2]):
        cur_slice = str((angle_idx * MRI_FRAMES) + i + 1)  # Instance Number off by 1
        tensor[:, :, i, 0] = np.array(hdf5[MRI_ZOOM_INPUT].get(cur_slice), dtype=np.float32)
        label_tensor = np.array(hdf5[mask_group].get(cur_slice), dtype=np.float32)
        dependents[tm.dependent_map][:, :, i, :] = to_categorical(label_tensor, tm.dependent_map.shape[-1])
        tensor[:, :, i, 0] *= np.not_equal(label_tensor, 0, dtype=np.float32)
    return tm.normalize_and_validate(tensor)

def _hdf5_load_cine_segmented_sax(tm: "TensorMap", hdf5, dependents={}):
    mask_group = MRI_SEGMENTED
    slice_idx = np.random.choice(list(hdf5[MRI_TO_SEGMENT].keys()))
    angle_idx = int(slice_idx) // MRI_FRAMES
    tensor = np.zeros(tm.shape, dtype=np.float32)
    dependents[tm.dependent_map] = np.zeros(tm.dependent_map.shape, dtype=np.float32)
    for i in range(tm.shape[-2]):
        cur_slice = str((angle_idx * MRI_FRAMES) + i + 1)  # Instance Number off by 1
        tensor[:, :, i, 0] = np.array(hdf5[MRI_TO_SEGMENT].get(cur_slice), dtype=np.float32)
        label_tensor = np.array(hdf5[mask_group].get(cur_slice), dtype=np.float32)
        dependents[tm.dependent_map][:, :, i, :] = to_categorical(label_tensor, tm.dependent_map.shape[-1])
        tensor[:, :, i, 0] *= np.not_equal(label_tensor, 0, dtype=np.float32)
    return tm.normalize_and_validate(tensor)
    
def _hdf5_load_lms_optimised(tm: "TensorMap", hdf5, dependents={}):
    whole_liver = np.array(hdf5['lms_ideal_optimised_low_flip_6dyn'])
    cur_index = np.random.randint(whole_liver.shape[2] - tm.shape[2])
    tensor = whole_liver[:, :, cur_index:cur_index + 4]
    return tm.normalize_and_validate(tensor)
    
def _hdf5_load_mri_systole_diastole(tm: "TensorMap", hdf5, dependents={}):
    if tm.hdf5_override is not None:
        b_number = 'b' + str(np.random.choice(tm.hdf5_override))
        diastole_slice = 'diastole_frame_' + b_number
    else:
        frames = [str(frame) for frame in hdf5.keys() if 'diastole_frame_' in str(frame)]
        if len(frames) == 0:
            raise ValueError('No diastole frames found.')
        diastole_slice = np.random.choice(frames)
        b_number = diastole_slice.split('_')[-1]  # (e.g b1, b2, b3, ... b12 ish)
    tensor = np.zeros(tm.shape, dtype=np.float32)
    dependents[tm.dependent_map] = np.zeros(tm.dependent_map.shape, dtype=np.float32)
    tensor[:, :, 0, 0] = np.array(hdf5[diastole_slice], dtype=np.float32)
    dependents[tm.dependent_map][:, :, 0, :] = to_categorical(np.array(hdf5['diastole_mask_' + b_number]), tm.dependent_map.shape[-1])
    tensor[:, :, 1, 0] = np.array(hdf5['systole_frame_' + b_number], dtype=np.float32)
    dependents[tm.dependent_map][:, :, 1, :] = to_categorical(np.array(hdf5['systole_mask_' + b_number]), tm.dependent_map.shape[-1])
    return tm.normalize_and_validate(tensor)
    
def _hdf5_load_mri_systole_diastole_8(tm: "TensorMap", hdf5, dependents={}):
    tensor = np.zeros(tm.shape, dtype=np.float32)
    dependents[tm.dependent_map] = np.zeros(tm.dependent_map.shape, dtype=np.float32)
    tensor[:, :, 0, 0] = np.array(hdf5['diastole_frame_b2'], dtype=np.float32)
    dependents[tm.dependent_map][:, :, 0, :] = to_categorical(np.array(hdf5['diastole_mask_b2']), tm.dependent_map.shape[-1])
    tensor[:, :, 1, 0] = np.array(hdf5['systole_frame_b2'], dtype=np.float32)
    dependents[tm.dependent_map][:, :, 1, :] = to_categorical(np.array(hdf5['systole_mask_b2']), tm.dependent_map.shape[-1])
    tensor[:, :, 2, 0] = np.array(hdf5['diastole_frame_b4'], dtype=np.float32)
    dependents[tm.dependent_map][:, :, 2, :] = to_categorical(np.array(hdf5['diastole_mask_b4']), tm.dependent_map.shape[-1])
    tensor[:, :, 3, 0] = np.array(hdf5['systole_frame_b4'], dtype=np.float32)
    dependents[tm.dependent_map][:, :, 3, :] = to_categorical(np.array(hdf5['systole_mask_b4']), tm.dependent_map.shape[-1])
    tensor[:, :, 4, 0] = np.array(hdf5['diastole_frame_b6'], dtype=np.float32)
    dependents[tm.dependent_map][:, :, 4, :] = to_categorical(np.array(hdf5['diastole_mask_b6']), tm.dependent_map.shape[-1])
    tensor[:, :, 5, 0] = np.array(hdf5['systole_frame_b6'], dtype=np.float32)
    dependents[tm.dependent_map][:, :, 5, :] = to_categorical(np.array(hdf5['systole_mask_b6']), tm.dependent_map.shape[-1])
    tensor[:, :, 6, 0] = np.array(hdf5['diastole_frame_b8'], dtype=np.float32)
    dependents[tm.dependent_map][:, :, 6, :] = to_categorical(np.array(hdf5['diastole_mask_b8']), tm.dependent_map.shape[-1])
    tensor[:, :, 7, 0] = np.array(hdf5['systole_frame_b8'], dtype=np.float32)
    dependents[tm.dependent_map][:, :, 7, :] = to_categorical(np.array(hdf5['systole_mask_b8']), tm.dependent_map.shape[-1])
    return tm.normalize_and_validate(tensor)

def _hdf5_load_sax_all_diastole(tm: "TensorMap", hdf5, dependents={}):
    missing = 0
    tensor = np.zeros(tm.shape, dtype=np.float32)
    dependents[tm.dependent_map] = np.zeros(tm.dependent_map.shape, dtype=np.float32)
    for b in range(tm.shape[-2]):
        try:
            tensor[:, :, b, 0] = np.array(hdf5[f'diastole_frame_b{b}'], dtype=np.float32)
            dependents[tm.dependent_map][:, :, b, :] = to_categorical(np.array(hdf5[f'diastole_mask_b{b}']), tm.dependent_map.shape[-1])
        except KeyError:
            missing += 1
            tensor[:, :, b, 0] = 0
            dependents[tm.dependent_map][:, :, b, MRI_SEGMENTED_CHANNEL_MAP['background']] = 1
    if missing == tm.shape[-2]:
        raise ValueError(f'Could not find any slices in {tm.name} was hoping for {tm.shape[-2]}')
    return tm.normalize_and_validate(tensor)
    
def _hdf5_load_root_array(tm: "TensorMap", hdf5, dependents={}):
    tensor = np.zeros(tm.shape, dtype=np.float32)
    tensor[:] = np.array(hdf5[tm.name], dtype=np.float32)
    return tm.normalize_and_validate(tensor)
    
def _hdf5_load_mri_annotation_needed(tm: "TensorMap", hdf5, dependents={}):
    continuous_data = np.zeros(tm.shape, dtype=np.float32)  # Automatic left ventricular analysis with InlineVF
    if MRI_ANNOTATION_NAME in hdf5['categorical'] and hdf5['categorical'][MRI_ANNOTATION_NAME][0] != MRI_ANNOTATION_CHANNEL_MAP['good']:
        raise ValueError('MRI Critic annotation not good or unreviewed.')
    continuous_data[0] = float(hdf5['continuous'][tm.name][0])
    if continuous_data[0] == 0 and (tm.sentinel == None and tm.name in CONTINUOUS_NEVER_ZERO):
        raise ValueError(tm.name + ' is a continuous value that cannot be set to 0, but no value was found.')
    if continuous_data[0] == 0 and tm.sentinel is not None:
        continuous_data[:] = tm.sentinel
        return continuous_data
    return tm.normalize_and_validate(continuous_data)
    
def _hdf5_load_ecg_coarse(tm: "TensorMap", hdf5, dependents={}):
    categorical_data = np.zeros(tm.shape, dtype=np.float32)
    if 'poor_data_quality' in hdf5['categorical']:
        raise ValueError('Poor data skipped by ecg_coarse.')
    ecg_interpretation = str(hdf5['ecg_rest_text'][0])
    for afib in ['Atrial fibrillation']:
        if afib in ecg_interpretation:
            categorical_data[tm.channel_map['Atrial_fibrillation']] = 1.0
            return categorical_data
    for rhythm in ['sinus', 'Sinus']:
        if rhythm in ecg_interpretation:
            categorical_data[tm.channel_map['Sinus_rhythm']] = 1.0
            return categorical_data
    categorical_data[tm.channel_map['Other_rhythm']] = 1.0
    return categorical_data
    
def _hdf5_load_ecg_semi_coarse(tm: "TensorMap", hdf5, dependents={}):
    categorical_data = np.zeros(tm.shape, dtype=np.float32)
    if 'poor_data_quality' in hdf5['categorical']:
        raise ValueError('Poor data skipped by ecg_coarse.')
    ecg_interpretation = str(hdf5['ecg_rest_text'][0])
    for channel in tm.channel_map:
        if channel in hdf5['categorical']:
            categorical_data[tm.channel_map[channel]] = 1.0
            return categorical_data
    for afib in ['Atrial fibrillation']:
        if afib in ecg_interpretation:
            categorical_data[tm.channel_map['Atrial_fibrillation']] = 1.0
            return categorical_data
    for rhythm in ['sinus', 'Sinus']:
        if rhythm in ecg_interpretation:
            categorical_data[tm.channel_map['Other_sinus_rhythm']] = 1.0
            return categorical_data
    categorical_data[tm.channel_map['Other_rhythm']] = 1.0
    return categorical_data
    
def _hdf5_load_ecg_semi_coarse_with_poor(tm: "TensorMap", hdf5, dependents={}):
    categorical_data = np.zeros(tm.shape, dtype=np.float32)
    ecg_interpretation = str(hdf5['ecg_rest_text'][0])
    for channel in tm.channel_map:
        if channel in hdf5['categorical']:
            categorical_data[tm.channel_map[channel]] = 1.0
            return categorical_data
    for afib in ['Atrial fibrillation']:
        if afib in ecg_interpretation:
            categorical_data[tm.channel_map['Atrial_fibrillation']] = 1.0
            return categorical_data
    for rhythm in ['sinus', 'Sinus']:
        if rhythm in ecg_interpretation:
            categorical_data[tm.channel_map['Other_sinus_rhythm']] = 1.0
            return categorical_data
    categorical_data[tm.channel_map['Other_rhythm']] = 1.0
    return categorical_data
    
def _hdf5_load_ecg_categoical_interpretation(tm: "TensorMap", hdf5, dependents={}):
    categorical_data = np.zeros(tm.shape, dtype=np.float32)
    for channel in tm.channel_map:
        if channel in str(hdf5['ecg_rest_text'][0]):
            categorical_data[tm.channel_map[channel]] = 1.0
            return categorical_data
    if 'no_' + tm.name in tm.channel_map:
        categorical_data[tm.channel_map['no_' + tm.name]] = 1.0
        return categorical_data
    else:
        raise ValueError(f"ECG categorical interpretation could not find any of these keys: {tm.channel_map.keys()}")
    
def _hdf5_load_categorical(tm: "TensorMap", hdf5, dependents={}):
    categorical_data = np.zeros(tm.shape, dtype=np.float32)
    for channel in tm.channel_map:
        if channel in hdf5['categorical']:
            categorical_data[tm.channel_map[channel]] = 1.0
    return categorical_data

def _hdf5_load_merged_maps(tm: "TensorMap", hdf5, dependents={}):
    return tm._merged_tensor_from_file(hdf5)

def _hdf5_load_continuous(tm: "TensorMap", hdf5, dependents={}):
    continuous_data = np.zeros(tm.shape, dtype=np.float32)
    if tm.name in hdf5:
        if hasattr(hdf5[tm.name], "__shape__"):
            continuous_data[0] = hdf5[tm.name][0]
        else:
            continuous_data[0] = hdf5[tm.name][()]
    missing = True
    for k in tm.channel_map:
        if k in hdf5[tm.group]:
            value = hdf5[tm.group][k][0]
            missing = False
            if k in CONTINUOUS_WITH_CATEGORICAL_ANSWERS:
                if value in CODING_VALUES_LESS_THAN_ONE:
                    value = .5
                if value in CODING_VALUES_MISSING:
                    # need to set missing values to 0 so normalization works
                    value = 0
                    missing = True
            continuous_data[tm.channel_map[k]] = value
    if NOT_MISSING in tm.channel_map and not missing:
        continuous_data[tm.channel_map[NOT_MISSING]] = 1
    if continuous_data[0] == 0 and (tm.sentinel is None and tm.name in CONTINUOUS_NEVER_ZERO):
        raise ValueError(tm.name + ' is a continuous value that cannot be set to 0, but no value was found.')
    return tm.normalize_and_validate(continuous_data)
    
def _hdf5_load_multi_field_continuous(tm: "TensorMap", hdf5, dependents={}):
    if tm.is_multi_field_continuous_with_missing_channel():
        multiplier = 2
    else:
        multiplier = 1
    missing_array = [False] * len(tm.channel_map)
    continuous_data = np.zeros(tm.shape, dtype=np.float32)
    for k in tm.channel_map:
        missing = True
        if k in hdf5['continuous']:
            value = hdf5['continuous'][k][0]
            missing = False
            if tm.name in CONTINUOUS_WITH_CATEGORICAL_ANSWERS:
                if value in CODING_VALUES_LESS_THAN_ONE:
                    value = .5
                if value in CODING_VALUES_MISSING:
                    # need to set missing values to 0 so normalization works
                    value = 0
                    missing = True
            # Put value at index k (times 2 to make space for the not-missing channels), and put whether or not
            # this value is not missing in the following element.
            continuous_data[tm.channel_map[k] * multiplier] = value
        if tm.is_multi_field_continuous_with_missing_channel():
            continuous_data[tm.channel_map[k] * multiplier + 1] = not missing
        else:
            missing_array[tm.channel_map[k]] = missing
    if tm.is_multi_field_continuous_with_missing_channel():
        return tm.normalize_multi_field_continuous(continuous_data)
    else:
        return tm.normalize_multi_field_continuous_no_missing_channels(continuous_data, missing_array)
        
def _hdf5_load_ecg_bike(tm: "TensorMap", hdf5, dependents={}):
    tensor = np.array(hdf5[tm.group][tm.name], dtype=np.float32)
    return tm.normalize_and_validate(tensor)
    
def _hdf5_load_ecg_bike_recovery(tm: "TensorMap", hdf5, dependents={}):
    tensor = np.zeros(tm.shape)
    for channel, idx in tm.channel_map.items():
        tensor[:, idx] = hdf5[tm.group][channel]
    return tm.normalize_and_validate(tensor)

def _hdf5_load_ecg_text(tm: "TensorMap", hdf5, dependents={}):
    tensor = np.zeros(tm.shape, dtype=np.float32)
    dependents[tm.dependent_map] = np.zeros(tm.dependent_map.shape, dtype=np.float32)
    caption = str(hdf5[tm.name][0]).strip()
    char_idx = np.random.randint(len(caption) + 1)
    if char_idx == len(caption):
        next_char = '!'
    else:
        next_char = caption[char_idx]
    dependents[tm.dependent_map][tm.dependent_map.channel_map[next_char]] = 1.0
    window_offset = max(0, tm.shape[0] - char_idx)

    for k in range(max(0, char_idx - tm.shape[0]), char_idx):
        tensor[window_offset, tm.dependent_map.channel_map[caption[k]]] = 1.0
        window_offset += 1
    return tensor
    
def _hdf5_load_hidden_layer(tm: "TensorMap", hdf5, dependents={}):
    input_dict = {}
    for input_tm in tm.required_inputs:
        input_dict[input_tm.input_name()] = np.expand_dims(input_tm.tensor_from_file(input_tm, hdf5), axis=0)
    return tm.model.predict(input_dict)
    
def _hdf5_load_dependent_map(tm: "TensorMap", hdf5, dependents={}):
    dataset_key = np.random.choice(list(tm.dependent_map.channel_map.keys()))
    one_hot = np.zeros(tm.dependent_map.shape)
    one_hot[tm.dependent_map.channel_map[dataset_key]] = 1.0
    dependents[tm.dependent_map] = one_hot
    tensor = np.array(hdf5.get(dataset_key), dtype=np.float32)
    return tm.normalize_and_validate(tensor)

def _hdf5_load_channel_map_no_dependent(tm: "TensorMap", hdf5, dependents={}):
    return np.array(hdf5.get(tm.name), dtype=np.float32)

def hdf5_load_tensor(tm, hdf5, dependents={}):
    """Reconstruct a tensor from an hdf5 file

    Arguments
        tm: The TensorMap that describes the type of tensor to make
        hdf5: The file where the tensor was saved
        dependents: A dict that maps dependent TensorMaps to numpy arrays
            if self has a dependent TensorMap it will be constructed and added here

    Returns
        A numpy array whose dimension and type is dictated by tm
    """
    if tm.is_categorical_index():
        return _hdf5_load_categorical_index(tm, hdf5, dependents)
    elif tm.is_categorical_flag():
        return _hdf5_load_categorical_flag(tm, hdf5, dependents)
    elif tm.is_categorical_date():
        return _hdf5_load_categorical_date(tm, hdf5, dependents)
    elif tm.is_diagnosis_time():
        return _hdf5_load_diagnosis_time(tm, hdf5, dependents)
    elif tm.name in [MRI_TO_SEGMENT, MRI_ZOOM_INPUT]:
        return _hdf5_load_mri_segment(tm, hdf5, dependents)
    elif tm.name == 'aligned_distance':
        return _hdf5_load_aligned_distance(tm, hdf5, dependents)
    elif tm.name == 'lms_ideal_optimised_low_flip_6dyn_4slice':
        return _hdf5_load_lms_optimised(tm, hdf5, dependents)
    elif tm.name == 'mri_slice':
        return _hdf5_load_mri_slice(tm, hdf5, dependents)
    elif tm.name == 'sax_inlinevf_zoom_blackout':
        return _hdf5_load_sax_zoom_blackout(tm, hdf5, dependents)
    elif tm.name == 'cine_segmented_sax_inlinevf_blackout':
        return _hdf5_load_cine_segmented_sax(tm, hdf5, dependents)
    elif tm.name == 'mri_systole_diastole':
        return _hdf5_load_mri_systole_diastole(tm, hdf5, dependents)
    elif tm.name == 'mri_systole_diastole_8':
        return _hdf5_load_mri_systole_diastole_8(tm, hdf5, dependents)
    elif tm.name == 'sax_all_diastole':
        return _hdf5_load_sax_all_diastole(tm, hdf5, dependents)
    elif tm.is_root_array():
        return _hdf5_load_root_array(tm, hdf5, dependents)
    elif tm.name in MRI_ANNOTATION_GOOD_NEEDED:
        return _hdf5_load_mri_annotation_needed(tm, hdf5, dependents)
    elif tm.name == 'ecg_coarse':
        return _hdf5_load_ecg_coarse(tm, hdf5, dependents)
    elif tm.name == 'ecg_semi_coarse':
        return _hdf5_load_ecg_semi_coarse(tm, hdf5, dependents)
    elif tm.name == 'ecg_semi_coarse_with_poor':
        return _hdf5_load_ecg_semi_coarse_with_poor(tm, hdf5, dependents)
    elif tm.is_ecg_categorical_interpretation():
        return _hdf5_load_ecg_categoical_interpretation(tm, hdf5, dependents)
    elif tm.is_categorical() and tm.channel_map is not None:
        return _hdf5_load_categorical(tm, hdf5, dependents)
    elif tm.name in MERGED_MAPS:
        return _hdf5_load_merged_maps(tm, hdf5, dependents)
    elif tm.is_continuous():
        return _hdf5_load_continuous(tm, hdf5, dependents)
    elif tm.is_multi_field_continuous():
        return _hdf5_load_multi_field_continuous(tm, hdf5, dependents)
    elif tm.group == "ecg_bike":
        return _hdf5_load_ecg_bike(tm, hdf5, dependents)
    elif tm.group == "ecg_bike_recovery":
        return _hdf5_load_ecg_bike_recovery(tm, hdf5, dependents)
    elif tm.group == 'ecg_text':
        return _hdf5_load_ecg_text(tm, hdf5, dependents)
    elif tm.is_hidden_layer():
        return _hdf5_load_hidden_layer(tm, hdf5, dependents)
    elif tm.dependent_map is not None:  # Assumes dependent maps are 1-hot categoricals
        return _hdf5_load_dependent_map(tm, hdf5, dependents)
    elif tm.channel_map is None and tm.dependent_map is None:
        return _hdf5_load_channel_map_no_dependent(tm, hdf5, dependents)
