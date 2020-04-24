import os
import csv
import datetime
from typing import List, Dict, Tuple, Callable, Union, Optional

import vtk
import h5py
import biosppy
import logging
import scipy
import numpy as np
import pandas as pd
import vtk.util.numpy_support
from scipy.stats import linregress
from tensorflow.keras.utils import to_categorical

from ml4cvd.normalizer import Standardize, ZeroMeanStd1
from ml4cvd.metrics import weighted_crossentropy, pearson
from ml4cvd.tensor_writer_ukbb import tensor_path, first_dataset_at_path
from ml4cvd.TensorMap import TensorMap, no_nans, str2date, make_range_validator, Interpretation
from ml4cvd.defines import ECG_REST_LEADS, ECG_REST_MEDIAN_LEADS, ECG_REST_AMP_LEADS, ECG_SEGMENTED_CHANNEL_MAP
from ml4cvd.defines import StorageType, MRI_TO_SEGMENT, MRI_SEGMENTED, MRI_LAX_SEGMENTED, MRI_SEGMENTED_CHANNEL_MAP, MRI_FRAMES
from ml4cvd.defines import MRI_PIXEL_WIDTH, MRI_PIXEL_HEIGHT, MRI_SLICE_THICKNESS, MRI_PATIENT_ORIENTATION, MRI_PATIENT_POSITION
from ml4cvd.defines import MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP, MRI_SAX_SEGMENTED_CHANNEL_MAP, MRI_AO_SEGMENTED_CHANNEL_MAP


def normalized_first_date(tm: TensorMap, hd5: h5py.File, dependents=None):
    tensor = _get_tensor_at_first_date(hd5, tm.path_prefix, tm.name)
    if tm.axes() > 1:
        return _pad_or_crop_array_to_shape(tm.shape, tensor)
    else:
        return tensor


def _random_slice_tensor(tensor_key, dependent_key=None):
    def _random_slice_tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        big_tensor = _get_tensor_at_first_date(hd5, tm.path_prefix, tensor_key)
        cur_slice = np.random.choice(range(big_tensor.shape[-1]))
        tensor = np.zeros(tm.shape, dtype=np.float32)
        tensor[..., 0] = big_tensor[..., cur_slice]
        if dependent_key is not None:
            dependents[tm.dependent_map] = np.zeros(tm.dependent_map.shape, dtype=np.float32)
            label_tensor = np.array(hd5[dependent_key][..., cur_slice], dtype=np.float32)
            dependents[tm.dependent_map][:, :, :] = to_categorical(label_tensor, tm.dependent_map.shape[-1])
        return tensor
    return _random_slice_tensor_from_file


def _slice_subset_tensor(tensor_key, start, stop, step=1, dependent_key=None, pad_shape=None, dtype_override=None, allow_channels=True, flip_swap=False, swap_axes=-1):
    def _slice_subset_tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        if dtype_override is not None:
            big_tensor = _get_tensor_at_first_date(hd5, tm.path_prefix, tensor_key)
        else:
            big_tensor = _get_tensor_at_first_date(hd5, tm.path_prefix, tensor_key)

        if flip_swap:
            big_tensor = np.flip(np.swapaxes(big_tensor, 0, swap_axes))

        if pad_shape is not None:
            big_tensor = _pad_or_crop_array_to_shape(pad_shape, big_tensor)

        if allow_channels and tm.shape[-1] < (stop-start) // step:
            tensor = big_tensor[..., np.arange(start, stop, step), :]
        else:
            tensor = big_tensor[..., np.arange(start, stop, step)]

        if dependent_key is not None:
            label_tensor = np.array(hd5[dependent_key][..., start:stop], dtype=np.float32)
            dependents[tm.dependent_map] = to_categorical(label_tensor, tm.dependent_map.shape[-1])
        return tensor
    return _slice_subset_tensor_from_file


def _build_tensor_from_file(file_name: str, target_column: str, normalization: bool = False, delimiter: str = '\t'):
    """
    Build a tensor_from_file function from a column in a file.
    Only works for continuous values.
    When normalization is True values will be normalized according to the mean and std of all of the values in the column.
    """
    error = None
    try:
        with open(file_name, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader)
            index = header.index(target_column)
            table = {}
            fail_count = 0
            for row in reader:
                try:
                    table[row[0]] = np.array([float(row[index])])
                except ValueError:
                    fail_count += 1
                    continue
            logging.info(f'{fail_count} samples failed out of {len(table) + fail_count} in {file_name}.')
            if normalization:
                value_array = np.array([sub_array[0] for sub_array in table.values()])
                mean = value_array.mean()
                std = value_array.std()
                logging.info(f'Normalizing TensorMap from file {file_name}, column {target_column} with mean: '
                             f'{mean:.2f}, std: {std:.2f}')
    except FileNotFoundError as e:
        error = e

    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        if error:
            raise error
        if normalization:
            tm.normalization = {'mean': mean, 'std': std}
        try:
            return table[os.path.basename(hd5.filename).replace('.hd5', '')].copy()
        except KeyError:
            raise KeyError(f'User id not in file {file_name}.')
    return tensor_from_file


def _build_tensor_from_sample_id_file(file_name: str, id_column: str, delimiter: str):
    """
    Build a binary categorical tensor_from_file function based on inclusion in provided file.
    """
    error = None
    try:
        with open(file_name, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader)
            index = header.index(id_column)
            table = {row[index] for row in reader}
    except FileNotFoundError as e:
        error = e

    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        if error:
            raise error
        sample_id = os.path.basename(hd5.filename).replace('.hd5', '')
        return (np.array([1, 0]) if sample_id in table else np.array([0, 1]))
    return tensor_from_file


def _survival_tensor(start_date_key, day_window):
    def _survival_tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        assess_date = str2date(str(hd5[start_date_key][0]))
        has_disease = 0   # Assume no disease if the tensor does not have the dataset
        if tm.name in hd5['categorical']:
            has_disease = int(hd5['categorical'][tm.name][0])

        if tm.name + '_date' in hd5['dates']:
            censor_date = str2date(str(hd5['dates'][tm.name + '_date'][0]))
        elif 'phenotype_censor' in hd5['dates']:
            censor_date = str2date(str(hd5['dates/phenotype_censor']))
        else:
            raise ValueError(f'No date found for survival {tm.name}')

        intervals = int(tm.shape[0] / 2)
        days_per_interval = day_window / intervals
        survival_then_censor = np.zeros(tm.shape, dtype=np.float32)
        for i, day_delta in enumerate(np.arange(0, day_window, days_per_interval)):
            cur_date = assess_date + datetime.timedelta(days=day_delta)
            survival_then_censor[i] = float(cur_date < censor_date)
            survival_then_censor[intervals+i] = has_disease * float(censor_date <= cur_date < censor_date + datetime.timedelta(days=days_per_interval))
            if i == 0 and censor_date <= cur_date:  # Handle prevalent diseases
                survival_then_censor[intervals] = has_disease
        return survival_then_censor

    return _survival_tensor_from_file


def _age_in_years_tensor(date_key, birth_key='continuous/34_Year-of-birth_0_0'):
    def age_at_tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        assess_date = str2date(str(hd5[date_key][0]))
        birth_year = hd5[birth_key][0]
        return np.array([assess_date.year-birth_year])
    return age_at_tensor_from_file


def prevalent_incident_tensor(start_date_key, event_date_key):
    def _prevalent_incident_tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        index = 0
        categorical_data = np.zeros(tm.shape, dtype=np.float32)
        if tm.hd5_key_guess() in hd5:
            data = tm.hd5_first_dataset_in_group(hd5, tm.hd5_key_guess())
            if tm.storage_type == StorageType.CATEGORICAL_INDEX or tm.storage_type == StorageType.CATEGORICAL_FLAG:
                index = int(data[0])
                categorical_data[index] = 1.0
            else:
                categorical_data = np.array(data)
        elif tm.storage_type == StorageType.CATEGORICAL_FLAG:
            categorical_data[index] = 1.0
        else:
            raise ValueError(f"No HD5 Key at prefix {tm.path_prefix} found for tensor map: {tm.name}.")

        if index != 0:
            if event_date_key in hd5 and start_date_key in hd5:
                disease_date = str2date(str(hd5[event_date_key][0]))
                assess_date = str2date(str(hd5[start_date_key][0]))
            else:
                raise ValueError(f"No date found for tensor map: {tm.name}.")
            index = 1 if disease_date < assess_date else 2
        categorical_data[index] = 1.0
        return categorical_data
    return _prevalent_incident_tensor_from_file


def _all_dates(hd5: h5py.File, path_prefix: str, name: str) -> List[str]:
    """
    Gets the dates in the hd5 with path_prefix, dtype, name.
    """
    # TODO: This ideally would be implemented to not depend on the order of name, date, dtype, path_prefix in the hd5s
    # Unfortunately, that's hard to do efficiently
    return hd5[path_prefix][name]


def _pass_nan(tensor):
    return tensor


def _fail_nan(tensor):
    if np.isnan(tensor).any():
        raise ValueError('Tensor contains nans.')
    return tensor


def _filter_nan(tensor):
    return tensor[np.isfinite(tensor)]


def _nan_to_mean(tensor, max_allowed_nan_fraction=.2):
    tensor_isnan = np.isnan(tensor)
    if np.count_nonzero(tensor_isnan) / tensor.size > max_allowed_nan_fraction:
        raise ValueError('Tensor contains too many nans.')
    tensor[tensor_isnan] = np.nanmean(tensor)
    return tensor


def _get_tensor_at_first_date(hd5: h5py.File, path_prefix: str, name: str, handle_nan=_fail_nan):
    """
    Gets the numpy array at the first date of path_prefix, dtype, name.
    """
    dates = _all_dates(hd5, path_prefix, name)
    if not dates:
        raise ValueError(f'No {name} values values available.')
    tensor = np.array(hd5[f'{tensor_path(path_prefix=path_prefix, name=name)}{min(dates)}/'], dtype=np.float32)
    tensor = handle_nan(tensor)
    return tensor


def _pad_or_crop_array_to_shape(new_shape: Tuple, original: np.ndarray):
    if new_shape == original.shape:
        return original
    result = np.zeros(new_shape)
    slices = tuple(slice(min(original.shape[i], new_shape[i])) for i in range(len(original.shape)))

    # Allow expanding one dimension eg (256, 256) can become (256, 256, 1)
    if len(new_shape) - len(original.shape) == 1:
        padded = result[..., 0]
    else:
        padded = result

    padded[slices] = original[slices]
    return result


TMAPS: Dict[str, TensorMap] = dict()


# TMAPS['ecg_rest_afib_hazard'] = TensorMap('atrial_fibrillation_or_flutter', Interpretation.COX_PROPORTIONAL_HAZARDS, shape=(100,),
#                                           tensor_from_file=_survival_tensor('ecg_rest_date', 365 * 5))
# TMAPS['ecg_rest_cad_hazard'] = TensorMap('coronary_artery_disease',  Interpretation.COX_PROPORTIONAL_HAZARDS, shape=(100,),
#                                          tensor_from_file=_survival_tensor('ecg_rest_date', 365 * 5))
# TMAPS['ecg_rest_hyp_hazard'] = TensorMap('hypertension',  Interpretation.COX_PROPORTIONAL_HAZARDS, shape=(100,),
#                                          tensor_from_file=_survival_tensor('ecg_rest_date', 365 * 5))
# TMAPS['ecg_rest_cad_hazard'] = TensorMap('coronary_artery_disease',  Interpretation.COX_PROPORTIONAL_HAZARDS, shape=(100,),
#                                          tensor_from_file=_survival_tensor('ecg_rest_date', 365 * 5))
# TMAPS['enroll_cad_hazard'] = TensorMap('coronary_artery_disease',  Interpretation.COX_PROPORTIONAL_HAZARDS, shape=(100,),
#                                        tensor_from_file=_survival_tensor('dates/enroll_date', 365 * 10))
# TMAPS['enroll_hyp_hazard'] = TensorMap('hypertension',  Interpretation.COX_PROPORTIONAL_HAZARDS, shape=(100,),
#                                        tensor_from_file=_survival_tensor('dates/enroll_date', 365 * 10))
# TMAPS['enroll_afib_hazard'] = TensorMap('atrial_fibrillation_or_flutter',  Interpretation.COX_PROPORTIONAL_HAZARDS, shape=(100,),
#                                         tensor_from_file=_survival_tensor('dates/enroll_date', 365 * 10))
# TMAPS['enroll_chol_hazard'] = TensorMap('hypercholesterolemia',  Interpretation.COX_PROPORTIONAL_HAZARDS, shape=(100,),
#                                         tensor_from_file=_survival_tensor('dates/enroll_date', 365 * 10))
# TMAPS['enroll_diabetes2_hazard'] = TensorMap('diabetes_type_2',  Interpretation.COX_PROPORTIONAL_HAZARDS, shape=(100,),
#                                              tensor_from_file=_survival_tensor('dates/enroll_date', 365 * 10))


def _warp_ecg(ecg):
    i = np.arange(ecg.shape[0])
    warped = i + (np.random.rand() * 100 * np.sin(i / (500 + np.random.rand() * 100))
                  + np.random.rand() * 100 * np.cos(i / (500 + np.random.rand() * 100)))
    warped_ecg = np.zeros_like(ecg)
    for j in range(ecg.shape[1]):
        warped_ecg[:, j] = np.interp(i, warped, ecg[:, j])
    return warped_ecg


def _make_ecg_rest(population_normalize: float = None, random_roll: bool = False, warp: bool = False, downsample_steps: int = 0,
                   short_time_nperseg: int = 0, short_time_noverlap: int = 0):
    def ecg_rest_from_file(tm, hd5, dependents={}):
        tensor = np.zeros(tm.shape, dtype=np.float32)
        if random_roll:
            roll = np.random.randint(2500)
        if tm.dependent_map is not None:
            dependents[tm.dependent_map] = np.zeros(tm.dependent_map.shape, dtype=np.float32)
            key_choices = [k for k in hd5[tm.path_prefix] if tm.name in k]
            lead_idx = np.random.choice(key_choices)
            tensor = np.reshape(hd5[tm.path_prefix][lead_idx][: tensor.shape[0] * tensor.shape[1]], tensor.shape, order='F')
            dependents[tm.dependent_map][:, 0] = np.array(hd5[tm.path_prefix][lead_idx.replace(tm.name, tm.dependent_map.name)])
            dependents[tm.dependent_map] = tm.zero_mean_std1(dependents[tm.dependent_map])
        else:
            for k in hd5[tm.path_prefix]:
                if k in tm.channel_map:
                    data = tm.hd5_first_dataset_in_group(hd5, f'{tm.path_prefix}/{k}/')
                    if short_time_nperseg > 0 and short_time_noverlap > 0:
                        f, t, short_time_ft = scipy.signal.stft(data, nperseg=short_time_nperseg, noverlap=short_time_noverlap)
                        tensor[..., tm.channel_map[k]] = short_time_ft
                    elif downsample_steps > 1:
                        tensor[:, tm.channel_map[k]] = np.array(data, dtype=np.float32)[::downsample_steps]
                    elif random_roll:
                        tensor[:, tm.channel_map[k]] = np.roll(data, roll)
                    else:
                        tensor[:, tm.channel_map[k]] = data
        if population_normalize:
            tensor /= population_normalize
        if warp:
            tensor = _warp_ecg(tensor)
        return tensor
    return ecg_rest_from_file


TMAPS['ecg_rest_raw'] = TensorMap('ecg_rest_raw', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(population_normalize=2000.0),
                                  channel_map=ECG_REST_LEADS)

TMAPS['ecg_rest_raw_roll'] = TensorMap('ecg_rest_raw', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(population_normalize=2000.0, random_roll=True),
                                       channel_map=ECG_REST_LEADS, cacheable=False)
TMAPS['ecg_rest_raw_warp'] = TensorMap('ecg_rest_raw', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(population_normalize=2000.0, warp=True),
                                       channel_map=ECG_REST_LEADS, cacheable=False)
TMAPS['ecg_rest_raw_warp_n_roll'] = TensorMap('ecg_rest_raw', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(population_normalize=2000.0, random_roll=True, warp=True),
                                              channel_map=ECG_REST_LEADS, cacheable=False)
TMAPS['ecg_rest_raw_100'] = TensorMap('ecg_rest_raw_100', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(population_normalize=100.0),
                                      channel_map=ECG_REST_LEADS)

TMAPS['ecg_rest'] = TensorMap('strip', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
                              channel_map=ECG_REST_LEADS, normalization={'zero_mean_std1': 1.0})
TMAPS['ecg_rest_2500_ukb'] = TensorMap('ecg_rest_2500', Interpretation.CONTINUOUS, shape=(2500, 12), path_prefix='ukb_ecg_rest', channel_map=ECG_REST_LEADS,
                                       tensor_from_file=_make_ecg_rest(downsample_steps=2), normalization={'zero_mean_std1': 1.0})

TMAPS['ecg_rest_stft'] = TensorMap('ecg_rest_stft', Interpretation.CONTINUOUS, shape=(33, 158, 12), path_prefix='ukb_ecg_rest', channel_map=ECG_REST_LEADS,
                                   tensor_from_file=_make_ecg_rest(short_time_nperseg=64, short_time_noverlap=32), normalization={'zero_mean_std1': 1.0})
TMAPS['ecg_rest_stft_512'] = TensorMap('ecg_rest_stft_512', shape=(257, 314, 12), path_prefix='ukb_ecg_rest', channel_map=ECG_REST_LEADS,
                                       tensor_from_file=_make_ecg_rest(short_time_nperseg=512, short_time_noverlap=496), normalization={'zero_mean_std1': 1.0})

TMAPS['ecg_rest'] = TensorMap('strip', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
                              channel_map=ECG_REST_LEADS, normalization={'zero_mean_std1': 1.0})

TMAPS['ecg_rest_stack'] = TensorMap('strip', Interpretation.CONTINUOUS, shape=(600, 12, 8), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
                                    channel_map=ECG_REST_LEADS, normalization={'zero_mean_std1': 1.0})

TMAPS['ecg_rest_median_raw'] = TensorMap('median', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(600, 12), loss='logcosh', activation='linear', tensor_from_file=_make_ecg_rest(population_normalize=2000.0),
                                         metrics=['mse', 'mae', 'logcosh'], channel_map=ECG_REST_MEDIAN_LEADS)

TMAPS['ecg_rest_median'] = TensorMap('median', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(600, 12), loss='logcosh', activation='linear', tensor_from_file=_make_ecg_rest(),
                                     metrics=['mse', 'mae', 'logcosh'], channel_map=ECG_REST_MEDIAN_LEADS, normalization={'zero_mean_std1': 1.0})

TMAPS['ecg_rest_median_stack'] = TensorMap('median', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(600, 12, 1), activation='linear', tensor_from_file=_make_ecg_rest(),
                                           metrics=['mse', 'mae', 'logcosh'], loss='logcosh', loss_weight=1.0,
                                           channel_map=ECG_REST_MEDIAN_LEADS, normalization={'zero_mean_std1': 1.0})

TMAPS['ecg_median_1lead'] = TensorMap('median', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(600, 1), loss='logcosh', loss_weight=10.0, tensor_from_file=_make_ecg_rest(),
                                      activation='linear', metrics=['mse', 'mae', 'logcosh'], channel_map={'lead': 0}, normalization={'zero_mean_std1': 1.0})

TMAPS['ecg_rest_1lead'] = TensorMap('strip', Interpretation.CONTINUOUS, shape=(600, 8), path_prefix='ukb_ecg_rest', channel_map={'lead': 0}, tensor_from_file=_make_ecg_rest(),
                                    dependent_map=TMAPS['ecg_median_1lead'], normalization={'zero_mean_std1': 1.0})


def _get_lead_cm(length):
    lead_cm = {}
    lead_weights = []
    for i in range(length):
        wave_val = i - (length//2)
        lead_cm['w'+str(wave_val).replace('-', '_')] = i
        lead_weights.append((np.abs(wave_val+1)/(length/2)) + 1.0)
    return lead_cm, lead_weights


TMAPS['ecg_median_1lead_categorical'] = TensorMap('median',  Interpretation.CATEGORICAL, shape=(600, 32), activation='softmax', tensor_from_file=_make_ecg_rest(),
                                                  channel_map=_get_lead_cm(32)[0], normalization={'zero_mean_std1': 1.0},
                                                  loss=weighted_crossentropy(np.array(_get_lead_cm(32)[1]), 'ecg_median_categorical'))

TMAPS['ecg_rest_1lead_categorical'] = TensorMap('strip', shape=(600, 8), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
                                                normalization={'zero_mean_std1': 1.0},
                                                channel_map={'window0': 0, 'window1': 1, 'window2': 2, 'window3': 3,
                                                             'window4': 4, 'window5': 5, 'window6': 6, 'window7': 7},
                                                dependent_map=TMAPS['ecg_median_1lead_categorical'])


def _make_rhythm_tensor(skip_poor=True):
    def rhythm_tensor_from_file(tm, hd5, dependents={}):
        categorical_data = np.zeros(tm.shape, dtype=np.float32)
        ecg_interpretation = str(tm.hd5_first_dataset_in_group(hd5, 'ukb_ecg_rest/ecg_rest_text/')[()])
        if skip_poor and 'Poor data quality' in ecg_interpretation:
            raise ValueError(f'Poor data quality skipped by {tm.name}.')
        for channel in tm.channel_map:
            if channel.replace('_', ' ') in ecg_interpretation:
                categorical_data[tm.channel_map[channel]] = 1.0
                return categorical_data
        for rhythm in ['sinus', 'Sinus']:
            if rhythm in ecg_interpretation:
                categorical_data[tm.channel_map['Other_sinus_rhythm']] = 1.0
                return categorical_data
        categorical_data[tm.channel_map['Other_rhythm']] = 1.0
        return categorical_data
    return rhythm_tensor_from_file


TMAPS['ecg_rhythm'] = TensorMap('ecg_rhythm', Interpretation.CATEGORICAL, tensor_from_file=_make_rhythm_tensor(),
                                loss=weighted_crossentropy([1.0, 2.0, 3.0, 3.0, 20.0, 20.0], 'ecg_rhythm'),
                                channel_map={'Normal_sinus_rhythm': 0, 'Sinus_bradycardia': 1, 'Marked_sinus_bradycardia': 2, 'Other_sinus_rhythm': 3, 'Atrial_fibrillation': 4, 'Other_rhythm': 5})
TMAPS['ecg_rhythm_poor'] = TensorMap('ecg_rhythm', Interpretation.CATEGORICAL, tensor_from_file=_make_rhythm_tensor(False),
                                     loss=weighted_crossentropy([1.0, 2.0, 3.0, 3.0, 20.0, 20.0], 'ecg_rhythm_poor'),
                                     channel_map={'Normal_sinus_rhythm': 0, 'Sinus_bradycardia': 1, 'Marked_sinus_bradycardia': 2, 'Other_sinus_rhythm': 3, 'Atrial_fibrillation': 4, 'Other_rhythm': 5})

TMAPS['ecg_rest_age'] = TensorMap('ecg_rest_age', Interpretation.CONTINUOUS, tensor_from_file=_age_in_years_tensor('ecg_rest_date'), loss='logcosh',
                                  channel_map={'ecg_rest_age': 0}, validator=make_range_validator(0, 110), normalization={'mean': 65, 'std': 7.7})


def label_from_ecg_interpretation_text(tm, hd5, dependents={}):
    categorical_data = np.zeros(tm.shape, dtype=np.float32)
    ecg_interpretation = str(tm.hd5_first_dataset_in_group(hd5, 'ukb_ecg_rest/ecg_rest_text/')[()])
    for channel in tm.channel_map:
        if channel in ecg_interpretation:
            categorical_data[tm.channel_map[channel]] = 1.0
            return categorical_data
    if 'no_' + tm.name in tm.channel_map:
        categorical_data[tm.channel_map['no_' + tm.name]] = 1.0
        return categorical_data
    else:
        raise ValueError(f"ECG categorical interpretation could not find any of these keys: {tm.channel_map.keys()}")


TMAPS['acute_mi'] = TensorMap('acute_mi', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text, channel_map={'no_acute_mi': 0, 'ACUTE MI': 1},
                              loss=weighted_crossentropy([0.1, 10.0], 'acute_mi'))

TMAPS['anterior_blocks'] = TensorMap('anterior_blocks', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text,
                                     channel_map={'no_anterior_blocks': 0, 'Left anterior fascicular block': 1, 'Left posterior fascicular block': 2},
                                     loss=weighted_crossentropy([0.1, 10.0, 10.0], 'anterior_blocks'))

TMAPS['av_block'] = TensorMap('av_block', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text, channel_map={'no_av_block': 0, 'st degree AV block': 1},
                              loss=weighted_crossentropy([0.1, 10.0], 'av_block'))

TMAPS['incomplete_right_bundle_branch_block'] = TensorMap('incomplete_right_bundle_branch_block', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text,
                                                          channel_map={'no_incomplete_right_bundle_branch_block': 0, 'Incomplete right bundle branch block': 1},
                                                          loss=weighted_crossentropy([0.1, 10.0], 'incomplete_right_bundle_branch_block'))

TMAPS['infarcts'] = TensorMap('infarcts', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text,
                              channel_map={'no_infarcts': 0, 'Anterior infarct': 1, 'Anteroseptal infarct': 2, 'Inferior infarct': 3, 'Lateral infarct': 4, 'Septal infarct': 5},
                              loss=weighted_crossentropy([0.1, 4.0, 6.0, 7.0, 6.0, 4.0], 'infarcts'))

TMAPS['left_atrial_enlargement'] = TensorMap('left_atrial_enlargement', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text,
                                             channel_map={'no_left_atrial_enlargement': 0, 'Left atrial enlargement': 1},
                                             loss=weighted_crossentropy([0.1, 10.0], 'left_atrial_enlargement'))

TMAPS['left_ventricular_hypertrophy'] = TensorMap('left_ventricular_hypertrophy', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text,
                                                  channel_map={'no_left_ventricular_hypertrophy': 0, 'Left ventricular hypertrophy': 1},
                                                  loss=weighted_crossentropy([0.1, 10.0], 'left_ventricular_hypertrophy'))

TMAPS['lvh_fine'] = TensorMap('lvh_fine', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text, loss=weighted_crossentropy([0.5, 12.0, 16.0, 30.0, 36.0], 'lvh_fine'),
                              channel_map={'no_lvh_fine': 0, 'Minimal voltage criteria for LVH may be normal variant': 1,
                                           'Moderate voltage criteria for LVH may be normal variant': 2, 'Voltage criteria for left ventricular hypertrophy': 3,
                                           'Left ventricular hypertrophy': 4})

TMAPS['poor_data_quality'] = TensorMap('poor_data_quality', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text, channel_map={'no_poor_data_quality': 0, 'Poor data quality': 1},
                                       loss=weighted_crossentropy([0.1, 3.0], 'poor_data_quality'))

TMAPS['premature_atrial_complexes'] = TensorMap('premature_atrial_complexes', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text,
                                                channel_map={'no_premature_atrial_complexes': 0, 'premature atrial complexes': 1},
                                                loss=weighted_crossentropy([0.1, 10.0], 'premature_atrial_complexes'))

TMAPS['premature_supraventricular_complexes'] = TensorMap('premature_supraventricular_complexes', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text,
                                                          channel_map={'no_premature_supraventricular_complexes': 0, 'premature supraventricular complexes': 1},
                                                          loss=weighted_crossentropy([0.1, 10.0], 'premature_supraventricular_complexes'))

TMAPS['premature_ventricular_complexes'] = TensorMap('premature_ventricular_complexes', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text,
                                                     channel_map={'no_premature_ventricular_complexes': 0, 'premature ventricular complexes': 1},
                                                     loss=weighted_crossentropy([0.1, 10.0], 'premature_ventricular_complexes'))

TMAPS['prolonged_qt'] = TensorMap('prolonged_qt', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text, channel_map={'no_prolonged_qt': 0, 'Prolonged QT': 1},
                                  loss=weighted_crossentropy([0.1, 10.0], 'prolonged_qt'))


# Extract RAmplitude and SAmplitude for LVH criteria
def _make_ukb_ecg_rest(population_normalize: float = None):
    def ukb_ecg_rest_from_file(tm, hd5, dependents={}):
        if 'ukb_ecg_rest' not in hd5:
            raise ValueError('Group with R and S amplitudes not present in hd5')
        tensor = _get_tensor_at_first_date(hd5, tm.path_prefix, tm.name, _pass_nan)
        try:
            if population_normalize is None:
                tensor = tm.zero_mean_std1(tensor)
            else:
                tensor /= population_normalize
        except:
            ValueError(f'Cannot normalize {tm.name}')
        return tensor
    return ukb_ecg_rest_from_file


TMAPS['ecg_rest_ramplitude_raw'] = TensorMap('ramplitude', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(12,), tensor_from_file=_make_ukb_ecg_rest(1.0),
                                             loss='logcosh', metrics=['mse', 'mape', 'mae'], loss_weight=1.0)

TMAPS['ecg_rest_samplitude_raw'] = TensorMap('samplitude', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(12,), tensor_from_file=_make_ukb_ecg_rest(1.0),
                                             loss='logcosh', metrics=['mse', 'mape', 'mae'], loss_weight=1.0)

TMAPS['ecg_rest_ramplitude'] = TensorMap('ramplitude', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(12,), tensor_from_file=_make_ukb_ecg_rest(),
                                         loss='logcosh', metrics=['mse', 'mape', 'mae'], loss_weight=1.0)

TMAPS['ecg_rest_samplitude'] = TensorMap('samplitude', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(12,), tensor_from_file=_make_ukb_ecg_rest(),
                                         loss='logcosh', metrics=['mse', 'mape', 'mae'], loss_weight=1.0)


def _make_ukb_ecg_rest_lvh():
    def ukb_ecg_rest_lvh_from_file(tm, hd5, dependents={}):
        # Lead order seems constant and standard throughout, but we could eventually tensorize it from XML
        lead_order = ECG_REST_AMP_LEADS
        avl_min = 1100.0
        sl_min = 3500.0
        cornell_female_min = 2000.0
        cornell_male_min = 2800.0
        if 'ukb_ecg_rest' not in hd5:
            raise ValueError('Group with R and S amplitudes not present in hd5')
        tensor_ramp = _get_tensor_at_first_date(hd5, tm.path_prefix, 'ramplitude', _pass_nan)
        tensor_samp = _get_tensor_at_first_date(hd5, tm.path_prefix, 'samplitude', _pass_nan)
        criteria_sleads = [lead_order[l] for l in ['V1', 'V3']]
        criteria_rleads = [lead_order[l] for l in ['aVL', 'V5', 'V6']]
        if np.any(np.isnan(np.union1d(tensor_ramp[criteria_rleads], tensor_samp[criteria_sleads]))):
            raise ValueError('Missing some of the R and S amplitude readings needed to evaluate LVH criteria')
        is_female = 'Genetic-sex_Female_0_0' in hd5['categorical']
        is_male = 'Genetic-sex_Male_0_0' in hd5['categorical']
        # If genetic sex not available, try phenotypic
        if not(is_female or is_male):
            is_female = 'Sex_Female_0_0' in hd5['categorical']
            is_male = 'Sex_Male_0_0' in hd5['categorical']
        # If neither available, raise error
        if not(is_female or is_male):
            raise ValueError('Sex info required to evaluate LVH criteria')
        if tm.name == 'avl_lvh':
            is_lvh = tensor_ramp[lead_order['aVL']] > avl_min
        elif tm.name == 'sokolow_lyon_lvh':
            is_lvh = tensor_samp[lead_order['V1']] +\
                     np.maximum(tensor_ramp[lead_order['V5']], tensor_ramp[lead_order['V6']]) > sl_min
        elif tm.name == 'cornell_lvh':
            is_lvh = tensor_ramp[lead_order['aVL']] + tensor_samp[lead_order['V3']]
            if is_female:
                is_lvh = is_lvh > cornell_female_min
            if is_male:
                is_lvh = is_lvh > cornell_male_min
        else:
            raise ValueError(f'{tm.name} criterion for LVH is not accounted for')
        # Following convention from categorical TMAPS, positive has cmap index 1
        tensor = np.zeros(tm.shape, dtype=np.float32)
        index = 0
        if is_lvh:
            index = 1
        tensor[index] = 1.0
        return tensor
    return ukb_ecg_rest_lvh_from_file


TMAPS['ecg_rest_lvh_avl'] = TensorMap('avl_lvh', Interpretation.CATEGORICAL, path_prefix='ukb_ecg_rest', tensor_from_file=_make_ukb_ecg_rest_lvh(),
                                      channel_map={'no_avl_lvh': 0, 'aVL LVH': 1},
                                      loss=weighted_crossentropy([0.006, 1.0], 'avl_lvh'))

TMAPS['ecg_rest_lvh_sokolow_lyon'] = TensorMap('sokolow_lyon_lvh', Interpretation.CATEGORICAL, path_prefix='ukb_ecg_rest', tensor_from_file=_make_ukb_ecg_rest_lvh(),
                                               channel_map={'no_sokolow_lyon_lvh': 0, 'Sokolow Lyon LVH': 1},
                                               loss=weighted_crossentropy([0.005, 1.0], 'sokolov_lyon_lvh'))

TMAPS['ecg_rest_lvh_cornell'] = TensorMap('cornell_lvh', Interpretation.CATEGORICAL, path_prefix='ukb_ecg_rest', tensor_from_file=_make_ukb_ecg_rest_lvh(),
                                          channel_map={'no_cornell_lvh': 0, 'Cornell LVH': 1},
                                          loss=weighted_crossentropy([0.003, 1.0], 'cornell_lvh'))


def _ecg_rest_to_segment(population_normalize=None, hertz=500, random_offset_seconds=0):
    def ecg_rest_section_to_segment(tm, hd5, dependents={}):
        tensor = np.zeros(tm.shape, dtype=np.float32)
        segmented = tm.dependent_map.hd5_first_dataset_in_group(hd5, tm.dependent_map.hd5_key_guess())
        offset_seconds = float(segmented.attrs['offset_seconds'])
        random_offset_samples = 0
        if random_offset_seconds > 0:
            random_offset_begin = np.random.uniform(random_offset_seconds)
            offset_seconds += random_offset_begin
            random_offset_samples = int(random_offset_begin * hertz)
        offset_begin = int(offset_seconds * hertz)
        segment_index = np.array(segmented[random_offset_samples:random_offset_samples+tm.dependent_map.shape[0]], dtype=np.float32)
        dependents[tm.dependent_map] = to_categorical(segment_index, tm.dependent_map.shape[-1])
        for k in hd5[tm.path_prefix]:
            if k in tm.channel_map:
                tensor[:, tm.channel_map[k]] = np.array(hd5[tm.path_prefix][k], dtype=np.float32)[offset_begin:offset_begin+tm.shape[0]]
        if population_normalize is None:
            tm.normalization = {'zero_mean_std1': 1.0}
        else:
            tensor /= population_normalize
        return tensor
    return ecg_rest_section_to_segment


TMAPS['ecg_segmented'] = TensorMap('ecg_segmented', Interpretation.CATEGORICAL, shape=(1224, len(ECG_SEGMENTED_CHANNEL_MAP)), path_prefix='ecg_rest',
                                   cacheable=False, channel_map=ECG_SEGMENTED_CHANNEL_MAP)
TMAPS['ecg_section_to_segment'] = TensorMap('ecg_section_to_segment', shape=(1224, 12), path_prefix='ecg_rest', dependent_map=TMAPS['ecg_segmented'],
                                            channel_map=ECG_REST_LEADS, tensor_from_file=_ecg_rest_to_segment())
TMAPS['ecg_section_to_segment_warp'] = TensorMap('ecg_section_to_segment', shape=(1224, 12), path_prefix='ecg_rest', dependent_map=TMAPS['ecg_segmented'],
                                                 cacheable=False, channel_map=ECG_REST_LEADS, tensor_from_file=_ecg_rest_to_segment(),
                                                 augmentations=[_warp_ecg])

TMAPS['ecg_segmented_second'] = TensorMap('ecg_segmented', Interpretation.CATEGORICAL, shape=(496, len(ECG_SEGMENTED_CHANNEL_MAP)), path_prefix='ecg_rest',
                                          cacheable=False, channel_map=ECG_SEGMENTED_CHANNEL_MAP)
TMAPS['ecg_second_to_segment'] = TensorMap('ecg_second_to_segment', shape=(496, 12), path_prefix='ecg_rest', dependent_map=TMAPS['ecg_segmented_second'],
                                           cacheable=False, channel_map=ECG_REST_LEADS, tensor_from_file=_ecg_rest_to_segment(random_offset_seconds=1.5))
TMAPS['ecg_second_to_segment_warp'] = TensorMap('ecg_second_to_segment', shape=(496, 12), path_prefix='ecg_rest', dependent_map=TMAPS['ecg_segmented_second'],
                                                cacheable=False, channel_map=ECG_REST_LEADS, tensor_from_file=_ecg_rest_to_segment(random_offset_seconds=1.5),
                                                augmentations=[_warp_ecg])


TMAPS['t2_flair_sag_p2_1mm_fs_ellip_pf78_1'] = TensorMap('t2_flair_sag_p2_1mm_fs_ellip_pf78_1', shape=(256, 256, 192), path_prefix='ukb_brain_mri/float_array/',
                                                         tensor_from_file=normalized_first_date, normalization={'zero_mean_std1': True})
TMAPS['t2_flair_sag_p2_1mm_fs_ellip_pf78_2'] = TensorMap('t2_flair_sag_p2_1mm_fs_ellip_pf78_2', shape=(256, 256, 192), path_prefix='ukb_brain_mri/float_array/',
                                                         tensor_from_file=normalized_first_date, normalization={'zero_mean_std1': True})
TMAPS['t2_flair_slice_1'] = TensorMap('t2_flair_slice_1', shape=(256, 256, 1), path_prefix='ukb_brain_mri/float_array/', tensor_from_file=_random_slice_tensor('t2_flair_sag_p2_1mm_fs_ellip_pf78_1'), normalization={'zero_mean_std1': True})
TMAPS['t2_flair_slice_2'] = TensorMap('t2_flair_slice_2', shape=(256, 256, 1), path_prefix='ukb_brain_mri/float_array/', tensor_from_file=_random_slice_tensor('t2_flair_sag_p2_1mm_fs_ellip_pf78_2'), normalization={'zero_mean_std1': True})
TMAPS['t1_p2_1mm_fov256_sag_ti_880_1'] = TensorMap('t1_p2_1mm_fov256_sag_ti_880_1', shape=(256, 256, 208), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True}, tensor_from_file=normalized_first_date)
TMAPS['t1_p2_1mm_fov256_sag_ti_880_2'] = TensorMap('t1_p2_1mm_fov256_sag_ti_880_2', shape=(256, 256, 208), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True}, tensor_from_file=normalized_first_date)
TMAPS['t1_dicom_30_slices'] = TensorMap('t1_dicom_30_slices', shape=(192, 256, 30), path_prefix='ukb_brain_mri/float_array/',  normalization={'zero_mean_std1': True},
                                        tensor_from_file=_slice_subset_tensor('t1_p2_1mm_fov256_sag_ti_880_1', 130, 190, 2, pad_shape=(192, 256, 256), flip_swap=True))
TMAPS['t2_dicom_30_slices'] = TensorMap('t2_dicom_30_slices', shape=(192, 256, 30), path_prefix='ukb_brain_mri/',  normalization={'zero_mean_std1': True},
                                        tensor_from_file=_slice_subset_tensor('t2_flair_sag_p2_1mm_fs_ellip_pf78_1', 130, 190, 2, pad_shape=(192, 256, 256), flip_swap=True))

TMAPS['t1_slice_1'] = TensorMap('t1_slice_1', shape=(256, 256, 1), path_prefix='ukb_brain_mri/float_array/',  normalization={'zero_mean_std1': True},
                                tensor_from_file=_random_slice_tensor('t1_p2_1mm_fov256_sag_ti_880_1'))
TMAPS['t1_slice_2'] = TensorMap('t1_slice_2', shape=(256, 256, 1), path_prefix='ukb_brain_mri/float_array/',  normalization={'zero_mean_std1': True},
                                tensor_from_file=_random_slice_tensor('t1_p2_1mm_fov256_sag_ti_880_2'))
TMAPS['t1_20_slices_1'] = TensorMap('t1_20_slices_1', shape=(256, 256, 20), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True},
                                    tensor_from_file=_slice_subset_tensor('t1_p2_1mm_fov256_sag_ti_880_1', 94, 114))
TMAPS['t1_20_slices_2'] = TensorMap('t1_20_slices_2', shape=(256, 256, 20), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True},
                                    tensor_from_file=_slice_subset_tensor('t1_p2_1mm_fov256_sag_ti_880_2', 94, 114))
TMAPS['t2_20_slices_1'] = TensorMap('t2_20_slices_1', shape=(256, 256, 20), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True},
                                    tensor_from_file=_slice_subset_tensor('t2_flair_sag_p2_1mm_fs_ellip_pf78_1', 86, 106))
TMAPS['t2_20_slices_2'] = TensorMap('t2_20_slices_2', shape=(256, 256, 20), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True},
                                    tensor_from_file=_slice_subset_tensor('t2_flair_sag_p2_1mm_fs_ellip_pf78_2', 86, 106))
TMAPS['t1_40_slices_1'] = TensorMap('t1_40_slices_1', shape=(256, 256, 40), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True},
                                    tensor_from_file=_slice_subset_tensor('t1_p2_1mm_fov256_sag_ti_880_1', 64, 144, 2))
TMAPS['t2_40_slices_1'] = TensorMap('t2_40_slices_1', shape=(256, 256, 40), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True},
                                    tensor_from_file=_slice_subset_tensor('t2_flair_sag_p2_1mm_fs_ellip_pf78_1', 56, 136, 2))
TMAPS['sos_te1'] = TensorMap('SOS_TE1', shape=(256, 288, 48), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True}, tensor_from_file=normalized_first_date)
TMAPS['sos_te2'] = TensorMap('SOS_TE2', shape=(256, 288, 48), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True}, tensor_from_file=normalized_first_date)
TMAPS['swi'] = TensorMap('SWI', shape=(256, 288, 48), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True}, tensor_from_file=normalized_first_date)
TMAPS['swi_total_mag'] = TensorMap('SWI_TOTAL_MAG', shape=(256, 288, 48), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True}, tensor_from_file=normalized_first_date)
TMAPS['swi_total_mag_te2_orig'] = TensorMap('SWI_TOTAL_MAG_TE2_orig', shape=(256, 288, 48), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True}, tensor_from_file=normalized_first_date)
TMAPS['swi_total_mag_orig'] = TensorMap('SWI_TOTAL_MAG_orig', shape=(256, 288, 48), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True}, tensor_from_file=normalized_first_date)
TMAPS['t2star'] = TensorMap('T2star', shape=(256, 288, 48), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True}, tensor_from_file=normalized_first_date)
TMAPS['brain_mask_normed'] = TensorMap('brain_mask_normed', shape=(256, 288, 48), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True}, tensor_from_file=normalized_first_date)

TMAPS['filtered_phase'] = TensorMap('filtered_phase', shape=(256, 288, 48), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True}, tensor_from_file=normalized_first_date)
TMAPS['swi_to_t1_40_slices'] = TensorMap('swi_to_t1_40_slices', shape=(173, 231, 40), path_prefix='ukb_brain_mri/float_array/',
                                         normalization={'zero_mean_std1': True},
                                         tensor_from_file=_slice_subset_tensor('SWI_TOTAL_MAG_to_T1', 60, 140, 2))
TMAPS['t2star_to_t1_40_slices'] = TensorMap('t2star_to_t1_40_slices', shape=(173, 231, 40), path_prefix='ukb_brain_mri/float_array/',
                                            normalization={'zero_mean_std1': True},
                                            tensor_from_file=_slice_subset_tensor('T2star_to_T1', 60, 140, 2))

TMAPS['t1'] = TensorMap('T1', shape=(192, 256, 256, 1), path_prefix='ukb_brain_mri/float_array/',  normalization={'zero_mean_std1': True}, tensor_from_file=normalized_first_date)
TMAPS['t1_brain'] = TensorMap('T1_brain', shape=(192, 256, 256, 1), path_prefix='ukb_brain_mri/float_array/',  normalization={'zero_mean_std1': True}, tensor_from_file=normalized_first_date)
TMAPS['t1_brain_30_slices'] = TensorMap('t1_brain_30_slices', shape=(192, 256, 30), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True}, tensor_from_file=_slice_subset_tensor('T1_brain', 66, 126, 2, pad_shape=(192, 256, 256)))
TMAPS['t1_30_slices'] = TensorMap('t1_30_slices', shape=(192, 256, 30), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True}, tensor_from_file=_slice_subset_tensor('T1', 90, 150, 2, pad_shape=(192, 256, 256)))
TMAPS['t1_30_slices_4d'] = TensorMap('t1_30_slices_4d', shape=(192, 256, 30, 1), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True}, tensor_from_file=_slice_subset_tensor('T1', 90, 150, 2, pad_shape=(192, 256, 256, 1)))
TMAPS['t1_30_slices_fs'] = TensorMap('t1_30_slices', shape=(192, 256, 30), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True},
                                     tensor_from_file=_slice_subset_tensor('T1', 90, 150, 2, pad_shape=(192, 256, 256), flip_swap=True, swap_axes=1))
TMAPS['t1_30_slices_4d_fs'] = TensorMap('t1_30_slices_4d', shape=(192, 256, 30, 1), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True},
                                        tensor_from_file=_slice_subset_tensor('T1', 90, 150, 2, pad_shape=(192, 256, 256, 1), flip_swap=True, swap_axes=1))

TMAPS['t1_brain_to_mni'] = TensorMap('T1_brain_to_MNI', shape=(192, 256, 256, 1), path_prefix='ukb_brain_mri/float_array/',  normalization={'zero_mean_std1': True}, tensor_from_file=normalized_first_date)
TMAPS['t1_fast_t1_brain_bias'] = TensorMap('T1_fast_T1_brain_bias', shape=(192, 256, 256, 1), path_prefix='ukb_brain_mri/float_array/',  normalization={'zero_mean_std1': True}, tensor_from_file=normalized_first_date)

TMAPS['t2_flair'] = TensorMap('T2_FLAIR', shape=(192, 256, 256, 1), path_prefix='ukb_brain_mri/float_array/',  normalization={'zero_mean_std1': True}, tensor_from_file=normalized_first_date)
TMAPS['t2_flair_brain'] = TensorMap('T2_FLAIR_brain', shape=(192, 256, 256, 1), path_prefix='ukb_brain_mri/float_array/',  normalization={'zero_mean_std1': True}, tensor_from_file=normalized_first_date)
TMAPS['t2_flair_brain_30_slices'] = TensorMap('t2_flair_brain_30_slices', shape=(192, 256, 30), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True},
                                              tensor_from_file=_slice_subset_tensor('T2_FLAIR_brain', 66, 126, 2, pad_shape=(192, 256, 256)))
TMAPS['t2_flair_30_slices'] = TensorMap('t2_flair_30_slices', shape=(192, 256, 30), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True},
                                        tensor_from_file=_slice_subset_tensor('T2_FLAIR', 90, 150, 2, pad_shape=(192, 256, 256)))
TMAPS['t2_flair_30_slices_4d'] = TensorMap('t2_flair_30_slices_4d', shape=(192, 256, 30, 1), path_prefix='ukb_brain_mri/float_array/',  tensor_from_file=_slice_subset_tensor('T2_FLAIR', 90, 150, 2, pad_shape=(192, 256, 256, 1)),
                                           normalization={'zero_mean_std1': True})
TMAPS['t2_flair_30_slices_fs'] = TensorMap('t2_flair_30_slices', shape=(192, 256, 30), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True},
                                           tensor_from_file=_slice_subset_tensor('T2_FLAIR', 90, 150, 2, pad_shape=(192, 256, 256), flip_swap=True, swap_axes=1))
TMAPS['t2_flair_30_slices_4d_fs'] = TensorMap('t2_flair_30_slices_4d', shape=(192, 256, 30, 1), path_prefix='ukb_brain_mri/float_array/', normalization={'zero_mean_std1': True},
                                              tensor_from_file=_slice_subset_tensor('T2_FLAIR', 90, 150, 2, pad_shape=(192, 256, 256, 1), flip_swap=True, swap_axes=1))
TMAPS['t2_flair_unbiased_brain'] = TensorMap('T2_FLAIR_unbiased_brain', shape=(192, 256, 256, 1), path_prefix='ukb_brain_mri/float_array/',  normalization={'zero_mean_std1': True}, tensor_from_file=normalized_first_date)


def _mask_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
    original = _get_tensor_at_first_date(hd5, tm.path_prefix, tm.name)
    reshaped = _pad_or_crop_array_to_shape(tm.shape, original)
    tensor = to_categorical(reshaped[..., 0], tm.shape[-1])
    return tensor


def _mask_subset_tensor(tensor_key, start, stop, step=1, pad_shape=None):
    slice_subset_tensor_from_file = _slice_subset_tensor(tensor_key, start, stop, step=step, pad_shape=pad_shape, dtype_override='float_array')

    def mask_subset_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        original = slice_subset_tensor_from_file(tm, hd5, dependents)
        tensor = to_categorical(original[..., 0], tm.shape[-1])
        return tensor
    return mask_subset_from_file


TMAPS['swi_brain_mask'] = TensorMap('SWI_brain_mask', Interpretation.CATEGORICAL, shape=(256, 288, 48, 2), path_prefix='ukb_brain_mri/float_array/',
                                    tensor_from_file=_mask_from_file, channel_map={'not_brain': 0, 'brain': 1})
TMAPS['t1_brain_mask'] = TensorMap('T1_brain_mask', Interpretation.CATEGORICAL, shape=(192, 256, 256, 2), path_prefix='ukb_brain_mri/float_array/',
                                   tensor_from_file=_mask_from_file, channel_map={'not_brain': 0, 'brain': 1})
TMAPS['t1_seg'] = TensorMap('T1_fast_T1_brain_seg', Interpretation.CATEGORICAL, shape=(192, 256, 256, 4), path_prefix='ukb_brain_mri/float_array/',
                            tensor_from_file=_mask_from_file, channel_map={'not_brain_tissue': 0, 'csf': 1, 'grey': 2, 'white': 3})
TMAPS['t1_seg_30_slices'] = TensorMap('T1_fast_T1_brain_seg_30_slices', Interpretation.CATEGORICAL, shape=(192, 256, 30, 4), path_prefix='ukb_brain_mri/float_array/',
                                      tensor_from_file=_mask_subset_tensor('T1_fast_T1_brain_seg', 90, 150, 2, pad_shape=(192, 256, 256, 1)),
                                      channel_map={'not_brain_tissue': 0, 'csf': 1, 'grey': 2, 'white': 3})
TMAPS['t1_brain_mask_30_slices'] = TensorMap('T1_brain_mask_30_slices', Interpretation.CATEGORICAL, shape=(192, 256, 30, 2), path_prefix='ukb_brain_mri/float_array/',
                                             tensor_from_file=_mask_subset_tensor('T1_brain_mask', 90, 150, 2, pad_shape=(192, 256, 256, 1)),
                                             channel_map={'not_brain': 0, 'brain': 1})
TMAPS['lesions'] = TensorMap('lesions_final_mask', Interpretation.CATEGORICAL, shape=(192, 256, 256, 2), path_prefix='ukb_brain_mri/float_array/',
                             tensor_from_file=_mask_from_file, channel_map={'not_lesion': 0, 'lesion': 1}, loss=weighted_crossentropy([0.01, 10.0], 'lesion'))


def _combined_subset_tensor(tensor_keys, start, stop, step=1, pad_shape=None, flip_swap=False):
    slice_subsets = [_slice_subset_tensor(k, start, stop, step=step, pad_shape=pad_shape, allow_channels=False, flip_swap=flip_swap) for k in tensor_keys]

    def mask_subset_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        tensor = np.zeros(tm.shape, dtype=np.float32)
        for i, slice_subset_tensor_from_file in enumerate(slice_subsets):
            tensor[..., i] = slice_subset_tensor_from_file(tm, hd5, dependents)
        return tensor
    return mask_subset_from_file


TMAPS['t1_and_t2_flair_30_slices'] = TensorMap('t1_and_t2_flair_30_slices', Interpretation.CONTINUOUS, shape=(192, 256, 30, 2), path_prefix='ukb_brain_mri',
                                               tensor_from_file=_combined_subset_tensor(['T1', 'T2_FLAIR'], 90, 150, 2, pad_shape=(192, 256, 256)),
                                               normalization={'zero_mean_std1': True})
_dicom_keys = ['t1_p2_1mm_fov256_sag_ti_880_1', 't2_flair_sag_p2_1mm_fs_ellip_pf78_1']
TMAPS['t1_t2_dicom_30_slices'] = TensorMap('t1_t2_dicom_30_slices', Interpretation.CONTINUOUS, shape=(192, 256, 30, 2), path_prefix='ukb_brain_mri',
                                           tensor_from_file=_combined_subset_tensor(_dicom_keys, 130, 190, 2, pad_shape=(192, 256, 256), flip_swap=True),
                                           normalization={'zero_mean_std1': True})
TMAPS['t1_t2_dicom_50_slices'] = TensorMap('t1_t2_dicom_50_slices', Interpretation.CONTINUOUS, shape=(192, 256, 50, 2), path_prefix='ukb_brain_mri',
                                           tensor_from_file=_combined_subset_tensor(_dicom_keys, 100, 200, 2, pad_shape=(192, 256, 256), flip_swap=True),
                                           normalization={'zero_mean_std1': True})


def _ttn_tensor_from_file(tm, hd5, dependents={}):
    index = 0
    categorical_data = np.zeros(tm.shape, dtype=np.float32)
    if 'has_exome' not in hd5['categorical']:
        raise ValueError('Skipping people without exome sequencing.')
    if tm.name in hd5['categorical'] and int(hd5['categorical'][tm.name][0]) != 0:
        index = 1
    categorical_data[index] = 1.0
    return categorical_data


TMAPS['ttntv'] = TensorMap('has_ttntv',  Interpretation.CATEGORICAL, channel_map={'no_TTN_tv': 0, 'TTN_tv': 1}, tensor_from_file=_ttn_tensor_from_file)
TMAPS['ttntv_10x'] = TensorMap('has_ttntv',  Interpretation.CATEGORICAL, channel_map={'no_TTN_tv': 0, 'TTN_tv': 1}, loss_weight=10.0, tensor_from_file=_ttn_tensor_from_file)


def _make_index_tensor_from_file(index_map_name):
    def indexed_lvmass_tensor_from_file(tm, hd5, dependents={}):
        tensor = np.zeros(tm.shape, dtype=np.float32)
        for k in tm.channel_map:
            tensor = np.array(hd5[tm.path_prefix][k], dtype=np.float32)
        index = np.array(hd5[tm.path_prefix][index_map_name], dtype=np.float32)
        return tensor / index
    return indexed_lvmass_tensor_from_file


TMAPS['lv_mass_dubois_index'] = TensorMap('lv_mass_dubois_index', Interpretation.CONTINUOUS, activation='linear', loss='logcosh', loss_weight=1.0,
                                          tensor_from_file=_make_index_tensor_from_file('bsa_dubois'),
                                          channel_map={'lv_mass': 0}, normalization={'mean': 89.7, 'std': 24.8})
TMAPS['lv_mass_mosteller_index'] = TensorMap('lv_mass_mosteller_index', Interpretation.CONTINUOUS, activation='linear', loss='logcosh', loss_weight=1.0,
                                             tensor_from_file=_make_index_tensor_from_file('bsa_mosteller'),
                                             channel_map={'lv_mass': 0}, normalization={'mean': 89.7, 'std': 24.8})
TMAPS['lv_mass_dubois_index_sentinel'] = TensorMap('lv_mass_dubois_index', Interpretation.CONTINUOUS, activation='linear', sentinel=0, loss_weight=1.0,
                                                   tensor_from_file=_make_index_tensor_from_file('bsa_dubois'),
                                                   channel_map={'lv_mass': 0}, normalization={'mean': 89.7, 'std': 24.8})
TMAPS['lv_mass_mosteller_index_sentinel'] = TensorMap('lv_mass_mosteller_index', Interpretation.CONTINUOUS, activation='linear', sentinel=0, loss_weight=1.0,
                                                      tensor_from_file=_make_index_tensor_from_file('bsa_mosteller'),
                                                      channel_map={'lv_mass': 0}, normalization={'mean': 89.7, 'std': 24.8})

TMAPS['lvm_dubois_index'] = TensorMap('lvm_dubois_index', Interpretation.CONTINUOUS, activation='linear', loss='logcosh', loss_weight=1.0,
                                      tensor_from_file=_make_index_tensor_from_file('bsa_dubois'),
                                      channel_map={'LVM': 0}, normalization={'mean': 89.7, 'std': 24.8})
TMAPS['lvm_mosteller_index'] = TensorMap('lvm_mosteller_index', Interpretation.CONTINUOUS, activation='linear', loss='logcosh', loss_weight=1.0,
                                         tensor_from_file=_make_index_tensor_from_file('bsa_mosteller'),
                                         channel_map={'LVM': 0}, normalization={'mean': 89.7, 'std': 24.8})
TMAPS['lvm_dubois_index_w4'] = TensorMap('lvm_dubois_index', Interpretation.CONTINUOUS, activation='linear', loss='logcosh', loss_weight=4.0,
                                         tensor_from_file=_make_index_tensor_from_file('bsa_dubois'),
                                         channel_map={'LVM': 0}, normalization={'mean': 89.7, 'std': 24.8})
TMAPS['lvm_mosteller_index_w4'] = TensorMap('lvm_mosteller_index', Interpretation.CONTINUOUS, activation='linear', loss='logcosh', loss_weight=4.0,
                                            tensor_from_file=_make_index_tensor_from_file('bsa_mosteller'),
                                            channel_map={'LVM': 0}, normalization={'mean': 89.7, 'std': 24.8})
TMAPS['lvm_dubois_index_sentinel'] = TensorMap('lvm_dubois_index', Interpretation.CONTINUOUS, activation='linear', sentinel=0, loss_weight=1.0,
                                               tensor_from_file=_make_index_tensor_from_file('bsa_dubois'),
                                               channel_map={'LVM': 0}, normalization={'mean': 89.7, 'std': 24.8})
TMAPS['lvm_mosteller_index_sentinel'] = TensorMap('lvm_mosteller_index', Interpretation.CONTINUOUS, activation='linear', sentinel=0, loss_weight=1.0,
                                                  tensor_from_file=_make_index_tensor_from_file('bsa_mosteller'),
                                                  channel_map={'LVM': 0}, normalization={'mean': 89.7, 'std': 24.8})


def _select_tensor_from_file(selection_predicate: Callable):
    def selected_tensor_from_file(tm, hd5, dependents={}):
        if not selection_predicate(hd5):
            raise ValueError(f'Tensor did not meet selection criteria:{selection_predicate.__name__} with Tensor Map:{tm.name}')
        tensor = np.zeros(tm.shape, dtype=np.float32)
        for k in tm.channel_map:
            tensor = np.array(hd5[tm.path_prefix][k], dtype=np.float32)
        return tensor
    return selected_tensor_from_file


def _is_genetic_man(hd5):
    return 'Genetic-sex_Male_0_0' in hd5['categorical']


def _is_genetic_woman(hd5):
    return 'Genetic-sex_Female_0_0' in hd5['categorical']


TMAPS['myocardial_mass_noheritable_men_only'] = TensorMap('inferred_myocardial_mass_noheritable', Interpretation.CONTINUOUS, activation='linear', loss='logcosh',
                                                          tensor_from_file=_select_tensor_from_file(_is_genetic_man),
                                                          channel_map={'inferred_myocardial_mass_noheritable': 0}, normalization={'mean': 100.0, 'std': 18.0})
TMAPS['myocardial_mass_noheritable_women_only'] = TensorMap('inferred_myocardial_mass_noheritable', Interpretation.CONTINUOUS, activation='linear', loss='logcosh',
                                                            tensor_from_file=_select_tensor_from_file(_is_genetic_woman),
                                                            channel_map={'inferred_myocardial_mass_noheritable': 0}, normalization={'mean': 78.0, 'std': 16.0})


def _make_lvh_from_lvm_tensor_from_file(lvm_key, group_key='continuous', male_lvh_threshold=72, female_lvh_threshold=55):
    def lvh_from_lvm_tensor_from_file(tm, hd5, dependents={}):
        tensor = np.zeros(tm.shape, dtype=np.float32)
        lvm_indexed = float(hd5[group_key][lvm_key][0])
        index = 0
        if _is_genetic_man(hd5) and lvm_indexed > male_lvh_threshold:
            index = 1
        elif _is_genetic_woman(hd5) and lvm_indexed > female_lvh_threshold:
            index = 1
        tensor[index] = 1
        return tensor
    return lvh_from_lvm_tensor_from_file


TMAPS['lvh_from_indexed_lvm'] = TensorMap('lvh_from_indexed_lvm', Interpretation.CATEGORICAL, channel_map={'no_lvh': 0, 'left_ventricular_hypertrophy': 1},
                                          tensor_from_file=_make_lvh_from_lvm_tensor_from_file('adjusted_myocardium_mass_indexed'))
TMAPS['lvh_from_indexed_lvm_weighted'] = TensorMap('lvh_from_indexed_lvm', Interpretation.CATEGORICAL, channel_map={'no_lvh': 0, 'left_ventricular_hypertrophy': 1},
                                          tensor_from_file=_make_lvh_from_lvm_tensor_from_file('adjusted_myocardium_mass_indexed'),
                                          loss=weighted_crossentropy([1.0, 25.0], 'lvh_from_indexed_lvm'))
TMAPS['adjusted_myocardium_mass'] = TensorMap('adjusted_myocardium_mass', Interpretation.CONTINUOUS, validator=make_range_validator(0, 400), path_prefix='continuous',
                                              loss='logcosh', channel_map={'adjusted_myocardium_mass': 0}, normalization={'mean': 89.70, 'std': 24.80})
TMAPS['adjusted_myocardium_mass_indexed'] = TensorMap('adjusted_myocardium_mass_indexed', Interpretation.CONTINUOUS, validator=make_range_validator(0, 400),
                                                      loss='logcosh', channel_map={'adjusted_myocardium_mass_indexed': 0}, path_prefix='continuous',
                                                      normalization={'mean': 89.70, 'std': 24.80})
TMAPS['lvh_from_indexed_lvm_parented'] = TensorMap('lvh_from_indexed_lvm', Interpretation.CATEGORICAL, channel_map={'no_lvh': 0, 'left_ventricular_hypertrophy': 1},
                                                   tensor_from_file=_make_lvh_from_lvm_tensor_from_file('adjusted_myocardium_mass_indexed'),
                                                   loss=weighted_crossentropy([1.0, 25.0], 'lvh_from_indexed_lvm_parented'),
                                                   parents=[TMAPS['adjusted_myocardium_mass_indexed'], TMAPS['adjusted_myocardium_mass']])


def _mri_slice_blackout_tensor_from_file(tm, hd5, dependents={}):
    cur_slice = np.random.choice(list(hd5[MRI_TO_SEGMENT].keys()))
    tensor = np.zeros(tm.shape, dtype=np.float32)
    dependents[tm.dependent_map] = np.zeros(tm.dependent_map.shape, dtype=np.float32)
    tensor[:, :, 0] = np.array(hd5[MRI_TO_SEGMENT][cur_slice], dtype=np.float32)
    label_tensor = np.array(hd5[MRI_SEGMENTED][cur_slice], dtype=np.float32)
    dependents[tm.dependent_map][:, :, :] = to_categorical(label_tensor, tm.dependent_map.shape[-1])
    tensor[:, :, 0] *= np.not_equal(label_tensor, 0, dtype=np.float32)
    return tm.zero_mean_std1(tensor)


TMAPS['mri_slice_blackout_segmented_weighted'] = TensorMap('mri_slice_segmented', Interpretation.CATEGORICAL, shape=(256, 256, 3), channel_map=MRI_SEGMENTED_CHANNEL_MAP,
                                                           loss=weighted_crossentropy([0.1, 25.0, 25.0], 'mri_slice_blackout_segmented'))
TMAPS['mri_slice_blackout'] = TensorMap('mri_slice_blackout', Interpretation.CONTINUOUS, shape=(256, 256, 1), tensor_from_file=_mri_slice_blackout_tensor_from_file,
                                        dependent_map=TMAPS['mri_slice_blackout_segmented_weighted'])


def _mri_tensor_2d(hd5, name):
    """
    Returns MRI image annotation tensors as 2-D numpy arrays. Useful for annotations that may vary from slice to slice
    """
    if isinstance(hd5[name], h5py.Group):
        nslices = len(hd5[name]) // MRI_FRAMES
        for ann in hd5[name]:
            ann_shape = hd5[name][ann].shape
            break
        shape = (ann_shape[0], nslices)
        arr = np.zeros(shape)
        t = 0
        s = 0
        for k in sorted(hd5[name], key=int):
            t += 1
            if t == MRI_FRAMES:
                arr[:, s] = hd5[name][k]
                s += 1
                t = 0
    elif isinstance(hd5[name], h5py.Dataset):
        nslices = 1
        shape = (hd5[name].shape[0], nslices)
        arr = np.zeros(shape)
        arr[:, 0] = hd5[name]
    else:
        raise ValueError(f'{name} is neither a HD5 Group nor a HD5 dataset')
    return arr


def _make_mri_series_orientation_and_position_from_file(population_normalize=None):
    def mri_series_orientation_and_position(tm, hd5):
        if len(tm.shape) < 2:
            tensor = np.array(hd5[tm.name], dtype=np.float32)
        else:
            arr = _mri_tensor_2d(hd5, tm.name)
            tensor = np.array(arr, dtype=np.float32)
        if population_normalize is not None:
            tensor /= population_normalize
        return tensor
    return mri_series_orientation_and_position


TMAPS['mri_patient_orientation_cine_segmented_lax_2ch'] = TensorMap('mri_patient_orientation_cine_segmented_lax_2ch', Interpretation.CONTINUOUS, shape=(6,), path_prefix='mri_orientation',
                                                                    tensor_from_file=_make_mri_series_orientation_and_position_from_file())
TMAPS['mri_patient_orientation_cine_segmented_lax_3ch'] = TensorMap('mri_patient_orientation_cine_segmented_lax_3ch', Interpretation.CONTINUOUS, shape=(6,), path_prefix='mri_orientation',
                                                                    tensor_from_file=_make_mri_series_orientation_and_position_from_file())
TMAPS['mri_patient_orientation_cine_segmented_lax_4ch'] = TensorMap('mri_patient_orientation_cine_segmented_lax_4ch', Interpretation.CONTINUOUS, shape=(6,), path_prefix='mri_orientation',
                                                                    tensor_from_file=_make_mri_series_orientation_and_position_from_file())
TMAPS['mri_patient_orientation_cine_segmented_sax_b1'] = TensorMap('mri_patient_orientation_cine_segmented_sax_b1', Interpretation.CONTINUOUS, shape=(6,), path_prefix='mri_orientation',
                                                                   tensor_from_file=_make_mri_series_orientation_and_position_from_file())
TMAPS['mri_patient_orientation_cine_segmented_sax_inlinevf'] = TensorMap('mri_patient_orientation_cine_segmented_sax_inlinevf', Interpretation.CONTINUOUS, shape=(6, 750), path_prefix='mri_orientation',
                                                                         tensor_from_file=_make_mri_series_orientation_and_position_from_file())
TMAPS['mri_patient_position_cine_segmented_lax_2ch'] = TensorMap('mri_patient_position_cine_segmented_lax_2ch', Interpretation.CONTINUOUS, shape=(3,), path_prefix='mri_position',
                                                                 tensor_from_file=_make_mri_series_orientation_and_position_from_file())
TMAPS['mri_patient_position_cine_segmented_lax_3ch'] = TensorMap('mri_patient_position_cine_segmented_lax_3ch', Interpretation.CONTINUOUS, shape=(3,), path_prefix='mri_position',
                                                                 tensor_from_file=_make_mri_series_orientation_and_position_from_file())
TMAPS['mri_patient_position_cine_segmented_lax_4ch'] = TensorMap('mri_patient_position_cine_segmented_lax_4ch', Interpretation.CONTINUOUS, shape=(3,), path_prefix='mri_position',
                                                                 tensor_from_file=_make_mri_series_orientation_and_position_from_file())
TMAPS['mri_patient_position_cine_segmented_sax_b1'] = TensorMap('mri_patient_position_cine_segmented_sax_b1', Interpretation.CONTINUOUS, shape=(3,), path_prefix='mri_position',
                                                                tensor_from_file=_make_mri_series_orientation_and_position_from_file())
TMAPS['mri_patient_position_cine_segmented_sax_inlinevf'] = TensorMap('mri_patient_position_cine_segmented_sax_inlinevf', Interpretation.CONTINUOUS, shape=(3, 750), path_prefix='mri_position',
                                                                      tensor_from_file=_make_mri_series_orientation_and_position_from_file())


def _mri_tensor_4d(hd5, name):
    """
    Returns MRI image tensors from HD5 as 4-D numpy arrays. Useful for raw SAX and LAX images and segmentations.
    """
    if isinstance(hd5[name], h5py.Group):
        nslices = len(hd5[name]) // MRI_FRAMES
        for img in hd5[name]:
            img_shape = hd5[name][img].shape
            break
        shape = (img_shape[0], img_shape[1], nslices, MRI_FRAMES)
        arr = np.zeros(shape)
        t = 0
        s = 0
        for k in sorted(hd5[name], key=int):
            arr[:, :, s, t] = np.array(hd5[name][k]).T
            t += 1
            if t == MRI_FRAMES:
                s += 1
                t = 0
    elif isinstance(hd5[name], h5py.Dataset):
        nslices = 1
        shape = (hd5[name].shape[0], hd5[name].shape[1], nslices, MRI_FRAMES)
        arr = np.zeros(shape)
        for t in range(MRI_FRAMES):
            arr[:, :, 0, t] = np.array(hd5[name][:, :, t]).T
    else:
        raise ValueError(f'{name} is neither a HD5 Group nor a HD5 dataset')
    return arr


def _mri_hd5_to_structured_grids(hd5, name, save_path=None, order='F'):
    """
    Returns MRI tensors as list of VTK structured grids aligned to the reference system of the patient
    """
    arr = _mri_tensor_4d(hd5, name)
    width = hd5['_'.join([MRI_PIXEL_WIDTH, name])]
    height = hd5['_'.join([MRI_PIXEL_HEIGHT, name])]
    positions = _mri_tensor_2d(hd5, '_'.join([MRI_PATIENT_POSITION, name]))
    orientations = _mri_tensor_2d(hd5, '_'.join([MRI_PATIENT_ORIENTATION, name]))
    thickness = hd5['_'.join([MRI_SLICE_THICKNESS, name])]
    _, dataset_indices, dataset_counts = np.unique(orientations, axis=1, return_index=True, return_counts=True)
    grids = []
    for d_idx, d_cnt in zip(dataset_indices, dataset_counts):
        grids.append(vtk.vtkStructuredGrid())
        nslices = d_cnt
        # If multislice, override thickness as distance between voxel centers. Note: removes eventual gaps between slices
        if nslices > 1:
            thickness = np.linalg.norm(positions[:, d_idx] - positions[:, d_idx+1])
        transform = vtk.vtkTransform()
        n_orientation = np.cross(orientations[3:, d_idx], orientations[:3, d_idx])
        # 4x4 transform matrix to align to the patient reference system
        transform.SetMatrix([orientations[3, d_idx]*height, orientations[0, d_idx]*width, n_orientation[0]*thickness, positions[0, d_idx],
                             orientations[4, d_idx]*height, orientations[1, d_idx]*width, n_orientation[1]*thickness, positions[1, d_idx],
                             orientations[5, d_idx]*height, orientations[2, d_idx]*width, n_orientation[2]*thickness, positions[2, d_idx],
                             0, 0, 0, 1])
        x_coors = np.arange(0, arr.shape[0]+1) - 0.5
        y_coors = np.arange(0, arr.shape[1]+1) - 0.5
        z_coors = np.arange(0, d_cnt+1) - 0.5
        xyz_meshgrid = np.meshgrid(x_coors, y_coors, z_coors)
        xyz_pts = np.zeros(((arr.shape[0]+1) * (arr.shape[1]+1) * (d_cnt+1), 3))
        for dim in range(3):
            xyz_pts[:, dim] = xyz_meshgrid[dim].ravel(order=order)
        vtk_pts = vtk.vtkPoints()
        vtk_pts.SetData(vtk.util.numpy_support.numpy_to_vtk(xyz_pts))
        grids[-1].SetPoints(vtk_pts)
        grids[-1].SetDimensions(len(x_coors), len(y_coors), len(z_coors))
        grids[-1].SetExtent(0, len(x_coors)-1, 0, len(y_coors)-1, 0, len(z_coors)-1)
        for t in range(MRI_FRAMES):
            arr_vtk = vtk.util.numpy_support.numpy_to_vtk(arr[:, :, d_idx:d_idx+d_cnt, t].ravel(order=order), deep=True)
            arr_vtk.SetName(f'{name}_{t}')
            grids[-1].GetCellData().AddArray(arr_vtk)
        transform_filter = vtk.vtkTransformFilter()
        transform_filter.SetInputData(grids[-1])
        transform_filter.SetTransform(transform)
        transform_filter.Update()
        grids[-1].DeepCopy(transform_filter.GetOutput())
        if save_path:
            writer = vtk.vtkXMLStructuredGridWriter()
            writer.SetFileName(os.path.join(save_path, f'grid_{name}_{d_idx}.vts'))
            writer.SetInputData(grids[-1])
            writer.Update()
    return grids


def _cut_through_plane(dataset, plane_center, plane_orientation):
    plane = vtk.vtkPlane()
    plane.SetOrigin(plane_center)
    plane.SetNormal(plane_orientation)
    cutter = vtk.vtkCutter()
    cutter.SetInputData(dataset)
    cutter.SetCutFunction(plane)
    poly = vtk.vtkDataSetSurfaceFilter()
    poly.SetInputConnection(cutter.GetOutputPort())
    poly.Update()
    return poly.GetOutput()


def _map_points_to_cells(pts, dataset, tol=1e-3):
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(dataset)
    locator.BuildLocator()
    closest_pt = np.zeros(3)
    generic_cell = vtk.vtkGenericCell()
    cell_id, sub_id, dist2, inside = vtk.mutable(0), vtk.mutable(0), vtk.mutable(0.0), vtk.mutable(0)
    map_to_cells = np.zeros(len(pts), dtype=np.int64)
    for pt_id, pt in enumerate(pts):
        if locator.FindClosestPointWithinRadius(pt, tol, closest_pt, generic_cell, cell_id, sub_id, dist2, inside):
            map_to_cells[pt_id] = cell_id.get()
    return map_to_cells


def _make_mri_projected_segmentation_from_file(to_segment_name, segmented_name, save_path=None):
    def mri_projected_segmentation(tm, hd5):
        if segmented_name not in [MRI_SEGMENTED, MRI_LAX_SEGMENTED]:
            raise ValueError(f'{segmented_name} is recognized neither as SAX nor LAX segmentation')
        cine_segmented_grids = _mri_hd5_to_structured_grids(hd5, segmented_name)
        cine_to_segment_grids = _mri_hd5_to_structured_grids(hd5, to_segment_name)
        tensor = np.zeros(tm.shape, dtype=np.float32)
        # Loop through segmentations and datasets
        for ds_i, ds_segmented in enumerate(cine_segmented_grids):
            for ds_j, ds_to_segment in enumerate(cine_to_segment_grids):
                dims = ds_to_segment.GetDimensions()
                pts = vtk.util.numpy_support.vtk_to_numpy(ds_to_segment.GetPoints().GetData())
                npts_per_slice = dims[0] * dims[1]
                ncells_per_slice = (dims[0]-1) * (dims[1]-1)
                n_orientation = (pts[npts_per_slice] - pts[0])
                n_orientation /= np.linalg.norm(n_orientation)
                cell_centers = vtk.vtkCellCenters()
                cell_centers.SetInputData(ds_to_segment)
                cell_centers.Update()
                cell_pts = vtk.util.numpy_support.vtk_to_numpy(cell_centers.GetOutput().GetPoints().GetData())
                # Loop through dataset slices
                for s in range(dims[2]-1):
                    slice_center = np.mean(pts[s*npts_per_slice:(s+2)*npts_per_slice], axis=0)
                    slice_cell_pts = cell_pts[s*ncells_per_slice:(s+1)*ncells_per_slice]
                    slice_segmented = _cut_through_plane(ds_segmented, slice_center, n_orientation)
                    map_to_segmented = _map_points_to_cells(slice_cell_pts, slice_segmented)
                    # Loop through time
                    for t in range(MRI_FRAMES):
                        arr_name = f'{segmented_name}_{t}'
                        segmented_arr = vtk.util.numpy_support.vtk_to_numpy(slice_segmented.GetCellData().GetArray(arr_name))
                        projected_arr = segmented_arr[map_to_segmented]
                        if len(tm.shape) == 3:
                            tensor[:, :, t] = np.maximum(tensor[:, :, t], projected_arr.reshape(tm.shape[0], tm.shape[1]))
                        elif len(tm.shape) == 4:
                            tensor[:, :, s, t] = np.maximum(tensor[:, :, s, t], projected_arr.reshape(tm.shape[0], tm.shape[1]))
                    if save_path:
                        writer_segmented = vtk.vtkXMLPolyDataWriter()
                        writer_segmented.SetInputData(slice_segmented)
                        writer_segmented.SetFileName(os.path.join(save_path, f'{tm.name}_segmented_{ds_i}_{ds_j}_{s}.vtp'))
                        writer_segmented.Update()
        return tensor
    return mri_projected_segmentation


TMAPS['cine_segmented_lax_2ch_proj_from_sax'] = TensorMap('cine_segmented_lax_2ch_proj_from_sax', Interpretation.CONTINUOUS, shape=(256, 256, 50), loss='logcosh',
                                                          tensor_from_file=_make_mri_projected_segmentation_from_file('cine_segmented_lax_2ch', MRI_SEGMENTED))
TMAPS['cine_segmented_lax_3ch_proj_from_sax'] = TensorMap('cine_segmented_lax_3ch_proj_from_sax', Interpretation.CONTINUOUS,  shape=(256, 256, 50), loss='logcosh',
                                                          tensor_from_file=_make_mri_projected_segmentation_from_file('cine_segmented_lax_3ch', MRI_SEGMENTED))
TMAPS['cine_segmented_lax_4ch_proj_from_sax'] = TensorMap('cine_segmented_lax_4ch_proj_from_sax', Interpretation.CONTINUOUS,  shape=(256, 256, 50), loss='logcosh',
                                                          tensor_from_file=_make_mri_projected_segmentation_from_file('cine_segmented_lax_4ch', MRI_SEGMENTED))
TMAPS['cine_segmented_lax_2ch_proj_from_lax'] = TensorMap('cine_segmented_lax_2ch_proj_from_lax', Interpretation.CONTINUOUS,  shape=(256, 256, 50), loss='logcosh',
                                                          tensor_from_file=_make_mri_projected_segmentation_from_file('cine_segmented_lax_2ch', MRI_LAX_SEGMENTED))
TMAPS['cine_segmented_lax_3ch_proj_from_lax'] = TensorMap('cine_segmented_lax_3ch_proj_from_lax', Interpretation.CONTINUOUS,  shape=(256, 256, 50), loss='logcosh',
                                                          tensor_from_file=_make_mri_projected_segmentation_from_file('cine_segmented_lax_3ch', MRI_LAX_SEGMENTED))
TMAPS['cine_segmented_lax_4ch_proj_from_lax'] = TensorMap('cine_segmented_lax_4ch_proj_from_lax', Interpretation.CONTINUOUS,  shape=(256, 256, 50), loss='logcosh',
                                                          tensor_from_file=_make_mri_projected_segmentation_from_file('cine_segmented_lax_4ch', MRI_LAX_SEGMENTED))


def _slice_tensor(tensor_key, slice_index):
    def _slice_tensor_from_file(tm, hd5, dependents={}):
        if tm.shape[-1] == 1:
            t = _pad_or_crop_array_to_shape(tm.shape[:-1], np.array(hd5[tensor_key][..., slice_index], dtype=np.float32))
            tensor = np.expand_dims(t, axis=-1)
        else:
            tensor = _pad_or_crop_array_to_shape(tm.shape, np.array(hd5[tensor_key][..., slice_index], dtype=np.float32))
        return tensor
    return _slice_tensor_from_file


TMAPS['lax_4ch_diastole_slice0_3d'] = TensorMap('lax_4ch_diastole_slice0_3d', Interpretation.CONTINUOUS, shape=(200, 160, 1), loss='logcosh',
                                                normalization={'zero_mean_std1': True}, tensor_from_file=_slice_tensor('ukb_cardiac_mri/cine_segmented_lax_4ch/instance_0', 0))
TMAPS['lax_3ch_diastole_slice0_3d'] = TensorMap('lax_3ch_diastole_slice0_3d', Interpretation.CONTINUOUS, shape=(200, 160, 1), loss='logcosh',
                                                normalization={'zero_mean_std1': True}, tensor_from_file=_slice_tensor('ukb_cardiac_mri/cine_segmented_lax_3ch/instance_0', 0))
TMAPS['cine_segmented_ao_dist_slice0_3d'] = TensorMap('cine_segmented_ao_dist_slice0_3d', Interpretation.CONTINUOUS, shape=(256, 256, 1), loss='logcosh',
                                                      normalization={'zero_mean_std1': True}, tensor_from_file=_slice_tensor('ukb_cardiac_mri/cine_segmented_ao_dist/instance_0', 0))
TMAPS['lax_4ch_diastole_slice0'] = TensorMap('lax_4ch_diastole_slice0', Interpretation.CONTINUOUS, shape=(256, 256), loss='logcosh',
                                             normalization={'zero_mean_std1': True}, tensor_from_file=_slice_tensor('ukb_cardiac_mri/cine_segmented_lax_4ch/instance_0', 0))
TMAPS['lax_3ch_diastole_slice0'] = TensorMap('lax_3ch_diastole_slice0', Interpretation.CONTINUOUS, shape=(256, 256), loss='logcosh',
                                             normalization={'zero_mean_std1': True}, tensor_from_file=_slice_tensor('ukb_cardiac_mri/cine_segmented_lax_3ch/instance_0', 0))
TMAPS['cine_segmented_ao_dist_slice0'] = TensorMap('cine_segmented_ao_dist_slice0', Interpretation.CONTINUOUS, shape=(256, 256), loss='logcosh',
                                                   normalization={'zero_mean_std1': True}, tensor_from_file=_slice_tensor('ukb_cardiac_mri/cine_segmented_ao_dist/instance_0', 0))



def _pad_crop_tensor(tm, hd5, dependents={}):
    return _pad_or_crop_array_to_shape(tm.shape, np.array(tm.hd5_first_dataset_in_group(hd5, tm.hd5_key_guess()), dtype=np.float32))


TMAPS['cine_lax_3ch_192'] = TensorMap('cine_segmented_lax_3ch', Interpretation.CONTINUOUS, shape=(192, 192, 50), path_prefix='ukb_cardiac_mri',
                                      tensor_from_file=_pad_crop_tensor, normalization={'zero_mean_std1': True})
TMAPS['cine_lax_3ch_160_1'] = TensorMap('cine_segmented_lax_3ch', Interpretation.CONTINUOUS, shape=(160, 160, 50, 1), path_prefix='ukb_cardiac_mri',
                                        tensor_from_file=_pad_crop_tensor, normalization={'zero_mean_std1': True})
TMAPS['cine_lax_3ch_192_160_1'] = TensorMap('cine_segmented_lax_3ch', Interpretation.CONTINUOUS, shape=(192, 160, 50, 1), path_prefix='ukb_cardiac_mri',
                                            tensor_from_file=_pad_crop_tensor, normalization={'zero_mean_std1': True})
TMAPS['cine_ao_dist_4d'] = TensorMap('cine_segmented_ao_dist', Interpretation.CONTINUOUS, shape=(160, 192, 100, 1), path_prefix='ukb_cardiac_mri',
                                     tensor_from_file=_pad_crop_tensor, normalization={'zero_mean_std1': True})
TMAPS['cine_lax_4ch_192'] = TensorMap('cine_segmented_lax_3ch', Interpretation.CONTINUOUS, shape=(192, 192, 50), path_prefix='ukb_cardiac_mri',
                                      tensor_from_file=_pad_crop_tensor, normalization={'zero_mean_std1': True})
TMAPS['cine_lax_4ch_192_1'] = TensorMap('cine_segmented_lax_3ch', Interpretation.CONTINUOUS, shape=(192, 192, 50, 1), path_prefix='ukb_cardiac_mri',
                                        tensor_from_file=_pad_crop_tensor, normalization={'zero_mean_std1': True})
TMAPS['cine_sax_b6_192'] = TensorMap('cine_segmented_sax_b6', Interpretation.CONTINUOUS, shape=(192, 192, 50), path_prefix='ukb_cardiac_mri',
                                     tensor_from_file=_pad_crop_tensor, normalization={'zero_mean_std1': True})
TMAPS['cine_sax_b6_192_1'] = TensorMap('cine_segmented_sax_b6', Interpretation.CONTINUOUS, shape=(192, 192, 50, 1),  path_prefix='ukb_cardiac_mri',
                                       tensor_from_file=_pad_crop_tensor, normalization={'zero_mean_std1': True})


def _segmented_dicom_slices(dicom_key_prefix, path_prefix='ukb_cardiac_mri'):
    def _segmented_dicom_tensor_from_file(tm, hd5, dependents={}):
        tensor = np.zeros(tm.shape, dtype=np.float32)
        for i in range(tm.shape[-2]):
            categorical_index_slice = _get_tensor_at_first_date(hd5, path_prefix, dicom_key_prefix + str(i+1))
            categorical_one_hot = to_categorical(categorical_index_slice, len(tm.channel_map))
            tensor[..., i, :] = _pad_or_crop_array_to_shape(tensor[..., i, :].shape, categorical_one_hot)
        return tensor
    return _segmented_dicom_tensor_from_file


TMAPS['lax_3ch_segmented'] = TensorMap('lax_3ch_segmented',  Interpretation.CATEGORICAL, shape=(256, 256, 50, 6),
                                       tensor_from_file=_segmented_dicom_slices('cine_segmented_lax_3ch_annotated_'),
                                       channel_map=MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP)
TMAPS['lax_3ch_segmented_192'] = TensorMap('lax_3ch_segmented', Interpretation.CATEGORICAL, shape=(192, 192, 50, 6),
                                           tensor_from_file=_segmented_dicom_slices('cine_segmented_lax_3ch_annotated_'),
                                           channel_map=MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP)
TMAPS['lax_3ch_segmented_192_160'] = TensorMap('lax_3ch_segmented', Interpretation.CATEGORICAL, shape=(192, 160, 50, 6),
                                               tensor_from_file=_segmented_dicom_slices('cine_segmented_lax_3ch_annotated_'),
                                               channel_map=MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP)
TMAPS['lax_3ch_segmented_192_160'] = TensorMap('lax_3ch_segmented', Interpretation.CATEGORICAL, shape=(192, 160, 50, 6),
                                               tensor_from_file=_segmented_dicom_slices('cine_segmented_lax_3ch_annotated_'),
                                               channel_map=MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP)
TMAPS['lax_4ch_segmented'] = TensorMap('lax_4ch_segmented', Interpretation.CATEGORICAL, shape=(256, 256, 50, 14),
                                       tensor_from_file=_segmented_dicom_slices('cine_segmented_lax_4ch_annotated_'),
                                       channel_map=MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP)
TMAPS['lax_4ch_segmented_192'] = TensorMap('lax_4ch_segmented', Interpretation.CATEGORICAL, shape=(192, 192, 50, 14),
                                           tensor_from_file=_segmented_dicom_slices('cine_segmented_lax_4ch_annotated_'),
                                           channel_map=MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP)
TMAPS['lax_4ch_segmented_192_w'] = TensorMap('lax_4ch_segmented', Interpretation.CATEGORICAL, shape=(192, 192, 50, 14),
                                             tensor_from_file=_segmented_dicom_slices('cine_segmented_lax_4ch_annotated_'),
                                             channel_map=MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP,
                                             loss=weighted_crossentropy([0.01, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 5.0, 0.5]))
TMAPS['sax_segmented_b6'] = TensorMap('sax_segmented_b6', Interpretation.CATEGORICAL, shape=(256, 256, 50, 11),
                                      tensor_from_file=_segmented_dicom_slices('cine_segmented_sax_b6_annotated_'),
                                      channel_map=MRI_SAX_SEGMENTED_CHANNEL_MAP)
TMAPS['sax_segmented_b6_192'] = TensorMap('sax_segmented_b6', Interpretation.CATEGORICAL, shape=(192, 192, 50, 11),
                                          tensor_from_file=_segmented_dicom_slices('cine_segmented_sax_b6_annotated_'),
                                          channel_map=MRI_SAX_SEGMENTED_CHANNEL_MAP)

TMAPS['cine_segmented_ao_dist'] = TensorMap('cine_segmented_ao_dist', Interpretation.CATEGORICAL, shape=(160, 192, 100, len(MRI_AO_SEGMENTED_CHANNEL_MAP)),
                                            tensor_from_file=_segmented_dicom_slices('cine_segmented_ao_dist_annotated_'), channel_map=MRI_AO_SEGMENTED_CHANNEL_MAP)


def _make_fallback_tensor_from_file(tensor_keys):
    def fallback_tensor_from_file(tm, hd5, dependents={}):
        for k in tensor_keys:
            if k in hd5:
                tensor = np.array(hd5[k], dtype=np.float32)
                return tensor
        raise ValueError(f'No fallback tensor found from keys: {tensor_keys}')
    return fallback_tensor_from_file


TMAPS['shmolli_192i_both'] = TensorMap('shmolli_192i', Interpretation.CONTINUOUS, shape=(288, 384, 7),
                                       tensor_from_file=_make_fallback_tensor_from_file(['shmolli_192i', 'shmolli_192i_liver']))
TMAPS['shmolli_192i_liver_only'] = TensorMap('shmolli_192i', Interpretation.CONTINUOUS, shape=(288, 384, 7),
                                             tensor_from_file=_make_fallback_tensor_from_file(['shmolli_192i_liver']))


def preprocess_with_function(fxn, hd5_key=None):
    def preprocess_tensor_from_file(tm, hd5, dependents={}):
        missing = True
        continuous_data = np.zeros(tm.shape, dtype=np.float32)
        my_key = tm.hd5_key_guess() if hd5_key is None else hd5_key
        if my_key in hd5:
            missing = False
            continuous_data[0] = tm.hd5_first_dataset_in_group(hd5, my_key)[0]
        if missing and tm.sentinel is None:
            raise ValueError(f'No value found for {tm.name}, a continuous TensorMap with no sentinel value, and channel keys:{list(tm.channel_map.keys())}.')
        elif missing:
            continuous_data[:] = tm.sentinel
        return fxn(continuous_data)
    return preprocess_tensor_from_file


TMAPS['log_25781_2'] = TensorMap('25781_Total-volume-of-white-matter-hyperintensities-from-T1-and-T2FLAIR-images_2_0', loss='logcosh', path_prefix='continuous',
                                 normalization={'mean': 7, 'std': 8}, tensor_from_file=preprocess_with_function(np.log),
                                 channel_map={'white-matter-hyper-intensities': 0})
TMAPS['weight_lbs_2'] = TensorMap('weight_lbs',  Interpretation.CONTINUOUS, normalization={'mean': 168.74, 'std': 34.1}, loss='logcosh',
                                  channel_map={'weight_lbs': 0}, tensor_from_file=preprocess_with_function(lambda x: x*2.20462, 'continuous/21002_Weight_2_0'))


def _weekly_alcohol(instance):
    alcohol_keys = [f'1568_Average-weekly-red-wine-intake_{instance}_0', f'1578_Average-weekly-champagne-plus-white-wine-intake_{instance}_0',
                    f'1588_Average-weekly-beer-plus-cider-intake_{instance}_0', f'1598_Average-weekly-spirits-intake_{instance}_0',
                    f'1608_Average-weekly-fortified-wine-intake_{instance}_0']

    def alcohol_from_file(tm, hd5, dependents={}):
        drinks = 0
        for k in alcohol_keys:
            data = tm.hd5_first_dataset_in_group(hd5, key_prefix=f'{tm.path_prefix}/{k}')
            drinks += float(data[0])
        return np.array([drinks], dtype=np.float32)
    return alcohol_from_file


TMAPS['weekly_alcohol_0'] = TensorMap('weekly_alcohol_0', loss='logcosh', path_prefix='continuous', channel_map={'weekly_alcohol_0': 0}, tensor_from_file=_weekly_alcohol(0))
TMAPS['weekly_alcohol_1'] = TensorMap('weekly_alcohol_1', loss='logcosh', path_prefix='continuous', channel_map={'weekly_alcohol_1': 0}, tensor_from_file=_weekly_alcohol(1))
TMAPS['weekly_alcohol_2'] = TensorMap('weekly_alcohol_2', loss='logcosh', path_prefix='continuous', channel_map={'weekly_alcohol_2': 0}, tensor_from_file=_weekly_alcohol(2))


def sax_tensor(b_series_prefix):
    def sax_tensor_from_file(tm, hd5, dependents={}):
        missing = 0
        tensor = np.zeros(tm.shape, dtype=np.float32)
        dependents[tm.dependent_map] = np.zeros(tm.dependent_map.shape, dtype=np.float32)
        for b in range(tm.shape[-2]):
            try:
                tm_shape = (tm.shape[0], tm.shape[1])
                tensor[:, :, b, 0] = _pad_or_crop_array_to_shape(tm_shape, np.array(hd5[f'{b_series_prefix}_frame_b{b}'], dtype=np.float32))
                index_tensor = _pad_or_crop_array_to_shape(tm_shape, np.array(hd5[f'{b_series_prefix}_mask_b{b}'], dtype=np.float32))
                dependents[tm.dependent_map][:, :, b, :] = to_categorical(index_tensor, tm.dependent_map.shape[-1])
            except KeyError:
                missing += 1
                tensor[:, :, b, 0] = 0
                dependents[tm.dependent_map][:, :, b, MRI_SEGMENTED_CHANNEL_MAP['background']] = 1
        if missing == tm.shape[-2]:
            raise ValueError(f'Could not find any slices in {tm.name} was hoping for {tm.shape[-2]}')
        return tensor
    return sax_tensor_from_file


TMAPS['sax_all_diastole_segmented'] = TensorMap('sax_all_diastole_segmented', Interpretation.CATEGORICAL, shape=(256, 256, 13, 3),
                                                channel_map=MRI_SEGMENTED_CHANNEL_MAP)
TMAPS['sax_all_diastole_segmented_weighted'] = TensorMap('sax_all_diastole_segmented', Interpretation.CATEGORICAL, shape=(256, 256, 13, 3),
                                                         channel_map=MRI_SEGMENTED_CHANNEL_MAP,
                                                         loss=weighted_crossentropy([1.0, 40.0, 40.0], 'sax_all_diastole_segmented'))

TMAPS['sax_all_diastole'] = TensorMap('sax_all_diastole', shape=(256, 256, 13, 1), tensor_from_file=sax_tensor('diastole'),
                                      dependent_map=TMAPS['sax_all_diastole_segmented'])
TMAPS['sax_all_diastole_weighted'] = TensorMap('sax_all_diastole', shape=(256, 256, 13, 1), tensor_from_file=sax_tensor('diastole'),
                                               dependent_map=TMAPS['sax_all_diastole_segmented_weighted'])

TMAPS['sax_all_systole_segmented'] = TensorMap('sax_all_systole_segmented', Interpretation.CATEGORICAL, shape=(256, 256, 13, 3),
                                               channel_map=MRI_SEGMENTED_CHANNEL_MAP)
TMAPS['sax_all_systole_segmented_weighted'] = TensorMap('sax_all_systole_segmented_weighted', Interpretation.CATEGORICAL, shape=(256, 256, 13, 3),
                                                        channel_map=MRI_SEGMENTED_CHANNEL_MAP,
                                                        loss=weighted_crossentropy([1.0, 40.0, 40.0], 'sax_all_systole_segmented'))

TMAPS['sax_all_systole'] = TensorMap('sax_all_systole', shape=(256, 256, 13, 1), tensor_from_file=sax_tensor('systole'),
                                     dependent_map=TMAPS['sax_all_systole_segmented'])
TMAPS['sax_all_systole_weighted'] = TensorMap('sax_all_systole_weighted', shape=(256, 256, 13, 1), tensor_from_file=sax_tensor('systole'),
                                              dependent_map=TMAPS['sax_all_systole_segmented_weighted'])


def all_sax_tensor(total_b_slices=13):
    def sax_tensor_from_file(tm, hd5, dependents={}):
        missing = 0
        tensor = np.zeros(tm.shape, dtype=np.float32)
        dependents[tm.dependent_map] = np.zeros(tm.dependent_map.shape, dtype=np.float32)
        for b in range(total_b_slices):
            try:
                tm_shape = (tm.shape[0], tm.shape[1])
                tensor[:, :, b, 0] = _pad_or_crop_array_to_shape(tm_shape, np.array(hd5[f'diastole_frame_b{b}'], dtype=np.float32))
                index_tensor = _pad_or_crop_array_to_shape(tm_shape, np.array(hd5[f'diastole_mask_b{b}'], dtype=np.float32))
                dependents[tm.dependent_map][:, :, b, :] = to_categorical(index_tensor, tm.dependent_map.shape[-1])
                tensor[:, :, b + total_b_slices, 0] = _pad_or_crop_array_to_shape(tm_shape, np.array(hd5[f'systole_frame_b{b}'], dtype=np.float32))
                index_tensor = _pad_or_crop_array_to_shape(tm_shape, np.array(hd5[f'systole_mask_b{b}'], dtype=np.float32))
                dependents[tm.dependent_map][:, :, b + total_b_slices, :] = to_categorical(index_tensor, tm.dependent_map.shape[-1])
            except KeyError:
                missing += 1
                tensor[:, :, b, 0] = 0
                dependents[tm.dependent_map][:, :, b, MRI_SEGMENTED_CHANNEL_MAP['background']] = 1
        if missing == tm.shape[-2]:
            raise ValueError(f'Could not find any slices in {tm.name} was hoping for {tm.shape[-2]}')
        return tensor
    return sax_tensor_from_file


TMAPS['sax_all_segmented'] = TensorMap('sax_all_segmented', Interpretation.CATEGORICAL, shape=(256, 256, 26, 3), channel_map=MRI_SEGMENTED_CHANNEL_MAP)
TMAPS['sax_all_segmented_weighted'] = TensorMap('sax_all_segmented_weighted', Interpretation.CATEGORICAL, shape=(256, 256, 26, 3),
                                                channel_map=MRI_SEGMENTED_CHANNEL_MAP, loss=weighted_crossentropy([1.0, 40.0, 40.0], 'sax_all_segmented'))

TMAPS['sax_all'] = TensorMap('sax_all', shape=(256, 256, 26, 1), tensor_from_file=all_sax_tensor(), dependent_map=TMAPS['sax_all_segmented'])
TMAPS['sax_all_weighted'] = TensorMap('sax_all_weighted', shape=(256, 256, 26, 1), tensor_from_file=all_sax_tensor(), dependent_map=TMAPS['sax_all_segmented_weighted'])


# BIKE ECG
def _check_phase_full_len(hd5: h5py.File, phase: str):
    phase_len = _get_tensor_at_first_date(hd5, 'ecg_bike/continuous', f'{phase}_duration')
    valid = True
    if phase == 'pretest':
        valid &= phase_len == 15
    elif phase == 'exercise':
        valid &= phase_len == 360
    elif phase == 'rest':
        valid &= phase_len == 60
    else:
        raise ValueError(f'Phase {phase} is not a valid phase.')
    if not valid:
        raise ValueError(f'{phase} phase is not full length')


def _get_bike_ecg(hd5, tm: TensorMap, start: int, leads: Union[List[int], slice], stop: Optional[int] = None):
    path_prefix, name = 'ecg_bike/float_array', 'full'
    ecg_dataset = first_dataset_at_path(hd5, tensor_path(path_prefix, name))
    tensor = np.array(ecg_dataset[start: tm.shape[0] + start if stop is None else stop, leads], dtype=np.float32)
    if np.max(np.abs(tensor)) > 1000:
        raise ValueError('ECG range too large.')
    return _fail_nan(tensor)


def _get_r_peaks(ecg):
    downsample = 2
    sampling_rate = 500 / downsample
    order = int(0.1 * sampling_rate)
    filtered, _, _ = biosppy.signals.tools.filter_signal(
        signal=ecg[::downsample],
        ftype='FIR',
        band='bandpass',
        order=order,
        frequency=[3, 45],
        sampling_rate=sampling_rate
    )
    rpeaks, = biosppy.signals.ecg.hamilton_segmenter(signal=filtered, sampling_rate=sampling_rate)
    return np.array(rpeaks) * downsample


def _simple_r_peak(ecg):
    return np.argmax(ecg)


ECG_ALIGN_OFFSET = 800  # Should see at least one beat in ECG_ALIGN_OFFSET / 500 seconds


def _get_aligned_bike_ecg(hd5, tm: TensorMap, start: int, leads: Union[List[int], slice], length=None):
    ecg = _get_bike_ecg(hd5, tm, start - ECG_ALIGN_OFFSET, leads, stop=(length or tm.shape[0]) + start)
    first_peak = _simple_r_peak(ecg[:ECG_ALIGN_OFFSET, 0])
    ecg = ecg[first_peak: first_peak + (length or tm.shape[0])]
    return _fail_nan(ecg)


# ECG AUGMENTATIONS #
def _warp_ecg(ecg):
    """
    Warning: does some weird stuff at the boundaries
    """
    i = np.arange(ecg.shape[0])
    warped = i + (np.random.rand() * 100 * np.sin(i / (500 + np.random.rand() * 100))
                  + np.random.rand() * 100 * np.cos(i / (500 + np.random.rand() * 100)))
    warped_ecg = np.zeros_like(ecg)
    for j in range(ecg.shape[1]):
        warped_ecg[:, j] = np.interp(i, warped, ecg[:, j])
    return warped_ecg


def _downsample(ecg, rate: float):
    """
    rate=2 halves the sampling rate. Uses linear interpolation.
    """
    i = np.arange(0, ecg.shape[0], rate)
    downsampled = np.zeros((ecg.shape[0] // rate, ecg.shape[1]))
    for j in range(ecg.shape[1]):
        downsampled[:, j] = np.interp(i, np.arange(ecg.shape[0]), ecg[:, j])
    return downsampled


def _smooth(ecg, rate: float):
    """
    Expensive.
    """
    smoothed_ecg = np.zeros_like(ecg)
    for i in range(ecg.shape[1]):
        smoothed_ecg[:, i] = (1 - rate) * ecg[:, i] + rate * biosppy.signals.ecg.ecg(ecg[:, i], show=False)[1]
    return smoothed_ecg


def _rand_smooth(ecg):
    return _smooth(ecg, np.random.rand())


def _rand_add_noise(ecg):
    noise_frac = np.random.rand() * .1  # max of 10% noise
    return ecg + noise_frac * ecg.mean(axis=0) * np.random.randn(*ecg.shape)


def _roll(ecg, shift: int):
    return np.roll(ecg, shift=shift, axis=0)


def _rand_roll(ecg):
    return _roll(ecg, shift=np.random.randint(ecg.shape[0]))


def _rand_offset(ecg):
    shift_frac = np.random.rand() * .1  # max of 10% noise
    return ecg + shift_frac * ecg.mean(axis=0)


# BIKE ECG TENSOR FROM FILE #
def _build_bike_ecg_tensor_from_file(start, leads: Union[List[int], slice]):
    return lambda tm, hd5, dependents: (_get_bike_ecg(hd5, tm, start, leads))


def _bike_ecg_shifted(leads: Union[List[int], slice]):
    def _tff(tm: TensorMap, hd5: h5py.File, dependents=None):
        pretest_len = 15 * 500
        start = np.random.randint(pretest_len - tm.shape[0])
        ecg = _get_bike_ecg(hd5, tm, start, leads)
        return _fail_nan(ecg)
    return _tff


def _bike_ecg_aligned_augmented(augmentations: [Callable], leads: Union[List[int], slice]):
    def _tff(tm: TensorMap, hd5: h5py.File, dependents=None):
        pretest_len = 15 * 500
        start = np.random.randint(ECG_ALIGN_OFFSET, pretest_len - tm.shape[0])
        ecg = _get_aligned_bike_ecg(hd5, tm, start, leads)
        for func in augmentations:
            ecg = func(ecg)
        return ecg
    return _tff


def _bike_ecg_first_r_aligned(augmentations: [Callable], leads: Union[List[int], slice]):
    def _tff(tm: TensorMap, hd5: h5py.File, dependents=None):
        ecg = _get_aligned_bike_ecg(hd5, tm, ECG_ALIGN_OFFSET, leads)
        for func in augmentations:
            ecg = func(ecg)
        return ecg
    return _tff


def _bike_ecg_shifted_downsampled(leads: Union[List[int], slice], rate: float):
    def _tff(tm: TensorMap, hd5: h5py.File, dependents=None):
        pretest_len = 15 * 500
        length = int(tm.shape[0] * rate)
        start = np.random.randint(pretest_len - length)
        ecg = _get_bike_ecg(hd5, tm, start, leads, stop=start + length)
        ecg = _downsample(ecg, rate)
        return _fail_nan(ecg)
    return _tff


def _bike_ecg_first_r_aligned_downsampled(augmentations: [Callable], leads: Union[List[int], slice], rate: float):
    def _tff(tm: TensorMap, hd5: h5py.File, dependents=None):
        ecg = _get_aligned_bike_ecg(hd5, tm, ECG_ALIGN_OFFSET, leads, length=int(rate * tm.shape[0]))
        ecg =_downsample(ecg, rate)
        for func in augmentations:
            ecg = func(ecg)
        return ecg
    return _tff


def _build_peak_exercise(augmentations: [Callable], leads: Union[List[int], slice]):

    def tff(tm: TensorMap, hd5: h5py.File, dependents=None):
        _check_phase_full_len(hd5, 'pretest')
        _check_phase_full_len(hd5, 'rest')

        exercise_dur = _get_tensor_at_first_date(hd5, 'ecg_bike/continuous', 'exercise_duration')
        shift = 250 - 500 * np.random.rand()
        exercise_end = int(500 * (15 + exercise_dur - 2.5) + shift)  # 2.5 seconds before recovery begins
        ecg = _get_bike_ecg(hd5, tm, exercise_end, leads=leads)
        for func in augmentations:
            ecg = func(ecg)
        return ecg

    return tff


def _build_recovery_downsampled(leads: Union[List[int], slice], downsample_rate=8):

    def tff(tm: TensorMap, hd5: h5py.File, dependents=None):
        _check_phase_full_len(hd5, 'rest')
        pretest_dur = _get_tensor_at_first_date(hd5, 'ecg_bike/continuous', 'exercise_duration')
        exercise_dur = _get_tensor_at_first_date(hd5, 'ecg_bike/continuous', 'exercise_duration')
        shift = 50 - 100 * np.random.rand()
        exercise_end = int(500 * (pretest_dur + exercise_dur - 7) + shift)  # 5 seconds before recovery begins
        ecg = _get_bike_ecg(hd5, tm, exercise_end, stop=exercise_end + (5 + 50) * 500, leads=leads)
        return _downsample(ecg, downsample_rate)

    return tff


def _build_end_recovery(augmentations: [Callable], leads: Union[List[int], slice]):

    def tff(tm: TensorMap, hd5: h5py.File, dependents=None):
        _check_phase_full_len(hd5, 'pretest')
        _check_phase_full_len(hd5, 'rest')

        exercise_dur = _get_tensor_at_first_date(hd5, 'ecg_bike/continuous', 'exercise_duration')
        shift = -500 * np.random.rand()
        recovery_end = int(500 * (15 + exercise_dur + 60 - 7) + shift)  # 5 seconds before recovery ends
        ecg = _get_bike_ecg(hd5, tm, recovery_end, leads=leads)
        for func in augmentations:
            ecg = func(ecg)
        return ecg

    return tff


TMAPS['ecg-bike-pretest-leadI'] = TensorMap('full', shape=(2048, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
                                            validator=no_nans, normalization={'mean': 7, 'std': 31}, metrics=['mse'],
                                            tensor_from_file=_build_bike_ecg_tensor_from_file(0, [0]),)


TMAPS['ecg-bike-recovery-downsampled8x'] = TensorMap(
    'recovery_ecg', shape=(3437, 3), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'mean': 7, 'std': 31}, metrics=['mse'],
    tensor_from_file=_build_recovery_downsampled([0, 1, 2])
)


def _build_augmented_bike_ecg_tmaps():
    ecg_augmentations = _warp_ecg, _rand_roll, _rand_add_noise,
    name = f'ecg_bike_shifted'
    TMAPS[name] = TensorMap('full', shape=(2048, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
                            validator=no_nans, normalization={'mean': 7, 'std': 31}, cacheable=False,
                            tensor_from_file=_bike_ecg_shifted([0]), )
    for i, aug in enumerate(ecg_augmentations):
        name += aug.__name__
        TMAPS[name] = TensorMap('full', shape=(2048, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
                                validator=no_nans, normalization={'mean': 7, 'std': 31}, cacheable=False,
                                augmentations=list(ecg_augmentations[:i + 1]),
                                tensor_from_file=_bike_ecg_shifted([0]))


_build_augmented_bike_ecg_tmaps()


def _make_incomplete_exercise(tff: Callable) -> Callable:
    def new_tff(tm: TensorMap, hd5: h5py.File, dependents=None):
        out = tff(tm, hd5, dependents)
        if _get_tensor_at_first_date(hd5, 'ecg_bike/continuous', 'exercise_duration') > 359:
            raise ValueError('Participant completed exercise.')
        return out
    return new_tff


def _make_ramp(tff: Callable) -> Callable:
    def new_tff(tm: TensorMap, hd5: h5py.File, dependents=None):
        out = tff(tm, hd5, dependents)
        if _ecg_protocol_string(hd5) in {'F30', 'M40'}:
            raise ValueError('Exercise was not ramped.')
        return out
    return new_tff


TMAPS['ecg_bike_aligned_first_r'] = TensorMap(
    'full', shape=(2048, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'mean': 7, 'std': 31}, cacheable=False, metrics=['mse'],
    tensor_from_file=_bike_ecg_first_r_aligned([], [0]),)
TMAPS['ecg_bike_aligned_first_r_noised'] = TensorMap(
    'full', shape=(2048, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'mean': 7, 'std': 31}, cacheable=False, metrics=['mse'],
    tensor_from_file=_bike_ecg_first_r_aligned([_rand_add_noise], [0]),)
TMAPS['ecg_bike_aligned_first_r_normalized'] = TensorMap(
    'full', shape=(2048, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'zero_mean_std1': True}, cacheable=False, metrics=['mse'],
    tensor_from_file=_bike_ecg_first_r_aligned([], [0]),)
TMAPS['ecg_bike_aligned_first_r_noised_normalized'] = TensorMap(
    'full', shape=(2048, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'zero_mean_std1': True}, cacheable=False, metrics=['mse'],
    tensor_from_file=_bike_ecg_first_r_aligned([_rand_add_noise], [0]),)
TMAPS['ecg_bike_aligned_first_r_noised_normalized_downsampled'] = TensorMap(
    'full', shape=(1024, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'zero_mean_std1': True}, cacheable=False, metrics=['mse'],
    tensor_from_file=_bike_ecg_first_r_aligned_downsampled([_rand_add_noise], [0], 4),)
TMAPS['ecg_bike_aligned_first_r_noised_normalized_8xdownsampled'] = TensorMap(
    'full', shape=(512, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'zero_mean_std1': True}, cacheable=False, metrics=['mse'],
    tensor_from_file=_bike_ecg_first_r_aligned_downsampled([_rand_add_noise], [0], 8),)
TMAPS['ecg_bike_shifted_8xdownsampled'] = TensorMap(
    'full', shape=(512, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'mean': 7, 'std': 31}, cacheable=False, metrics=['mse'],
    tensor_from_file=_bike_ecg_shifted_downsampled([0], 8),
    augmentations=[_warp_ecg, _rand_add_noise],
)
TMAPS['ecg_bike_shifted_8xdownsampled_12s'] = TensorMap(
    'full', shape=(750, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'mean': 7, 'std': 31}, cacheable=False, metrics=['mse'],
    tensor_from_file=_bike_ecg_shifted_downsampled([0], 8),
    augmentations=[_warp_ecg, _rand_add_noise],
)
TMAPS['ecg_bike_shifted_8xdownsampled_10s'] = TensorMap(
    'full', shape=(625, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'mean': 7, 'std': 31}, cacheable=False, metrics=['mse'],
    tensor_from_file=_bike_ecg_shifted_downsampled([0], 8),
    augmentations=[_warp_ecg, _rand_add_noise],
)
TMAPS['ecg_bike_shifted_8xdownsampled_10s_heavy_augment'] = TensorMap(
    'full', shape=(625, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'mean': 7, 'std': 31}, cacheable=False, metrics=['mse'],
    tensor_from_file=_bike_ecg_shifted_downsampled([0], 8),
    augmentations=[_rand_roll, _rand_add_noise, _rand_offset],
)
TMAPS['ecg_bike_shifted_8xdownsampled_10s_heavy_augment_2'] = TensorMap(
    'full', shape=(625, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'mean': 7, 'std': 31}, cacheable=False, metrics=['mse'],
    tensor_from_file=_bike_ecg_shifted_downsampled([0], 8),
    augmentations=[_warp_ecg, _rand_roll, _rand_add_noise, _rand_offset],
)
TMAPS['ecg_bike_shifted_8xdownsampled_10s_ramp'] = TensorMap(
    'full', shape=(625, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'mean': 7, 'std': 31}, cacheable=False, metrics=['mse'],
    tensor_from_file=_make_ramp(_bike_ecg_shifted_downsampled([0], 8)),
    augmentations=[_warp_ecg, _rand_add_noise],
)
TMAPS['ecg_bike_shifted_8xdownsampled_10s_heavy_augment_ramp'] = TensorMap(
    'full', shape=(625, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'mean': 7, 'std': 31}, cacheable=False, metrics=['mse'],
    tensor_from_file=_make_ramp(_bike_ecg_shifted_downsampled([0], 8)),
    augmentations=[_rand_roll, _rand_add_noise, _rand_offset],
)
TMAPS['ecg_bike_shifted_8xdownsampled_3lead'] = TensorMap(
    'full', shape=(625, 3), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'mean': 0, 'std': 100}, cacheable=False, metrics=['mse'],
    tensor_from_file=_bike_ecg_shifted_downsampled([0, 1, 2], 8),
    augmentations=[_warp_ecg, _rand_add_noise],
)
TMAPS['ecg_bike_shifted_4xdownsampled_10s'] = TensorMap(
    'full', shape=(1250, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'mean': 7, 'std': 31}, cacheable=False, metrics=['mse'],
    tensor_from_file=_bike_ecg_shifted_downsampled([0], 4),
    augmentations=[_warp_ecg, _rand_add_noise],
)
TMAPS['ecg_bike_shifted_8xdownsampled_3lead_no_warp'] = TensorMap(
    'full', shape=(625, 3), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'mean': 0, 'std': 100}, cacheable=False, metrics=['mse'],
    tensor_from_file=_bike_ecg_shifted_downsampled([0, 1, 2], 8),
    augmentations=[_rand_add_noise],
)
TMAPS['ecg_bike_shifted_8xdownsampled_1lead_no_warp'] = TensorMap(
    'full', shape=(625, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'mean': 0, 'std': 100}, cacheable=False, metrics=['mse'],
    tensor_from_file=_bike_ecg_shifted_downsampled([0], 8),
    augmentations=[_rand_add_noise],
)
TMAPS['ecg_bike_warped_noised_8xdownsampled'] = TensorMap(
    'full', shape=(1024, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'mean': 7, 'std': 31}, cacheable=False, metrics=['mse'],
    tensor_from_file=_bike_ecg_first_r_aligned_downsampled([_warp_ecg, _rand_add_noise], [0], 8),)


TMAPS['ecg_bike_shifted_normalized_8xdownsampled'] = TensorMap(
    'full', shape=(1024, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'zero_mean_std1': True}, cacheable=False, metrics=['mse'],
    tensor_from_file=_bike_ecg_first_r_aligned_downsampled([], [0], 8),)


TMAPS['ecg_bike_warped_noised_normalized_8xdownsampled'] = TensorMap(
    'full', shape=(1024, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'zero_mean_std1': True}, cacheable=False, metrics=['mse'],
    tensor_from_file=_bike_ecg_first_r_aligned_downsampled([_warp_ecg, _rand_add_noise], [0], 8),)


TMAPS['ecg_bike_aligned_shifted'] = TensorMap(
    'full', shape=(2048, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'mean': 7, 'std': 31}, cacheable=False, metrics=['mse'],
    tensor_from_file=_bike_ecg_aligned_augmented([], [0]),)


TMAPS['ecg_bike_normalized_aligned_shifted'] = TensorMap(
    'full', shape=(2048, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'zero_mean_std1': True}, cacheable=False, metrics=['mse'],
    tensor_from_file=_bike_ecg_aligned_augmented([], [0,]),)


TMAPS['ecg_bike_aligned_shifted_warped_noised'] = TensorMap(
    'full', shape=(2048, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'mean': 7, 'std': 31}, cacheable=False, metrics=['mse'],
    tensor_from_file=_bike_ecg_aligned_augmented([_warp_ecg, _rand_add_noise], [0]),)


TMAPS['ecg_bike_aligned_shifted_noised'] = TensorMap(
    'full', shape=(2048, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'mean': 7, 'std': 31}, cacheable=False, metrics=['mse'],
    tensor_from_file=_bike_ecg_aligned_augmented([_rand_add_noise], [0]),)


TMAPS['ecg_bike_normalized_aligned_shifted_noised'] = TensorMap(
    'full', shape=(2048, 1), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'zero_mean_std1': True}, cacheable=False, metrics=['mse'],
    tensor_from_file=_bike_ecg_aligned_augmented([_rand_add_noise], [0,]),)


TMAPS['ecg_bike_normalized_peak_exercise'] = TensorMap(
    'peak_exercise', shape=(2400, 3), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'zero_mean_std1': True}, cacheable=True, metrics=['mse'],
    tensor_from_file=_build_peak_exercise([], [0, 1, 2]),)


TMAPS['ecg_bike_normalized_end_recovery'] = TensorMap(
    'end_recovery', shape=(2400, 3), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'zero_mean_std1': True}, cacheable=True, metrics=['mse'],
    tensor_from_file=_build_end_recovery([], [0, 1, 2]),)


TMAPS['ecg_bike_normalized_warped_noised_peak_exercise'] = TensorMap(
    'peak_exercise', shape=(2400, 3), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'zero_mean_std1': True}, cacheable=False, metrics=['mse'],
    tensor_from_file=_build_peak_exercise([_warp_ecg, _rand_add_noise], [0, 1, 2]),)


TMAPS['ecg_bike_normalized_warped_noised_end_recovery'] = TensorMap(
    'end_recovery', shape=(2400, 3), path_prefix='ecg_bike/float_array', interpretation=Interpretation.CONTINUOUS,
    validator=no_nans, normalization={'zero_mean_std1': True}, cacheable=False, metrics=['mse'],
    tensor_from_file=_build_end_recovery([_warp_ecg, _rand_add_noise], [0, 1, 2]),)


def _ecg_protocol_string(hd5):
    path_prefix, name = 'ecg_bike/string', 'protocol'
    try:
        proto = str(np.array(first_dataset_at_path(hd5, tensor_path(path_prefix, name))))
    except TypeError:
        raise ValueError('Protocol is empty.')
    return proto


def _ecg_protocol(tm: TensorMap, hd5: h5py.File, dependents=None):
    """
    There are 22 possible protocols, M40 - M150, F30 - F140
    """
    out = np.zeros(tm.shape)
    out[tm.channel_map[_ecg_protocol_string(hd5)]] = 1
    return out


def _ecg_protocol_watts(tm: TensorMap, hd5: h5py.File, dependents=None):

    """
    There are 22 possible protocols, M40 - M150, F30 - F140
    """
    return np.array([_ecg_protocol_string(hd5)[1:]], dtype=np.float32)


def _regress_hr_exercise_recovery(hd5: h5py.File):
    times = _get_tensor_at_first_date(hd5, 'ecg_bike/float_array', 'trend_phasetime', handle_nan=_pass_nan)
    hrs = _get_tensor_at_first_date(hd5, 'ecg_bike/float_array', 'trend_heartrate', handle_nan=_pass_nan)
    phases = _get_tensor_at_first_date(hd5, 'ecg_bike/float_array', 'trend_phasename', handle_nan=_pass_nan)
    all_finite = np.isfinite(times) & np.isfinite(hrs) & np.isfinite(phases)
    exercise_mask = (phases == 1) & (hrs > 0) & all_finite
    recovery_mask = (phases == 2) & (hrs > 0) & all_finite

    # acceptable number of measurements
    if np.count_nonzero(exercise_mask) < 10:
        raise ValueError('Not enough exercise measurements.')
    if np.count_nonzero(recovery_mask) < 5:
        raise ValueError('Not enough recovery measurements.')

    return (linregress(times[exercise_mask], hrs[exercise_mask]),
            linregress(times[recovery_mask], hrs[recovery_mask]))


def _get_hrr_measurements(hd5):
    # Phase lengths long enough?
    _check_phase_full_len(hd5, 'pretest')
    _check_phase_full_len(hd5, 'rest')
    exercise_dur = _get_tensor_at_first_date(hd5, 'ecg_bike/continuous', 'exercise_duration')
    min_exercise_dur = 60
    if exercise_dur < min_exercise_dur:
        raise ValueError(f'Exercised less than {min_exercise_dur}s.')

    # get trend measurements
    times = _get_tensor_at_first_date(hd5, 'ecg_bike/float_array', 'trend_phasetime', handle_nan=_pass_nan)
    hrs = _get_tensor_at_first_date(hd5, 'ecg_bike/float_array', 'trend_heartrate', handle_nan=_pass_nan)
    phases = _get_tensor_at_first_date(hd5, 'ecg_bike/float_array', 'trend_phasename', handle_nan=_pass_nan)
    all_finite = np.isfinite(times) & np.isfinite(hrs) & np.isfinite(phases)
    exercise_mask = (phases == 1) & (hrs > 0) & all_finite
    recovery_mask = (phases == 2) & (hrs > 0) & all_finite

    # acceptable number of measurements
    if np.count_nonzero(exercise_mask) < 10:
        raise ValueError('Not enough exercise measurements.')
    if np.count_nonzero(recovery_mask) < 5:
        raise ValueError('Not enough recovery measurements.')

    # linear regression checks
    exercise_reg = linregress(times[exercise_mask], hrs[exercise_mask])
    recovery_reg = linregress(times[recovery_mask], hrs[recovery_mask])
    exercise_slope, exercise_intercept, exercise_r, _, _ = exercise_reg
    recovery_slope, recovery_intercept, recovery_r, _, _ = recovery_reg
    if exercise_r**2 < .45:  # 10% quantile
        raise ValueError('R^2 too low in exercise hr regression.')
    if exercise_slope < 0:
        raise ValueError('Exercise slope negative.')
    if recovery_slope > 0:
        raise ValueError('Recovery slope positive.')

    final_exercise_hr = hrs[exercise_mask][times[exercise_mask] >= exercise_dur - 5].mean()  # mean of last 5s of exercise
    final_recovery_hr = hrs[recovery_mask][times[recovery_mask] >= 60 - 5].mean()  # mean of last 5s of recovery

    if np.isnan(final_exercise_hr):
        raise ValueError('No hr measurements in last 5 seconds of exercise.')
    if np.isnan(final_recovery_hr):
        raise ValueError('No hr measurements in last 5 seconds of recovery.')

    if final_exercise_hr > 220:
        raise ValueError('Max HR too high.')
    if final_recovery_hr < 30:
        raise ValueError('Min HR too low.')

    return final_exercise_hr, final_recovery_hr


def _hrr_measurements_from_raw(hd5):
    _check_phase_full_len(hd5, 'pretest')
    _check_phase_full_len(hd5, 'rest')

    path_prefix, name = 'ecg_bike/float_array', 'full'
    ecg_dataset = first_dataset_at_path(hd5, tensor_path(path_prefix, name))

    exercise_dur = _get_tensor_at_first_date(hd5, 'ecg_bike/continuous', 'exercise_duration')
    exercise_end = int(500 * (15 + exercise_dur - 2.5))  # 2.5 seconds before recovery begins
    recovery_begin = int(500 * (15 + exercise_dur + 2.5))  # 2.5 seconds after recovery begins
    recovery_end = int(500 * (exercise_dur + 55))  # five seconds before the end of recovery
    recovery_final = int(500 * (exercise_dur + 60))  # end of recovery
    at_peak = np.array(ecg_dataset[exercise_end: recovery_begin, 0], dtype=np.float32)
    at_end = np.array(ecg_dataset[recovery_end: recovery_end, 0], dtype=np.float32)
    _, _, _, _, _, _, peak_hrs = biosppy.signals.ecg.ecg(at_peak, sampling_rate=500, show=False)
    _, _, _, _, _, _, end_hrs = biosppy.signals.ecg.ecg(at_end, sampling_rate=500, show=False)
    final_exercise_hr = _fail_nan(np.median(peak_hrs))
    final_recovery_hr = _fail_nan(np.median(end_hrs))
    if final_exercise_hr > 220:
        raise ValueError('Max HR too high.')
    if final_recovery_hr < 30:
        raise ValueError('Min HR too low.')
    return final_exercise_hr, final_recovery_hr


def _hrr_qc(tm: TensorMap, hd5: h5py.File, dependents=None):
    final_exercise_hr, final_recovery_hr = _get_hrr_measurements(hd5)
    return np.array(final_exercise_hr - final_recovery_hr)


def _hrr_raw(tm: TensorMap, hd5: h5py.File, dependents=None):
    final_exercise_hr, final_recovery_hr = _hrr_measurements_from_raw(hd5)
    return np.array(final_exercise_hr - final_recovery_hr)


def _max_hr_qc(tm: TensorMap, hd5: h5py.File, dependents=None):
    final_exercise_hr, final_recovery_hr = _get_hrr_measurements(hd5)
    return np.array(final_exercise_hr)


def _recovery_hr_qc(tm: TensorMap, hd5: h5py.File, dependents=None):
    final_exercise_hr, final_recovery_hr = _get_hrr_measurements(hd5)
    return np.array(final_recovery_hr)


def _build_noised_tensor_from_file(file_name: str, target_column: str, noise_column: str, normalization: bool = False, delimiter: str = '\t'):
    """
    Build a tensor_from_file function from a column in a file.
    Only works for continuous values.
    When normalization is True values will be normalized according to the mean and std of all of the values in the column.
    """
    error = None
    try:
        with open(file_name, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader)
            index = header.index(target_column)
            noise_index = header.index(noise_column)
            table = {}
            fail_count = 0
            for row in reader:
                try:
                    table[row[0]] = np.array([float(row[index])])
                    table[f'{row[0]}_std'] = np.array([float(row[noise_index])])
                except ValueError:
                    fail_count += 1
                    continue
            logging.info(f'{fail_count} samples failed out of {len(table) + fail_count} in {file_name}.')
            if normalization:
                value_array = np.array([sub_array[0] for sub_array in table.values()])
                mean = value_array.mean()
                std = value_array.std()
                logging.info(f'Normalizing TensorMap from file {file_name}, column {target_column} with mean: '
                             f'{mean:.2f}, std: {std:.2f}')
    except FileNotFoundError as e:
        error = e

    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        if error:
            raise error
        if normalization:
            tm.normalization = {'mean': mean, 'std': std}
        try:
            key = os.path.basename(hd5.filename).replace('.hd5', '')
            return table[key].copy() + np.random.randn() * table[f'{key}_std']
        except KeyError:
            raise KeyError(f'User id not in file {file_name}.')
    return tensor_from_file


# FOR JEN
TMAPS['ecg-bike-pretest-duration'] = TensorMap('pretest_duration', path_prefix='ecg_bike/continuous', loss='logcosh', metrics=['mae'], shape=(1,),
                                               normalization={'mean': 0, 'std': 1},
                                               tensor_from_file=normalized_first_date, interpretation=Interpretation.CONTINUOUS)
TMAPS['ecg-bike-exercise-duration'] = TensorMap('exercise_duration', path_prefix='ecg_bike/continuous', loss='logcosh', metrics=['mae'], shape=(1,),
                                                normalization={'mean': 0, 'std': 1},
                                                tensor_from_file=normalized_first_date, interpretation=Interpretation.CONTINUOUS)
TMAPS['ecg-bike-recovery-duration'] = TensorMap('rest_duration', path_prefix='ecg_bike/continuous', loss='logcosh', metrics=['mae'], shape=(1,),
                                                normalization={'mean': 0, 'std': 1},
                                                tensor_from_file=normalized_first_date, interpretation=Interpretation.CONTINUOUS)
TMAPS['ecg-bike-max-pred-hr'] = TensorMap('max_pred_hr', path_prefix='ecg_bike/continuous', loss='logcosh', metrics=['mae'], shape=(1,),
                                          normalization={'mean': 0, 'std': 1},
                                          tensor_from_file=normalized_first_date, interpretation=Interpretation.CONTINUOUS)
TMAPS['ecg-bike-resting-hr'] = TensorMap('resting_hr', path_prefix='ecg_bike/continuous', loss='logcosh', metrics=['mae'], shape=(1,),
                                         normalization={'mean': 0, 'std': 1},
                                         tensor_from_file=normalized_first_date, interpretation=Interpretation.CONTINUOUS)
TMAPS['ecg-bike-age'] = TensorMap('age', path_prefix='ecg_bike/continuous', loss='logcosh', metrics=['mae'], shape=(1,),
                                  normalization={'mean': 0, 'std': 1},
                                  interpretation=Interpretation.CONTINUOUS)
TMAPS['ecg-bike-protocol'] = TensorMap('protocol', path_prefix='ecg_bike/continuous', shape=(22,), tensor_from_file=_ecg_protocol,
                                       interpretation=Interpretation.CATEGORICAL,
                                       channel_map={
                                           **{f'M{i[1]}': i[0] for i in enumerate(range(40, 150, 10))},
                                           **{f'F{i[1]}': i[0] + 11 for i in enumerate(range(30, 140, 10))},
                                       })
TMAPS['ecg-bike-protocol-watts'] = TensorMap('protocol', path_prefix='ecg_bike/continuous', shape=(1,), tensor_from_file=_ecg_protocol_watts,
                                       interpretation=Interpretation.CONTINUOUS, normalization={'mean': 82, 'std': 24},)
# trend measurements
TMAPS['ecg-bike-trend-hr'] = TensorMap('trend_heartrate', shape=(120, 1), path_prefix='ecg_bike/float_array',
                                       normalization={'mean': 0, 'std': 1},
                                       tensor_from_file=normalized_first_date, interpretation=Interpretation.CONTINUOUS)
TMAPS['ecg-bike-trend-load'] = TensorMap('trend_load', shape=(120, 1), path_prefix='ecg_bike/float_array',
                                         normalization={'mean': 0, 'std': 1},
                                         tensor_from_file=normalized_first_date, interpretation=Interpretation.CONTINUOUS)
TMAPS['ecg-bike-trend-grade'] = TensorMap('trend_grade', shape=(120, 1), path_prefix='ecg_bike/float_array',
                                          normalization={'mean': 0, 'std': 1},
                                          tensor_from_file=normalized_first_date, interpretation=Interpretation.CONTINUOUS)
TMAPS['ecg-bike-trend-time'] = TensorMap('trend_time', shape=(120, 1), path_prefix='ecg_bike/float_array',
                                         normalization={'mean': 0, 'std': 1},
                                         tensor_from_file=normalized_first_date, interpretation=Interpretation.CONTINUOUS)
TMAPS['ecg-bike-trend-phasename'] = TensorMap('trend_phasename', shape=(120, 1), path_prefix='ecg_bike/float_array',
                                              normalization={'mean': 0, 'std': 1},
                                              tensor_from_file=normalized_first_date, interpretation=Interpretation.CONTINUOUS)
TMAPS['ecg-bike-trend-phasetime'] = TensorMap('trend_phasetime', shape=(120, 1), path_prefix='ecg_bike/float_array',
                                              normalization={'mean': 0, 'std': 1},
                                              tensor_from_file=normalized_first_date, interpretation=Interpretation.CONTINUOUS)
TMAPS['ecg-bike-trend-artifact'] = TensorMap('trend_artifact', shape=(120, 1), path_prefix='ecg_bike/float_array',
                                             normalization={'mean': 0, 'std': 1},
                                             tensor_from_file=normalized_first_date, interpretation=Interpretation.CONTINUOUS)

# HRR FINAL
TMAPS['ecg-bike-afib'] = TensorMap('afib', shape=(2,),
                                   interpretation=Interpretation.CATEGORICAL, channel_map={'no_afib': 1, 'afib': 0},
                                   tensor_from_file=_build_tensor_from_sample_id_file('/home/ndiamant/exercise_afib.csv', delimiter=',', id_column='sample_id'))
TMAPS['ecg-bike-QT'] = TensorMap('qt', shape=(1,), interpretation=Interpretation.CONTINUOUS,
    tensor_from_file=_build_tensor_from_file('/home/ndiamant/inference_ecg_bike_1st__all_intervals_sentinel_strip_I.tsv', 'qt-interval-sentinel_prediction'))
TMAPS['ecg-bike-TP'] = TensorMap('tp', shape=(1,), interpretation=Interpretation.CONTINUOUS,
    tensor_from_file=_build_tensor_from_file('/home/ndiamant/tp.tsv', 'tp'))
TMAPS['ecg-bike-ensemble-TP'] = TensorMap('tp', shape=(1,), interpretation=Interpretation.CONTINUOUS,
    tensor_from_file=_build_tensor_from_file('/home/ndiamant/3_lead_ensemble.tsv', 'tp_ensemble'))
TMAPS['ecg-bike-cigarettes'] = TensorMap('cigarettes', shape=(2,),
                                         interpretation=Interpretation.CATEGORICAL, channel_map={'no_cig': 0, 'cig': 1},
                                         tensor_from_file=_build_tensor_from_sample_id_file('/home/ndiamant/20116_0_or_1.csv', delimiter=',', id_column='sample_id'))
TMAPS['ecg-bike-bb'] = TensorMap('beta_blockers', shape=(2,),
                                 interpretation=Interpretation.CATEGORICAL, channel_map={'no_bb': 0, 'bb': 1},
                                 tensor_from_file=_build_tensor_from_sample_id_file('/home/ndiamant/beta_blockers.csv', delimiter=',', id_column='sample_id'))
TMAPS['ecg-bike-hrr-raw-file'] = TensorMap('hrr', loss='logcosh', metrics=['mae'], shape=(1,),
                                           normalization={'mean': 25, 'std': 15},
                                           interpretation=Interpretation.CONTINUOUS,
                                           validator=make_range_validator(0, 110),
                                           tensor_from_file=_build_tensor_from_file('/home/ndiamant/raw_hrr.csv', 'hrr', delimiter=','))
TMAPS['ecg-bike-hrr-raw-ensemble'] = TensorMap('hrr', loss='logcosh', metrics=['mae'], shape=(1,),
                                               normalization={'mean': 25, 'std': 15},
                                               interpretation=Interpretation.CONTINUOUS,
                                               validator=make_range_validator(0, 110),
                                               tensor_from_file=_build_tensor_from_file('/home/ndiamant/ensemble_hrr.csv', 'hrr', delimiter=','))
TMAPS['ecg-bike-hrr-10s-ramp'] = TensorMap('hrr', loss='logcosh', metrics=['mae'], shape=(1,),
                                           normalization={'mean': 12.6, 'std': 8.4},
                                           interpretation=Interpretation.CONTINUOUS,
                                           validator=make_range_validator(0, 42),
                                           tensor_from_file=_make_ramp(_build_tensor_from_file('/home/ndiamant/filtered_hrr10_phenotype.tsv', 'hrr')))
TMAPS['ecg-bike-peak-hr-ramp'] = TensorMap('peak_hr', loss='logcosh', metrics=['mae'], shape=(1,),
                                           normalization={'mean': 113, 'std': 15},
                                           interpretation=Interpretation.CONTINUOUS,
                                           validator=make_range_validator(40, 220),
                                           tensor_from_file=_make_ramp(_build_tensor_from_file('/home/ndiamant/filtered_hrr10_phenotype.tsv', 'peak_hr')))
TMAPS['ecg-bike-peak-hr'] = TensorMap('peak_hr', loss='logcosh', metrics=['mae'], shape=(1,),
                                           normalization={'mean': 113, 'std': 15},
                                           interpretation=Interpretation.CONTINUOUS,
                                           validator=make_range_validator(40, 220),
                                           tensor_from_file=_build_tensor_from_file('/home/ndiamant/filtered_hrr10_phenotype.tsv', 'peak_hr'))
def _hr_achieved(tm: TensorMap, hd5: h5py.File, dependents=None):
    peak_hr_tmap = TMAPS['ecg-bike-peak-hr']
    peak_hr = peak_hr_tmap.tensor_from_file(peak_hr_tmap, hd5)
    age_tmap = TMAPS['ecg-bike-age']
    age = age_tmap.tensor_from_file(age_tmap, hd5)
    return peak_hr / (220 - age)

TMAPS['ecg-bike-hr-achieved-ramp'] = TensorMap(
    'hr_achieved', loss='logcosh', metrics=['mae'], shape=(1,),
    interpretation=Interpretation.CONTINUOUS,
    tensor_from_file=_make_ramp(_hr_achieved),
)
TMAPS['ecg-bike-hr-achieved'] = TensorMap(
    'hr_achieved', loss='logcosh', metrics=['mae'], shape=(1,),
    interpretation=Interpretation.CONTINUOUS,
    tensor_from_file=_hr_achieved,
)


def constant_tff(constant: float):
    def tff(*args, **kwargs):
        return np.array([constant])
    return tff


TMAPS['ecg-bike-hr-achieved-75'] = TensorMap(
    'hr_achieved', loss='logcosh', metrics=['mae'], shape=(1,),
    interpretation=Interpretation.CONTINUOUS,
    tensor_from_file=constant_tff(.75),
)
TMAPS['ecg-bike-hr-achieved-85'] = TensorMap(
    'hr_achieved', loss='logcosh', metrics=['mae'], shape=(1,),
    interpretation=Interpretation.CONTINUOUS,
    tensor_from_file=constant_tff(.85),
)



TMAPS['ecg-bike-hrr-50s-ramp'] = TensorMap('hrr', loss='logcosh', metrics=['mae'], shape=(1,),
                                           normalization={'mean': 27.6, 'std': 10.9},
                                           interpretation=Interpretation.CONTINUOUS,
                                           validator=make_range_validator(0, 100),
                                           tensor_from_file=_make_ramp(_build_tensor_from_file('/home/ndiamant/hrr50_phenotype.tsv', 'hrr50')))
TMAPS['ecg-bike-hrr-10s-ramp-less-noise'] = TensorMap('hrr', loss='logcosh', metrics=['mae'], shape=(1,),
                                                      normalization={'mean': 12.6, 'std': 8.4},
                                                      interpretation=Interpretation.CONTINUOUS,
                                                      validator=make_range_validator(0, 42),
                                                      tensor_from_file=_make_ramp(_build_tensor_from_file('/home/ndiamant/hrr10s_phenotype.tsv', 'hrr')))
TMAPS['ecg-bike-hrr-10s-quality'] = TensorMap('hrr_quality', loss='logcosh', metrics=['mae'], shape=(1,),
                                              interpretation=Interpretation.CONTINUOUS,
                                              validator=make_range_validator(0, 1),
                                              tensor_from_file=_build_tensor_from_file('/home/ndiamant/filtered_hrr10_phenotype.tsv', 'good_hrr_prob'))
TMAPS['ecg-bike-hrr-incomplete_exercise'] = TensorMap(
    'hrr_incomplete', loss='logcosh', metrics=['mae'], shape=(1,),
    normalization={'mean': 25, 'std': 15},
    interpretation=Interpretation.CONTINUOUS,
    validator=make_range_validator(0, 110),
    tensor_from_file=_make_incomplete_exercise(_build_tensor_from_file('/home/ndiamant/ensemble_hrr.csv', 'hrr', delimiter=',')))
TMAPS['ecg-bike-hrr-ramp'] = TensorMap(
    'hrr', loss='logcosh', metrics=['mae'], shape=(1,),
    normalization={'mean': 25, 'std': 15},
    interpretation=Interpretation.CONTINUOUS,
    validator=make_range_validator(0, 110),
    tensor_from_file=_make_ramp(_build_tensor_from_file('/home/ndiamant/ensemble_hrr.csv', 'hrr', delimiter=',')))
TMAPS['ecg-bike-hrr-ramp-transfer'] = TensorMap(
    'hrr_transfer', loss='logcosh', metrics=['mae'], shape=(1,),
    normalization={'mean': 25, 'std': 15},
    interpretation=Interpretation.CONTINUOUS,
    validator=make_range_validator(0, 110),
    tensor_from_file=_build_tensor_from_file('/home/ndiamant/ensemble_hrr.csv', 'hrr', delimiter=','))
TMAPS['ecg-bike-hrr-raw-ensemble-noised'] = TensorMap('hrr', loss='logcosh', metrics=['mae'], shape=(1,),
                                                      normalization={'mean': 25, 'std': 15}, cacheable=False,
                                                      interpretation=Interpretation.CONTINUOUS,
                                                      validator=make_range_validator(0, 110),
                                                      tensor_from_file=_build_noised_tensor_from_file('/home/ndiamant/ensemble_hrr.csv', 'hrr', 'hrr_std', delimiter=','))
TMAPS['ecg-bike-ensemble-log-squared-error'] = TensorMap('log_squared_error', loss='logcosh', metrics=['mae'], shape=(1,),
                                                         normalization={'mean': 2.66, 'std': 2.26},
                                                         interpretation=Interpretation.CONTINUOUS,
                                                         tensor_from_file=_build_tensor_from_file('/home/ndiamant/hrr_squared_errors.csv', 'log_squared_error', delimiter=','))
TMAPS['ecg-bike-hrr-raw-ensemble-binary'] = TensorMap(
    'hrr', interpretation=Interpretation.DISCRETIZED, validator=make_range_validator(0, 110), channel_map={'hrr': 0},
    discretization_bounds=[25,],
    tensor_from_file=_build_tensor_from_file('/home/ndiamant/ensemble_hrr.csv', 'hrr', delimiter=','))
TMAPS['ecg-bike-hrr-raw-ensemble-binary-perfect-cut'] = TensorMap(
    'hrr', interpretation=Interpretation.DISCRETIZED, validator=make_range_validator(0, 110), channel_map={'hrr': 0},
    discretization_bounds=[28.4,],
    tensor_from_file=_build_tensor_from_file('/home/ndiamant/ensemble_hrr.csv', 'hrr', delimiter=','))
TMAPS['ecg-bike-hrr-raw-ensemble-trinary'] = TensorMap(
    'hrr', interpretation=Interpretation.DISCRETIZED, validator=make_range_validator(0, 110), channel_map={'hrr': 0},
    discretization_bounds=[20, 40],
    tensor_from_file=_build_tensor_from_file('/home/ndiamant/ensemble_hrr.csv', 'hrr', delimiter=','))
TMAPS['ecg-bike-hrr-raw-ensemble-trinary-ramp'] = TensorMap(
    'hrr', interpretation=Interpretation.DISCRETIZED, validator=make_range_validator(0, 110), channel_map={'hrr': 0},
    discretization_bounds=[20, 40],
    tensor_from_file=_make_ramp(_build_tensor_from_file('/home/ndiamant/ensemble_hrr.csv', 'hrr', delimiter=',')))
TMAPS['ecg-bike-hrr-raw-ensemble-trinary-weighted'] = TensorMap(
    'hrr', interpretation=Interpretation.DISCRETIZED, validator=make_range_validator(0, 110), channel_map={'hrr': 0},
    discretization_bounds=[20, 40],
    loss=weighted_crossentropy([4., 1., 1.], 'hrr_trinary'),
    tensor_from_file=_build_tensor_from_file('/home/ndiamant/ensemble_hrr.csv', 'hrr', delimiter=','))
TMAPS['ecg-bike-hrr-raw-ensemble-discretized-not-weighted'] = TensorMap(
    'hrr', interpretation=Interpretation.DISCRETIZED, validator=make_range_validator(0, 110), channel_map={'hrr': 0},
    discretization_bounds=[20, 25, 30, 40],
    tensor_from_file=_build_tensor_from_file('/home/ndiamant/ensemble_hrr.csv', 'hrr', delimiter=','))
TMAPS['ecg-bike-hrr-raw-ensemble-discretized'] = TensorMap(
    'hrr', interpretation=Interpretation.DISCRETIZED, validator=make_range_validator(0, 110), channel_map={'hrr': 0},
    discretization_bounds=[20, 25, 30, 40],
    loss=weighted_crossentropy([4.151638434795473, 5.0337219730941705, 4.877128953771289, 3.7263311645199844, 11.48475547370575], 'hrr'),
    tensor_from_file=_build_tensor_from_file('/home/ndiamant/ensemble_hrr.csv', 'hrr', delimiter=','))


# TMAPS['ecg-bike-hrr-raw-file-4th-order'] = TensorMap('hrr', loss=mean_quartic_error, metrics=['mae', 'logcosh'], shape=(1,),
#                                            normalization={'mean': 25, 'std': 15},
#                                            interpretation=Interpretation.CONTINUOUS,
#                                            validator=make_range_validator(0, 110),
#                                            tensor_from_file=_build_tensor_from_file('/home/ndiamant/raw_hrr.csv', 'hrr', delimiter=','))
TMAPS['ecg-bike-hrr'] = TensorMap('hrr', loss='logcosh', metrics=['mae'], shape=(1,),
                                  normalization={'mean': 25, 'std': 15},
                                  interpretation=Interpretation.CONTINUOUS,
                                  tensor_from_file=_hrr_qc)
TMAPS['ecg-bike-hrr-trend'] = TensorMap('hrr_trend', loss='logcosh', metrics=['mae'], shape=(1,),
                                  normalization={'mean': 25, 'std': 15},
                                  interpretation=Interpretation.CONTINUOUS,
                                  tensor_from_file=_hrr_qc)
# TMAPS['ecg-bike-hrr-sqrt-loss'] = TensorMap('hrr', metrics=['mae', 'logcosh'], shape=(1,),
#                                             normalization={'mean': 25, 'std': 15},
#                                             interpretation=Interpretation.CONTINUOUS, loss=sqrt_error,
#                                             tensor_from_file=_hrr_qc)
TMAPS['ecg-bike-hrr-mse'] = TensorMap('hrr', metrics=['mae', 'logcosh'], shape=(1,),
                                      normalization={'mean': 25, 'std': 15},
                                      interpretation=Interpretation.CONTINUOUS, loss='mse',
                                      tensor_from_file=_hrr_qc)
TMAPS['ecg-bike-hrr-raw'] = TensorMap('hrr', loss='logcosh', metrics=['mae'], shape=(1,),
                                      normalization={'mean': 25, 'std': 15},
                                      interpretation=Interpretation.CONTINUOUS,
                                      tensor_from_file=_hrr_raw)
TMAPS['ecg-bike-hrr-raw-no-norm'] = TensorMap('hrr', loss='logcosh', metrics=['mae'], shape=(1,),
                                              interpretation=Interpretation.CONTINUOUS,
                                              tensor_from_file=_hrr_raw)
TMAPS['ecg-bike-max-hr-qc'] = TensorMap('max_hr', loss='logcosh', metrics=['mae'], shape=(1,),
                                        normalization={'mean': 110, 'std': 15},  # TODO: get actual numbers
                                        interpretation=Interpretation.CONTINUOUS,
                                        tensor_from_file=_max_hr_qc)
TMAPS['ecg-bike-last-hr-qc'] = TensorMap('last_hr', loss='logcosh', metrics=['mae'], shape=(1,),
                                         normalization={'mean': 85, 'std': 20},  # TODO: get actual numbers
                                         interpretation=Interpretation.CONTINUOUS,
                                         tensor_from_file=_recovery_hr_qc)
TMAPS['ecg-bike-hrr-parented'] = TensorMap('hrr', loss='logcosh', metrics=['mae'], shape=(1,),
                                           normalization={'mean': 25, 'std': 15},
                                           interpretation=Interpretation.CONTINUOUS,
                                           tensor_from_file=_hrr_qc,
                                           parents=[TMAPS['ecg-bike-last-hr-qc'], TMAPS['ecg-bike-max-hr-qc']])


# Transfer learning to rest ECGs
def _make_ecg_rest_downsampled(rate: float):
    def ecg_rest_from_file(tm, hd5, dependents=None):
        length = int(tm.shape[0] * rate)
        start = 0 if length == 5000 else np.random.randint(5000 - length)
        ecg = np.array(hd5[tm.path_prefix][start: start + length])[:, np.newaxis]
        ecg = _downsample(ecg, rate)
        return ecg
    return ecg_rest_from_file


TMAPS['ecg_rest_shifted_8xdownsampled'] = TensorMap('full_rest', Interpretation.CONTINUOUS, shape=(625, 1), path_prefix='ecg_rest/strip_I',
                                                    tensor_from_file=_make_ecg_rest_downsampled(8),
                                                    normalization={'mean': 17.5, 'std': 47.8}, cacheable=False,
                                                    augmentations=[_warp_ecg, _rand_add_noise, _rand_roll])
TMAPS['ecg_rest_shifted_8xdownsampled_transfer'] = TensorMap('full_rest', Interpretation.CONTINUOUS, shape=(625, 1), path_prefix='ukb_ecg_rest/strip_I/instance_0',
                                                             tensor_from_file=_make_ecg_rest_downsampled(8),
                                                             normalization={'mean': 17.5, 'std': 47.8}, cacheable=False,
                                                             augmentations=[_warp_ecg, _rand_add_noise, _rand_roll])
TMAPS['ecg_rest_age_transfer'] = TensorMap('21003_Age-when-attended-assessment-centre_2_0', Interpretation.CONTINUOUS, path_prefix='continuous', loss='logcosh', validator=make_range_validator(1, 120),
                           normalization=None, channel_map={'21003_Age-when-attended-assessment-centre_2_0': 0, })


def _hr_achieved_transfer(tm: TensorMap, hd5: h5py.File, dependents=None):
    peak_hr_tmap = TMAPS['ecg-bike-peak-hr']
    peak_hr = peak_hr_tmap.tensor_from_file(peak_hr_tmap, hd5)
    age_tmap = TMAPS['ecg_rest_age_transfer']
    age = age_tmap.tensor_from_file(age_tmap, hd5)
    return peak_hr / (220 - age)


TMAPS['ecg-bike-hr-achieved-transfer'] = TensorMap(
    'hr_achieved', loss='logcosh', metrics=['mae'], shape=(1,),
    interpretation=Interpretation.CONTINUOUS,
    tensor_from_file=_hr_achieved_transfer,
)


BIOSPPY_SENTINEL = -1000
BIOSPPY_DIFF_CUTOFF = 5
def _hr_biosppy(file_name: str, time: int, hrr=False):
    error = None
    try:
        df = pd.read_csv(file_name, dtype={'sample_id': str})
        df = df.set_index('sample_id')
    except FileNotFoundError as e:
        error = e
    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        if error:
            raise error
        sample_id = os.path.basename(hd5.filename).replace('.hd5', '')
        try:
            row = df.loc[sample_id]
            hr, diff = row[f'{time}_hr'], row[f'{time}_diff']
            if diff > BIOSPPY_DIFF_CUTOFF:
                out = BIOSPPY_SENTINEL
            if hrr:
                peak, peak_diff = row[f'0_hr'], row[f'0_diff']
                if peak_diff > BIOSPPY_DIFF_CUTOFF:
                    out = BIOSPPY_SENTINEL
                out = peak - hr
            else:
                out = hr
            if np.isnan(out):
                out = BIOSPPY_SENTINEL
            return np.array([out])
        except KeyError:
            raise KeyError('sample id not in data')
    return tensor_from_file


def _make_hr_biosppy_tmaps():
    for time in 0, 10, 20, 30, 40, 50:
        TMAPS[f'ecg-bike-{time}_hr'] = TensorMap(
            f'{time}_hr', metrics=['mae', pearson], shape=(1,),
            interpretation=Interpretation.CONTINUOUS,
            sentinel=BIOSPPY_SENTINEL,
            tensor_from_file=_hr_biosppy('/home/ndiamant/biosppy_hr_recovery_measurements.csv', time),
        )
    for time in 0, 10, 20, 30, 40, 50:
        TMAPS[f'ecg-bike-{time}_hrr'] = TensorMap(
            f'{time}_hrr', metrics=['mae', pearson], shape=(1,),
            interpretation=Interpretation.CONTINUOUS,
            sentinel=BIOSPPY_SENTINEL,
            tensor_from_file=_hr_biosppy('/home/ndiamant/biosppy_hr_recovery_measurements.csv', time, hrr=True),
            parents=[TMAPS[f'ecg-bike-{time}_hr'], TMAPS['ecg-bike-0_hr']],
        )


_make_hr_biosppy_tmaps()


def _hr_5_second_segment(file_name: str, col: str, delimiter: str = ','):
    error = None
    try:
        df = pd.read_csv(file_name, delimiter=delimiter, dtype={'sample_id': str})
        df = df.set_index('sample_id')
    except FileNotFoundError as e:
        error = e
    path_prefix, name = 'ecg_bike/float_array', 'full'

    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        _check_phase_full_len(hd5, 'pretest')
        _check_phase_full_len(hd5, 'rest')
        if error:
            raise error
        sample_id = os.path.basename(hd5.filename).replace('.hd5', '')
        try:
            row = df.loc[sample_id].sample(1)
            if col == 'bad_measure':
                return np.array([0, 1]) if row[col] else np.array([1, 0])  # label 1 if bad measure
            elif col in {'hr', 'bpm'}:
                return np.array(row[col])
            elif col == 'start_index':
                index = int(row[col])
                ecg_dataset = first_dataset_at_path(hd5, tensor_path(path_prefix, name))
                offset = np.random.randint(-10, 10)
                ecg_dataset = ecg_dataset[index + offset: 500 * 5 + index + offset]
                if len(ecg_dataset) < 2500:
                    raise ValueError('ECG segment too short.')
                if tm.shape[0] != len(ecg_dataset):
                    ecg_dataset = _downsample(ecg_dataset, len(ecg_dataset) // tm.shape[0])
                return ecg_dataset
            else:
                raise ValueError(f'Col {col} not in file.')
        except KeyError:
            raise KeyError('sample id not in data')
    return tensor_from_file


TMAPS['ecg-bike-hr-segment'] = TensorMap(
    'hr', loss='logcosh', metrics=['mae'], shape=(1,), normalization=Standardize(99, 20), interpretation=Interpretation.CONTINUOUS,
    tensor_from_file=_hr_5_second_segment('/home/ndiamant/hr_quality_4.csv', 'hr'),
)
TMAPS['ecg-bike-quality-segment'] = TensorMap(
    'quality', shape=(2,), interpretation=Interpretation.CATEGORICAL, channel_map={'high_quality': 0, 'low_quality': 1},
    tensor_from_file=_hr_5_second_segment('/home/ndiamant/hr_quality_4.csv', 'bad_measure'),
)
TMAPS['ecg-bike-segment'] = TensorMap(
    'ecg_segment', shape=(2500, 3), interpretation=Interpretation.CONTINUOUS, cacheable=False,
    tensor_from_file=_hr_5_second_segment('/home/ndiamant/hr_quality_4.csv', 'start_index'), normalization=ZeroMeanStd1(),
    augmentations=[_rand_offset, _rand_add_noise]
)
TMAPS['ecg-bike-bpm-segment'] = TensorMap(
    'bpm', loss='logcosh', metrics=['mae'], shape=(1,), normalization=Standardize(94, 17), interpretation=Interpretation.CONTINUOUS,
    tensor_from_file=_hr_5_second_segment('/home/ndiamant/hr_training_data.csv', 'bpm'), cacheable=False,
)
TMAPS['ecg-bike-segment-for-bpm'] = TensorMap(
    'ecg_segment', shape=(2500, 3), interpretation=Interpretation.CONTINUOUS, cacheable=False,
    tensor_from_file=_hr_5_second_segment('/home/ndiamant/hr_training_data.csv', 'start_index'), normalization=ZeroMeanStd1(),
    augmentations=[_rand_offset, _rand_add_noise]
)
TMAPS['ecg-bike-segment-for-bpm-down4x'] = TensorMap(
    'ecg_segment', shape=(625, 3), interpretation=Interpretation.CONTINUOUS, cacheable=False,
    tensor_from_file=_hr_5_second_segment('/home/ndiamant/hr_training_data.csv', 'start_index'), normalization=ZeroMeanStd1(),
    augmentations=[_rand_offset, _rand_add_noise]
)


### End HRR ###


def _segmented_index_slices(key_prefix: str, shape: Tuple[int], path_prefix: str='ukb_cardiac_mri') -> Callable:
    """Get semantic segmentation with label index as pixel values for an MRI slice"""
    def _segmented_dicom_tensor_from_file(tm, hd5, dependents={}):
        tensor = np.zeros(shape, dtype=np.float32)
        for i in range(shape[-1]):
            categorical_index_slice = _get_tensor_at_first_date(hd5, path_prefix, key_prefix + str(i + 1))
            tensor[..., i] = _pad_or_crop_array_to_shape(shape[:-1], categorical_index_slice)
        return tensor
    return _segmented_dicom_tensor_from_file


def _bounding_box_from_categorical(segmented_shape: Tuple[int], segmented_key: str, class_index: int) -> Callable:
    """Given an hd5 key of a semantic segmentation return a bounding box that covers the extent of a given class
    :param segmented_shape: The shape of each segmentation
    :param segmented_key: The key for the HD5 file where the segmentation is stored
    :param class_index: The index in the segmentation asssociated with the class we will find the bounding box for
    :return: A np.ndarray encoding a bounding box with min coordinates followed by max coordinates
            For example, a 2D bounding box will be returned as a 1D tensor of 4 numbers: [min_x, min_y, max_x, max_y]
            for a 3d bounding box we would get 6 numbers: [min_x, min_y, min_z max_x, max_y, max_z]
    """
    def bbox_from_file(tm, hd5, dependents={}):
        tensor = np.zeros(tm.shape, dtype=np.float32)
        index_tensor = _pad_or_crop_array_to_shape(segmented_shape, np.array(hd5[segmented_key], dtype=np.float32))
        bitmask = np.where(index_tensor == class_index)
        total_axes = tm.shape[-1] // 2  # Divide by 2 because we need min and max for each axis
        for i in range(total_axes):
            tensor[i] = np.min(bitmask[i])
            tensor[i+total_axes] = np.max(bitmask[i])
        return tensor
    return bbox_from_file


def _bounding_box_from_callable(class_index: int, tensor_from_file_fxn: Callable) -> Callable:
    """Given a tensor_from_file function that returns a semantic segmentation find the bounding box that covers the extent of a given class
    :param class_index: The index in the segmentation asssociated with the class we will find the bounding box for
    :param tensor_from_file_fxn: A tensor from file function that returns a class index tensor.
            This tensor should NOT be one hot, ie the segmentation before `to_categorical` has been applied.
    :return: A np.ndarray encoding a bounding box with min coordinates followed by max coordinates
            For example, a 2D bounding box will be returned as a 1D tensor of 4 numbers: [min_x, min_y, max_x, max_y]
            for a 3d bounding box we would get 6 numbers: [min_x, min_y, min_z max_x, max_y, max_z]
    """
    def bbox_from_file(tm, hd5, dependents={}):
        tensor = np.zeros(tm.shape, dtype=np.float32)
        index_tensor = tensor_from_file_fxn(None, hd5)
        bitmask = np.where(index_tensor == class_index)
        total_axes = tm.shape[-1] // 2  # Divide by 2 because we need min and max for each axis
        for i in range(total_axes):
            tensor[i] = np.min(bitmask[i])
            tensor[i+total_axes] = np.max(bitmask[i])
        return tensor
    return bbox_from_file


def _bounding_box_channel_map(total_axes: int) -> Dict[str, int]:
    channel_map = {}
    for i in range(total_axes):
        channel_map[f'min_axis_{i}'] = i
        channel_map[f'max_axis_{i}'] = i + total_axes
    return channel_map


TMAPS['lax_3ch_lv_cavity_bbox_slice0'] = TensorMap('lax_3ch_lv_cavity_bbox_slice0', Interpretation.MESH, shape=(4,),
                                                   tensor_from_file=_bounding_box_from_categorical((160, 160), 'ukb_cardiac_mri/cine_segmented_lax_3ch_annotated_1/instance_0', MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['LV_Cavity']),
                                                   channel_map=_bounding_box_channel_map(2))
TMAPS['lax_3ch_left_atrium_bbox_slice0'] = TensorMap('lax_3ch_left_atrium_bbox_slice0', Interpretation.MESH, shape=(4,),
                                                     tensor_from_file=_bounding_box_from_categorical((160, 160), 'ukb_cardiac_mri/cine_segmented_lax_3ch_annotated_1/instance_0', MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['left_atrium']),
                                                     channel_map=_bounding_box_channel_map(2))

aorta_descending_tff = _bounding_box_from_categorical((192, 224), 'ukb_cardiac_mri/cine_segmented_ao_dist_annotated_1/instance_0', MRI_AO_SEGMENTED_CHANNEL_MAP['descending_aorta'])
TMAPS['cine_segmented_ao_descending_aorta_bbox_slice0'] = TensorMap('cine_segmented_ao_descending_aorta_bbox_slice0', Interpretation.MESH, shape=(4,),
                                                                    tensor_from_file=aorta_descending_tff, channel_map=_bounding_box_channel_map(2))
aorta_ascending_tff = _bounding_box_from_categorical((192, 224), 'ukb_cardiac_mri/cine_segmented_ao_dist_annotated_1/instance_0', MRI_AO_SEGMENTED_CHANNEL_MAP['ascending_aorta'])
TMAPS['cine_segmented_ao_ascending_aorta_bbox_slice0'] = TensorMap('cine_segmented_ao_ascending_aorta_bbox_slice0', Interpretation.MESH, shape=(4,),
                                                                   tensor_from_file=aorta_ascending_tff, channel_map=_bounding_box_channel_map(2))

TMAPS['lax_3ch_lv_cavity_bbox'] = TensorMap('lax_3ch_lv_cavity_bbox', Interpretation.MESH, shape=(6,), channel_map=_bounding_box_channel_map(3),
                                            tensor_from_file=_bounding_box_from_callable(5, _segmented_index_slices('cine_segmented_lax_3ch_annotated_', (192, 160, 50))))

bbfc = _bounding_box_from_callable(MRI_AO_SEGMENTED_CHANNEL_MAP['descending_aorta'], _segmented_index_slices('cine_segmented_ao_dist_annotated_', (192, 224, 100)))
TMAPS['cine_segmented_ao_descending_aorta_bbox'] = TensorMap('cine_segmented_ao_descending_aorta_bbox', Interpretation.MESH, shape=(6,), tensor_from_file=bbfc,
                                                             channel_map=_bounding_box_channel_map(3))
abbfc = _bounding_box_from_callable(MRI_AO_SEGMENTED_CHANNEL_MAP['ascending_aorta'], _segmented_index_slices('cine_segmented_ao_dist_annotated_', (192, 224, 100)))
TMAPS['cine_segmented_ao_ascending_aorta_bbox'] = TensorMap('cine_segmented_ao_ascending_aorta_bbox', Interpretation.MESH, shape=(6,), tensor_from_file=abbfc,
                                                            channel_map=_bounding_box_channel_map(3))
