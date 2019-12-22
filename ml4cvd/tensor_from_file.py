from typing import List, Dict

import os
import vtk
import h5py
import numpy as np
import vtk.util.numpy_support
import logging
from keras.utils import to_categorical

from ml4cvd.TensorMap import TensorMap, no_nans
from ml4cvd.metrics import weighted_crossentropy
from ml4cvd.tensor_writer_ukbb import tensor_path, path_date_to_datetime
from ml4cvd.defines import ECG_REST_LEADS, ECG_REST_MEDIAN_LEADS, ECG_REST_AMP_LEADS
from ml4cvd.defines import DataSetType, EPS, MRI_TO_SEGMENT, MRI_LAX_TO_SEGMENT, MRI_SEGMENTED, MRI_LAX_SEGMENTED, MRI_SEGMENTED_CHANNEL_MAP
from ml4cvd.defines import MRI_PIXEL_WIDTH, MRI_PIXEL_HEIGHT, MRI_SLICE_THICKNESS, MRI_PATIENT_ORIENTATION, MRI_PATIENT_POSITION, MRI_FRAMES


"""
For now, all we will map `group` in TensorMap to `source` in tensor_path and `name` to `name`
"""


def normalized_first_date(tm: TensorMap, hd5: h5py.File, dependents=None):
    tensor = _get_tensor_at_first_date(hd5, tm.group, tm.dtype, tm.name)
    if tm.dtype == DataSetType.CONTINUOUS:
        return tm.normalize_and_validate(tensor)
    if tm.dtype == DataSetType.FLOAT_ARRAY:
        tensor = tm.normalize_and_validate(tensor)
        return _pad_array_to_shape(tm, tensor)
    raise ValueError(f'normalize_first_date not implemented for {tm.dtype}')


def random_slice_tensor(tensor_key, dependent_key=None):
    def _random_slice_tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        big_tensor = _get_tensor_at_first_date(hd5, tm.group, tm.dtype, tensor_key)
        cur_slice = np.random.choice(range(big_tensor.shape[-1]))
        tensor = np.zeros(tm.shape, dtype=np.float32)
        tensor[..., 0] = big_tensor[..., cur_slice]
        if dependent_key is not None:
            dependents[tm.dependent_map] = np.zeros(tm.dependent_map.shape, dtype=np.float32)
            label_tensor = np.array(hd5[dependent_key][..., cur_slice], dtype=np.float32)
            dependents[tm.dependent_map][:, :, :] = to_categorical(label_tensor, tm.dependent_map.shape[-1])
        return tm.normalize_and_validate(tensor)
    return _random_slice_tensor_from_file


def slice_subset_tensor(tensor_key, start, stop, step=1, dependent_key=None):
    def _slice_subset_tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        big_tensor = _get_tensor_at_first_date(hd5, tm.group, tm.dtype, tensor_key)
        tensor = big_tensor[..., np.arange(start, stop, step)]
        if dependent_key is not None:
            dependents[tm.dependent_map] = np.zeros(tm.dependent_map.shape, dtype=np.float32)
            label_tensor = np.array(hd5[dependent_key][..., start:stop], dtype=np.float32)
            dependents[tm.dependent_map][:, :, :] = to_categorical(label_tensor, tm.dependent_map.shape[-1])
        return tm.normalize_and_validate(tensor)
    return _slice_subset_tensor_from_file


def _all_dates(hd5: h5py.File, source: str, dtype: DataSetType, name: str) -> List[str]:
    """
    Gets the dates in the hd5 with source, dtype, name.
    """
    # TODO: This ideally would be implemented to not depend on the order of name, date, dtype, source in the hd5s
    # Unfortunately, that's hard to do efficiently
    return hd5[source][str(dtype)][name]

def _pass_nan(tensor):
    return (tensor)

def _fail_nan(tensor):
    if np.isnan(tensor).any():
        raise ValueError('Tensor contains nans.')
    return tensor


def _nan_to_mean(tensor, max_allowed_nan_fraction=.2):
    tensor_isnan = np.isnan(tensor)
    if np.count_nonzero(tensor_isnan) / tensor.size > max_allowed_nan_fraction:
        raise ValueError('Tensor contains too many nans.')
    tensor[tensor_isnan] = np.nanmean(tensor)
    return tensor


def _get_tensor_at_first_date(hd5: h5py.File, source: str, dtype: DataSetType, name: str, handle_nan=_fail_nan):
    """
    Gets the numpy array at the first date of source, dtype, name.
    """
    dates = _all_dates(hd5, source, dtype, name)
    if not dates:
        raise ValueError(f'No {name} values values available.')
    # TODO: weird to convert date from string to datetime, because it just gets converted back.
    first_date = path_date_to_datetime(min(dates))  # Date format is sortable. 
    first_date_path = tensor_path(source=source, dtype=dtype, name=name, date=first_date)
    tensor = np.array(hd5[first_date_path], dtype=np.float32)
    tensor = handle_nan(tensor)
    return tensor


def _pad_array_to_shape(tm: TensorMap, original: np.ndarray):
    if tm.shape == original.shape:
        return original
    padded = np.zeros(tm.shape)
    padded[:original.shape[0]] = original.reshape((original.shape[0],) + tm.shape[1:])
    return padded


# BIKE ECG
def _check_phase_full_len(hd5: h5py.File, phase: str):
    phase_len = _get_tensor_at_first_date(hd5, 'ecg_bike', DataSetType.CONTINUOUS, f'{phase}_duration')
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


def _first_date_bike_recovery(tm: TensorMap, hd5: h5py.File, dependents=None):
    _check_phase_full_len(hd5, 'rest')
    original = _get_tensor_at_first_date(hd5, tm.group, DataSetType.FLOAT_ARRAY, tm.name)
    recovery = original[-tm.shape[0]:]
    return tm.normalize_and_validate(recovery).reshape(tm.shape)


def _first_date_bike_pretest(tm: TensorMap, hd5: h5py.File, dependents=None):
    _check_phase_full_len(hd5, 'pretest')
    original = _get_tensor_at_first_date(hd5, tm.group, DataSetType.FLOAT_ARRAY, tm.name)
    pretest = original[:tm.shape[0]]
    return tm.normalize_and_validate(pretest).reshape(tm.shape)


def _first_date_hrr(tm: TensorMap, hd5: h5py.File, dependents=None):
    _check_phase_full_len(hd5, 'rest')
    last_hr = _get_tensor_at_first_date(hd5, 'ecg_bike', DataSetType.FLOAT_ARRAY, 'trend_heartrate')[-1]
    max_hr = _get_tensor_at_first_date(hd5, 'ecg_bike', DataSetType.CONTINUOUS, 'max_hr')
    return tm.normalize_and_validate(max_hr - last_hr)


def _healthy_check(hd5):
    for phase in ('pretest', 'exercise', 'rest'):
        _check_phase_full_len(hd5, phase)
    max_load = max(_get_tensor_at_first_date(hd5, 'ecg_bike', DataSetType.FLOAT_ARRAY, 'trend_load'))
    if max_load < 60:
        raise ValueError('Max load not high enough')


def _healthy_bike(tm: TensorMap, hd5: h5py.File, dependents=None):
    _healthy_check(hd5)
    return normalized_first_date(tm, hd5)


def _healthy_hrr(tm: TensorMap, hd5: h5py.File, dependents=None):
    _healthy_check(hd5)
    return _first_date_hrr(tm, hd5)


def _first_date_hrr(tm: TensorMap, hd5: h5py.File, dependents=None):
    _check_phase_full_len(hd5, 'rest')
    last_hr = _get_tensor_at_first_date(hd5, 'ecg_bike', DataSetType.FLOAT_ARRAY, 'trend_heartrate')[-1]
    max_hr = _get_tensor_at_first_date(hd5, 'ecg_bike', DataSetType.CONTINUOUS, 'max_hr')
    return tm.normalize_and_validate(max_hr - last_hr)


def _median_pretest(tm: TensorMap, hd5: h5py.File, dependents=None):
    _healthy_check(hd5)
    times = _get_tensor_at_first_date(hd5, 'ecg_bike', DataSetType.FLOAT_ARRAY, 'trend_time')
    tensor = np.abs(_get_tensor_at_first_date(hd5, tm.group, DataSetType.FLOAT_ARRAY, tm.name))
    return tm.normalize_and_validate(np.median(tensor[times <= 15]))


def _new_hrr(tm: TensorMap, hd5: h5py.File, dependents=None):
    _check_phase_full_len(hd5, 'rest')
    hrs = _get_tensor_at_first_date(hd5, 'ecg_bike', DataSetType.FLOAT_ARRAY, 'trend_heartrate')
    phases = _get_tensor_at_first_date(hd5, 'ecg_bike', DataSetType.FLOAT_ARRAY, 'trend_phasename')
    min_hr = hrs[phases == 2].min()
    max_hr = _get_tensor_at_first_date(hd5, 'ecg_bike', DataSetType.CONTINUOUS, 'max_hr')
    max_pred = _get_tensor_at_first_date(hd5, 'ecg_bike', DataSetType.CONTINUOUS, 'max_pred_hr')
    hrr = max_hr - min_hr
    if max_hr / max_pred > 150:
        raise ValueError('Max hr / max pred hr too high.')
    if hrr > 80:
        raise ValueError('HRR too high.')
    return tm.normalize_and_validate(hrr)


def _hr_achieved(tm: TensorMap, hd5: h5py.File, dependents=None):
    _check_phase_full_len(hd5, 'rest')
    max_hr = _get_tensor_at_first_date(hd5, 'ecg_bike', DataSetType.CONTINUOUS, 'max_hr')
    max_pred = _get_tensor_at_first_date(hd5, 'ecg_bike', DataSetType.CONTINUOUS, 'max_pred_hr')
    return tm.normalize_and_validate(max_hr / max_pred)


TMAPS: Dict[str, TensorMap] = dict()


TMAPS['ecg-bike-hrr'] = TensorMap('hrr', group='ecg_bike', loss='logcosh', metrics=['mae'], shape=(1,),
                                  normalization={'mean': 30.55, 'std': 12.81},
                                  tensor_from_file=_first_date_hrr, dtype=DataSetType.CONTINUOUS)
TMAPS['ecg-bike-healthy-max-hr'] = TensorMap('max_hr', group='ecg_bike', loss='logcosh', metrics=['mae'],
                                             normalization={'mean': 113.7, 'std': 13.3}, shape=(1,),
                                             tensor_from_file=_healthy_bike, dtype=DataSetType.CONTINUOUS)
TMAPS['ecg-bike-healthy-hrr'] = TensorMap('hrr', group='ecg_bike', loss='logcosh', metrics=['mae'], shape=(1,),
                                          normalization={'mean': 30.47, 'std': 11.76},
                                          tensor_from_file=_healthy_hrr, dtype=DataSetType.CONTINUOUS)
TMAPS['ecg-bike-healthy-resting'] = TensorMap('resting_hr', group='ecg_bike', loss='logcosh', metrics=['mae'], shape=(1,),
                                              normalization={'mean': 70.0, 'std': 11.62},
                                              tensor_from_file=_healthy_bike, dtype=DataSetType.CONTINUOUS)
TMAPS['ecg-bike-med-pretest-hr'] = TensorMap('trend_heartrate', group='ecg_bike', loss='logcosh', metrics=['mae'], shape=(1,),
                                             normalization={'mean': 70., 'std': 11.},
                                             tensor_from_file=_median_pretest, dtype=DataSetType.CONTINUOUS)
TMAPS['ecg-bike-med-pretest-stamp'] = TensorMap('trend_stamplitude', group='ecg_bike', loss='logcosh', metrics=['mae'], shape=(1,),
                                                normalization={'mean': .03, 'std': .03},
                                                tensor_from_file=_median_pretest, dtype=DataSetType.CONTINUOUS)
TMAPS['ecg-bike-med-pretest-jpoint'] = TensorMap('trend_jpointamplitude', group='ecg_bike', loss='logcosh', metrics=['mae'], shape=(1,),
                                                 normalization={'mean': .032, 'std': .46},
                                                 tensor_from_file=_median_pretest, dtype=DataSetType.CONTINUOUS)
TMAPS['ecg-bike-med-pretest-stamp20'] = TensorMap('trend_stamplitude20ms', group='ecg_bike', loss='logcosh', metrics=['mae'], shape=(1,),
                                                  normalization={'mean': .03, 'std': .03},
                                                  tensor_from_file=_median_pretest, dtype=DataSetType.CONTINUOUS)
TMAPS['ecg-bike-recovery'] = TensorMap('full', shape=(30000, 1), group='ecg_bike', validator=no_nans,
                                       tensor_from_file=_first_date_bike_recovery, dtype=DataSetType.FLOAT_ARRAY)
TMAPS['ecg-bike-pretest'] = TensorMap('full', shape=(500 * 15 - 4, 3), group='ecg_bike', validator=no_nans,
                                      normalization={'mean': np.array([7, -7, 3.5])[np.newaxis], 'std': np.array([31, 30, 16])[np.newaxis]},
                                      tensor_from_file=_first_date_bike_pretest, dtype=DataSetType.FLOAT_ARRAY)
TMAPS['ecg-bike-new-hrr'] = TensorMap('hrr', group='ecg_bike', loss='logcosh', metrics=['mae'], shape=(1,),
                                      normalization={'mean': 31, 'std': 12},
                                      tensor_from_file=_new_hrr, dtype=DataSetType.CONTINUOUS)
TMAPS['ecg-bike-hr-achieved'] = TensorMap('hr_achieved', group='ecg_bike', loss='logcosh', metrics=['mae'], shape=(1,),
                                          normalization={'mean': .68, 'std': .1},
                                          tensor_from_file=_hr_achieved, dtype=DataSetType.CONTINUOUS)


def _make_ecg_rest(population_normalize: float = None):
    def ecg_rest_from_file(tm, hd5, dependents={}):
        tensor = np.zeros(tm.shape, dtype=np.float32)
        if tm.dependent_map is not None:
            dependents[tm.dependent_map] = np.zeros(tm.dependent_map.shape, dtype=np.float32)
            key_choices = [k for k in hd5[tm.group] if tm.name in k]
            lead_idx = np.random.choice(key_choices)
            tensor = np.reshape(hd5[tm.group][lead_idx][: tensor.shape[0] * tensor.shape[1]], tensor.shape, order='F')
            dependents[tm.dependent_map][:, 0] = np.array(hd5[tm.group][lead_idx.replace(tm.name, tm.dependent_map.name)])
            dependents[tm.dependent_map] = tm.zero_mean_std1(dependents[tm.dependent_map])
        else:
            for k in hd5[tm.group]:
                if k in tm.channel_map:
                    if len(tensor.shape) == 3:  # Grab the stacked tensor maps
                        window_size = tensor.shape[0]
                        channels = tensor.shape[2]
                        new_shape = (window_size, channels)
                        new_total = window_size * channels
                        tensor[:, tm.channel_map[k], :] = np.reshape(hd5[tm.group][k][:new_total], new_shape, order='F')
                    elif tm.name == 'ecg_rest_fft':
                        tensor[:, tm.channel_map[k]] = np.log(np.abs(np.fft.fft(hd5[tm.group][k])) + EPS)
                    else:
                        tensor[:, tm.channel_map[k]] = hd5[tm.group][k]
        if population_normalize is None:
            tensor = tm.zero_mean_std1(tensor)
        else:
            tensor /= population_normalize
        return tensor
    return ecg_rest_from_file


TMAPS['ecg_rest_raw'] = TensorMap('ecg_rest_raw', shape=(5000, 12), group='ecg_rest', tensor_from_file=_make_ecg_rest(population_normalize=2000.0),
                                  channel_map=ECG_REST_LEADS)

TMAPS['ecg_rest'] = TensorMap('strip', shape=(5000, 12), group='ecg_rest', tensor_from_file=_make_ecg_rest(),
                              channel_map=ECG_REST_LEADS)

TMAPS['ecg_rest_fft'] = TensorMap('ecg_rest_fft', shape=(5000, 12), group='ecg_rest', tensor_from_file=_make_ecg_rest(),
                                  channel_map=ECG_REST_LEADS)

TMAPS['ecg_rest_stack'] = TensorMap('strip', shape=(600, 12, 8), group='ecg_rest', tensor_from_file=_make_ecg_rest(),
                                    channel_map=ECG_REST_LEADS)

TMAPS['ecg_rest_median_raw'] = TensorMap('median', group='ecg_rest', shape=(600, 12), loss='logcosh', activation='linear', tensor_from_file=_make_ecg_rest(population_normalize=2000.0),
                                     metrics=['mse', 'mae', 'logcosh'], channel_map=ECG_REST_MEDIAN_LEADS)

TMAPS['ecg_rest_median'] = TensorMap('median', group='ecg_rest', shape=(600, 12), loss='logcosh', activation='linear', tensor_from_file=_make_ecg_rest(),
                                     metrics=['mse', 'mae', 'logcosh'], channel_map=ECG_REST_MEDIAN_LEADS)

TMAPS['ecg_rest_median_stack'] = TensorMap('median', group='ecg_rest', shape=(600, 12, 1), activation='linear', tensor_from_file=_make_ecg_rest(),
                                           metrics=['mse', 'mae', 'logcosh'], loss='logcosh', loss_weight=1.0,
                                           channel_map=ECG_REST_MEDIAN_LEADS)

TMAPS['ecg_median_1lead'] = TensorMap('median', group='ecg_rest', shape=(600, 1), loss='logcosh', loss_weight=10.0, tensor_from_file=_make_ecg_rest(),
                                      activation='linear', metrics=['mse', 'mae', 'logcosh'], channel_map={'lead': 0})

TMAPS['ecg_rest_1lead'] = TensorMap('strip', shape=(600, 8), group='ecg_rest', channel_map={'lead': 0}, tensor_from_file=_make_ecg_rest(),
                                    dependent_map=TMAPS['ecg_median_1lead'])


def _get_lead_cm(length):
    lead_cm = {}
    lead_weights = []
    for i in range(length):
        wave_val = i - (length//2)
        lead_cm['w'+str(wave_val).replace('-', '_')] = i
        lead_weights.append((np.abs(wave_val+1)/(length/2)) + 1.0)
    return lead_cm, lead_weights


TMAPS['ecg_median_1lead_categorical'] = TensorMap('median', group='categorical', shape=(600, 32), activation='softmax', tensor_from_file=_make_ecg_rest(),
                                                  channel_map=_get_lead_cm(32)[0],
                                                  loss=weighted_crossentropy(_get_lead_cm(32)[1], 'ecg_median_categorical'))

TMAPS['ecg_rest_1lead_categorical'] = TensorMap('strip', shape=(600, 8), group='ecg_rest', tensor_from_file=_make_ecg_rest(),
                                                channel_map={'window0': 0, 'window1': 1, 'window2': 2, 'window3': 3,
                                                             'window4': 4, 'window5': 5, 'window6': 6, 'window7': 7},
                                                dependent_map=TMAPS['ecg_median_1lead_categorical'])


# Extract RAmplitude and SAmplitude for LVH criteria
def _make_ukb_ecg_rest(population_normalize: float = None):
    def ukb_ecg_rest_from_file(tm, hd5):
        if 'ukb_ecg_rest' not in hd5:
            raise ValueError('Group with R and S amplitudes not present in hd5')
        tensor = _get_tensor_at_first_date(hd5, tm.group, DataSetType.FLOAT_ARRAY, tm.name, _pass_nan)        
        try:            
            if population_normalize is None:
                tensor = tm.zero_mean_std1(tensor)
            else:
                tensor /= population_normalize
        except:
            ValueError(f'Cannot normalize {tm.name}')
        return tensor
    return ukb_ecg_rest_from_file


TMAPS['ecg_rest_ramplitude_raw'] = TensorMap('ramplitude', group='ukb_ecg_rest', shape=(12,), tensor_from_file=_make_ukb_ecg_rest(1.0),
                            loss='logcosh', metrics=['mse', 'mape', 'mae'], loss_weight=1.0)

TMAPS['ecg_rest_samplitude_raw'] = TensorMap('samplitude', group='ukb_ecg_rest', shape=(12,), tensor_from_file=_make_ukb_ecg_rest(1.0),
                            loss='logcosh', metrics=['mse', 'mape', 'mae'], loss_weight=1.0)

TMAPS['ecg_rest_ramplitude'] = TensorMap('ramplitude', group='ukb_ecg_rest', shape=(12,), tensor_from_file=_make_ukb_ecg_rest(),
                            loss='logcosh', metrics=['mse', 'mape', 'mae'], loss_weight=1.0)

TMAPS['ecg_rest_samplitude'] = TensorMap('samplitude', group='ukb_ecg_rest', shape=(12,), tensor_from_file=_make_ukb_ecg_rest(),
                            loss='logcosh', metrics=['mse', 'mape', 'mae'], loss_weight=1.0)


def _make_ukb_ecg_rest_lvh():
    def ukb_ecg_rest_lvh_from_file(tm, hd5):
        # Lead order seems constant and standard throughout, but we could eventually tensorize it from XML
        lead_order = ECG_REST_AMP_LEADS
        avl_min = 1100.0
        sl_min = 3500.0
        cornell_female_min = 2000.0
        cornell_male_min = 2800.0
        if 'ukb_ecg_rest' not in hd5:
            raise ValueError('Group with R and S amplitudes not present in hd5')        
        tensor_ramp = _get_tensor_at_first_date(hd5, tm.group, DataSetType.FLOAT_ARRAY, 'ramplitude', _pass_nan)
        tensor_samp = _get_tensor_at_first_date(hd5, tm.group, DataSetType.FLOAT_ARRAY, 'samplitude', _pass_nan)
        criteria_sleads = [lead_order[l] for l in ['V1', 'V3']]
        criteria_rleads = [lead_order[l] for l in ['aVL', 'V5', 'V6']]
        if np.any(np.isnan(np.union1d(tensor_ramp[criteria_rleads], tensor_samp[criteria_sleads]))):
            raise ValueError('Missing some of the R and S amplitude readings needed to evaluate LVH criteria')        
        is_female = 'Genetic-sex_Female_0_0' in hd5['categorical']
        is_male   = 'Genetic-sex_Male_0_0' in hd5['categorical']
        # If genetic sex not available, try phenotypic
        if not(is_female or is_male):
            is_female = 'Sex_Female_0_0' in hd5['categorical']
            is_male   = 'Sex_Male_0_0' in hd5['categorical']
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
        

TMAPS['ecg_rest_lvh_avl'] = TensorMap('avl_lvh', group='ukb_ecg_rest', tensor_from_file=_make_ukb_ecg_rest_lvh(),
                            channel_map={'no_avl_lvh': 0, 'aVL LVH': 1},
                            loss=weighted_crossentropy([0.006, 1.0], 'avl_lvh'))

TMAPS['ecg_rest_lvh_sokolow_lyon'] = TensorMap('sokolow_lyon_lvh', group='ukb_ecg_rest', tensor_from_file=_make_ukb_ecg_rest_lvh(),
                            channel_map={'no_sokolow_lyon_lvh': 0, 'Sokolow Lyon LVH': 1},
                            loss=weighted_crossentropy([0.005, 1.0], 'sokolov_lyon_lvh'))

TMAPS['ecg_rest_lvh_cornell'] = TensorMap('cornell_lvh', group='ukb_ecg_rest', tensor_from_file=_make_ukb_ecg_rest_lvh(),
                            channel_map={'no_cornell_lvh': 0, 'Cornell LVH': 1},
                            loss=weighted_crossentropy([0.003, 1.0], 'cornell_lvh'))
    

TMAPS['t2_flair_sag_p2_1mm_fs_ellip_pf78_1'] = TensorMap('t2_flair_sag_p2_1mm_fs_ellip_pf78_1', shape=(256, 256, 192), group='ukb_brain_mri',
                                                         tensor_from_file=normalized_first_date, dtype=DataSetType.FLOAT_ARRAY,
                                                         normalization={'zero_mean_std1': True})
TMAPS['t2_flair_sag_p2_1mm_fs_ellip_pf78_2'] = TensorMap('t2_flair_sag_p2_1mm_fs_ellip_pf78_2', shape=(256, 256, 192), group='ukb_brain_mri',
                                                         tensor_from_file=normalized_first_date, dtype=DataSetType.FLOAT_ARRAY,
                                                         normalization={'zero_mean_std1': True})
TMAPS['t2_flair_slice_1'] = TensorMap('t2_flair_slice_1', shape=(256, 256, 1), group='ukb_brain_mri', dtype=DataSetType.FLOAT_ARRAY,
                                      tensor_from_file=random_slice_tensor('t2_flair_sag_p2_1mm_fs_ellip_pf78_1'), normalization={'zero_mean_std1': True})
TMAPS['t2_flair_slice_2'] = TensorMap('t2_flair_slice_2', shape=(256, 256, 1), group='ukb_brain_mri', dtype=DataSetType.FLOAT_ARRAY,
                                      tensor_from_file=random_slice_tensor('t2_flair_sag_p2_1mm_fs_ellip_pf78_2'), normalization={'zero_mean_std1': True})
TMAPS['t1_p2_1mm_fov256_sag_ti_880_1'] = TensorMap('t1_p2_1mm_fov256_sag_ti_880_1', shape=(256, 256, 208), group='ukb_brain_mri', dtype=DataSetType.FLOAT_ARRAY,
                                                   normalization={'zero_mean_std1': True}, tensor_from_file=normalized_first_date)
TMAPS['t1_p2_1mm_fov256_sag_ti_880_2'] = TensorMap('t1_p2_1mm_fov256_sag_ti_880_2', shape=(256, 256, 208), group='ukb_brain_mri', dtype=DataSetType.FLOAT_ARRAY,
                                                   normalization={'zero_mean_std1': True}, tensor_from_file=normalized_first_date)
TMAPS['t1_slice_1'] = TensorMap('t1_slice_1', shape=(256, 256, 1), group='ukb_brain_mri', dtype=DataSetType.FLOAT_ARRAY, normalization={'zero_mean_std1': True},
                                tensor_from_file=random_slice_tensor('t1_p2_1mm_fov256_sag_ti_880_1'))
TMAPS['t1_slice_2'] = TensorMap('t1_slice_2', shape=(256, 256, 1), group='ukb_brain_mri', dtype=DataSetType.FLOAT_ARRAY, normalization={'zero_mean_std1': True},
                                tensor_from_file=random_slice_tensor('t1_p2_1mm_fov256_sag_ti_880_2'))
TMAPS['t1_20_slices_1'] = TensorMap('t1_20_slices_1', shape=(256, 256, 20), group='ukb_brain_mri', dtype=DataSetType.FLOAT_ARRAY,
                                    normalization={'zero_mean_std1': True},
                                    tensor_from_file=slice_subset_tensor('t1_p2_1mm_fov256_sag_ti_880_1', 94, 114))
TMAPS['t1_20_slices_2'] = TensorMap('t1_20_slices_2', shape=(256, 256, 20), group='ukb_brain_mri', dtype=DataSetType.FLOAT_ARRAY,
                                    normalization={'zero_mean_std1': True},
                                    tensor_from_file=slice_subset_tensor('t1_p2_1mm_fov256_sag_ti_880_2', 94, 114))
TMAPS['t2_20_slices_1'] = TensorMap('t2_20_slices_1', shape=(256, 256, 20), group='ukb_brain_mri', dtype=DataSetType.FLOAT_ARRAY,
                                    normalization={'zero_mean_std1': True},
                                    tensor_from_file=slice_subset_tensor('t2_flair_sag_p2_1mm_fs_ellip_pf78_1', 86, 106))
TMAPS['t2_20_slices_2'] = TensorMap('t2_20_slices_2', shape=(256, 256, 20), group='ukb_brain_mri', dtype=DataSetType.FLOAT_ARRAY,
                                    normalization={'zero_mean_std1': True},
                                    tensor_from_file=slice_subset_tensor('t2_flair_sag_p2_1mm_fs_ellip_pf78_2', 86, 106))
TMAPS['t1_40_slices_1'] = TensorMap('t1_40_slices_1', shape=(256, 256, 40), group='ukb_brain_mri', dtype=DataSetType.FLOAT_ARRAY,
                                    normalization={'zero_mean_std1': True},
                                    tensor_from_file=slice_subset_tensor('t1_p2_1mm_fov256_sag_ti_880_1', 64, 144, 2))
TMAPS['t2_40_slices_1'] = TensorMap('t2_40_slices_1', shape=(256, 256, 40), group='ukb_brain_mri', dtype=DataSetType.FLOAT_ARRAY,
                                    normalization={'zero_mean_std1': True},
                                    tensor_from_file=slice_subset_tensor('t2_flair_sag_p2_1mm_fs_ellip_pf78_1', 56, 136, 2))


def ttn_tensor_from_file(tm, hd5, dependents={}):
    index = 0
    categorical_data = np.zeros(tm.shape, dtype=np.float32)
    if 'has_exome' not in hd5['categorical']:
        raise ValueError('Skipping people without exome sequencing.')
    if tm.name in hd5['categorical'] and int(hd5['categorical'][tm.name][0]) != 0:
        index = 1
    categorical_data[index] = 1.0
    return categorical_data


TMAPS['ttntv'] = TensorMap('has_ttntv', group='categorical_flag', channel_map={'no_TTN_tv': 0, 'TTN_tv': 1}, tensor_from_file=ttn_tensor_from_file)
TMAPS['ttntv_10x'] = TensorMap('has_ttntv', group='categorical_flag', channel_map={'no_TTN_tv': 0, 'TTN_tv': 1}, loss_weight=10.0, tensor_from_file=ttn_tensor_from_file)


def make_index_tensor_from_file(index_map_name):
    def indexed_lvmass_tensor_from_file(tm, hd5, dependents={}):
        tensor = np.zeros(tm.shape, dtype=np.float32)
        for k in tm.channel_map:
            if k in hd5[tm.group]:
                tensor = np.array(hd5[tm.group][k], dtype=np.float32)
            else:
                return tensor
        index = np.array(hd5[tm.group][index_map_name], dtype=np.float32)
        return tm.normalize_and_validate(tensor / index)
    return indexed_lvmass_tensor_from_file


TMAPS['lv_mass_dubois_index'] = TensorMap('lv_mass_dubois_index', group='continuous', activation='linear', loss='logcosh', loss_weight=1.0,
                                          tensor_from_file=make_index_tensor_from_file('bsa_dubois'),
                                          channel_map={'lv_mass': 0}, normalization={'mean': 89.7, 'std': 24.8})
TMAPS['lv_mass_mosteller_index'] = TensorMap('lv_mass_mosteller_index', group='continuous', activation='linear', loss='logcosh', loss_weight=1.0,
                                             tensor_from_file=make_index_tensor_from_file('bsa_mosteller'),
                                             channel_map={'lv_mass': 0}, normalization={'mean': 89.7, 'std': 24.8})
TMAPS['lv_mass_dubois_index_sentinel'] = TensorMap('lv_mass_dubois_index', group='continuous', activation='linear', sentinel=0, loss_weight=1.0,
                                          tensor_from_file=make_index_tensor_from_file('bsa_dubois'),
                                          channel_map={'lv_mass': 0}, normalization={'mean': 89.7, 'std': 24.8})
TMAPS['lv_mass_mosteller_index_sentinel'] = TensorMap('lv_mass_mosteller_index', group='continuous', activation='linear', sentinel=0, loss_weight=1.0,
                                             tensor_from_file=make_index_tensor_from_file('bsa_mosteller'),
                                             channel_map={'lv_mass': 0}, normalization={'mean': 89.7, 'std': 24.8})
TMAPS['lv_mass_dubois_indexp'] = TensorMap('lv_mass_dubois_index', group='continuous', activation='linear', loss='logcosh', loss_weight=1.0,
                                           parents=['output_mri_systole_diastole_8_segmented_categorical'],
                                           tensor_from_file=make_index_tensor_from_file('bsa_dubois'),
                                           channel_map={'lv_mass': 0}, normalization={'mean': 89.7, 'std': 24.8})
TMAPS['lv_mass_mosteller_indexp'] = TensorMap('lv_mass_mosteller_index', group='continuous', activation='linear', loss='logcosh', loss_weight=1.0,
                                              parents=['output_mri_systole_diastole_8_segmented_categorical'],
                                              tensor_from_file=make_index_tensor_from_file('bsa_mosteller'),
                                              channel_map={'lv_mass': 0}, normalization={'mean': 89.7, 'std': 24.8})
TMAPS['lvm_dubois_index'] = TensorMap('lvm_dubois_index', group='continuous', activation='linear', loss='logcosh', loss_weight=1.0,
                                          tensor_from_file=make_index_tensor_from_file('bsa_dubois'),
                                          channel_map={'LVM': 0}, normalization={'mean': 89.7, 'std': 24.8})
TMAPS['lvm_mosteller_index'] = TensorMap('lvm_mosteller_index', group='continuous', activation='linear', loss='logcosh', loss_weight=1.0,
                                             tensor_from_file=make_index_tensor_from_file('bsa_mosteller'),
                                             channel_map={'LVM': 0}, normalization={'mean': 89.7, 'std': 24.8})
TMAPS['lvm_dubois_index_sentinel'] = TensorMap('lvm_dubois_index', group='continuous', activation='linear', sentinel=0, loss_weight=1.0,
                                          tensor_from_file=make_index_tensor_from_file('bsa_dubois'),
                                          channel_map={'LVM': 0}, normalization={'mean': 89.7, 'std': 24.8})
TMAPS['lvm_mosteller_index_sentinel'] = TensorMap('lvm_mosteller_index', group='continuous', activation='linear', sentinel=0, loss_weight=1.0,
                                             tensor_from_file=make_index_tensor_from_file('bsa_mosteller'),
                                             channel_map={'LVM': 0}, normalization={'mean': 89.7, 'std': 24.8})


def mri_slice_blackout_tensor_from_file(tm, hd5, dependents={}):
    cur_slice = np.random.choice(list(hd5[MRI_TO_SEGMENT].keys()))
    tensor = np.zeros(tm.shape, dtype=np.float32)
    dependents[tm.dependent_map] = np.zeros(tm.dependent_map.shape, dtype=np.float32)
    tensor[:, :, 0] = np.array(hd5[MRI_TO_SEGMENT][cur_slice], dtype=np.float32)
    label_tensor = np.array(hd5[MRI_SEGMENTED][cur_slice], dtype=np.float32)
    dependents[tm.dependent_map][:, :, :] = to_categorical(label_tensor, tm.dependent_map.shape[-1])
    tensor[:, :, 0] *= np.not_equal(label_tensor, 0, dtype=np.float32)
    return tm.zero_mean_std1(tensor)


TMAPS['mri_slice_blackout_segmented_weighted'] = TensorMap('mri_slice_segmented', (256, 256, 3), group='categorical', channel_map=MRI_SEGMENTED_CHANNEL_MAP,
                                                           loss=weighted_crossentropy([0.1, 25.0, 25.0], 'mri_slice_blackout_segmented'))
TMAPS['mri_slice_blackout'] = TensorMap('mri_slice_blackout', (256, 256, 1), tensor_from_file=mri_slice_blackout_tensor_from_file,
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


TMAPS['mri_patient_orientation_cine_segmented_lax_2ch'] = TensorMap('mri_patient_orientation_cine_segmented_lax_2ch', (6,), group='mri_orientation',
                                                                    tensor_from_file=_make_mri_series_orientation_and_position_from_file())
TMAPS['mri_patient_orientation_cine_segmented_lax_3ch'] = TensorMap('mri_patient_orientation_cine_segmented_lax_3ch', (6,), group='mri_orientation',
                                                                    tensor_from_file=_make_mri_series_orientation_and_position_from_file())
TMAPS['mri_patient_orientation_cine_segmented_lax_4ch'] = TensorMap('mri_patient_orientation_cine_segmented_lax_4ch', (6,), group='mri_orientation',
                                                                    tensor_from_file=_make_mri_series_orientation_and_position_from_file())
TMAPS['mri_patient_orientation_cine_segmented_sax_b1'] = TensorMap('mri_patient_orientation_cine_segmented_sax_b1', (6,), group='mri_orientation',
                                                                    tensor_from_file=_make_mri_series_orientation_and_position_from_file())
TMAPS['mri_patient_orientation_cine_segmented_sax_inlinevf'] = TensorMap('mri_patient_orientation_cine_segmented_sax_inlinevf', (6, 750), group='mri_orientation',
                                                                         tensor_from_file=_make_mri_series_orientation_and_position_from_file())
TMAPS['mri_patient_position_cine_segmented_lax_2ch'] = TensorMap('mri_patient_position_cine_segmented_lax_2ch', (3,), group='mri_position',
                                                                 tensor_from_file=_make_mri_series_orientation_and_position_from_file())
TMAPS['mri_patient_position_cine_segmented_lax_3ch'] = TensorMap('mri_patient_position_cine_segmented_lax_3ch', (3,), group='mri_position',
                                                                 tensor_from_file=_make_mri_series_orientation_and_position_from_file())
TMAPS['mri_patient_position_cine_segmented_lax_4ch'] = TensorMap('mri_patient_position_cine_segmented_lax_4ch', (3,), group='mri_position',
                                                                 tensor_from_file=_make_mri_series_orientation_and_position_from_file())
TMAPS['mri_patient_position_cine_segmented_sax_b1'] = TensorMap('mri_patient_position_cine_segmented_sax_b1', (3,), group='mri_position',
                                                                tensor_from_file=_make_mri_series_orientation_and_position_from_file())
TMAPS['mri_patient_position_cine_segmented_sax_inlinevf'] = TensorMap('mri_patient_position_cine_segmented_sax_inlinevf', (3, 750), group='mri_position',
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
    Returns MRI tensors as lists of VTK structured grids aligned to the reference system of the patient
    """
    arr = _mri_tensor_4d(hd5, name)
    width = hd5['_'.join([MRI_PIXEL_WIDTH, name])]
    height = hd5['_'.join([MRI_PIXEL_HEIGHT, name])]
    positions = _mri_tensor_2d(hd5, '_'.join([MRI_PATIENT_POSITION, name]))
    orientations = _mri_tensor_2d(hd5, '_'.join([MRI_PATIENT_ORIENTATION, name]))
    thickness = hd5['_'.join([MRI_SLICE_THICKNESS, name])]
    _, dataset_indices, dataset_counts = np.unique(orientations, axis=1, return_index=True, return_counts=True)
    ndatasets = len(dataset_indices)
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


def _make_mri_projected_segmentation_from_file(to_segment_name, segmented_name, save_path=None, population_normalize=None):
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


TMAPS['cine_segmented_lax_2ch_proj_from_sax'] = TensorMap('cine_segmented_lax_2ch_proj_from_sax', (256, 256, 50), loss='logcosh',
                                                          tensor_from_file=_make_mri_projected_segmentation_from_file('cine_segmented_lax_2ch', MRI_SEGMENTED))
TMAPS['cine_segmented_lax_3ch_proj_from_sax'] = TensorMap('cine_segmented_lax_3ch_proj_from_sax', (256, 256, 50), loss='logcosh',
                                                          tensor_from_file=_make_mri_projected_segmentation_from_file('cine_segmented_lax_3ch', MRI_SEGMENTED))
TMAPS['cine_segmented_lax_4ch_proj_from_sax'] = TensorMap('cine_segmented_lax_4ch_proj_from_sax', (256, 256, 50), loss='logcosh',
                                                          tensor_from_file=_make_mri_projected_segmentation_from_file('cine_segmented_lax_4ch', MRI_SEGMENTED))
TMAPS['cine_segmented_lax_2ch_proj_from_lax'] = TensorMap('cine_segmented_lax_2ch_proj_from_lax', (256, 256, 50), loss='logcosh',
                                                          tensor_from_file=_make_mri_projected_segmentation_from_file('cine_segmented_lax_2ch', MRI_LAX_SEGMENTED))
TMAPS['cine_segmented_lax_3ch_proj_from_lax'] = TensorMap('cine_segmented_lax_3ch_proj_from_lax', (256, 256, 50), loss='logcosh',
                                                          tensor_from_file=_make_mri_projected_segmentation_from_file('cine_segmented_lax_3ch', MRI_LAX_SEGMENTED))
TMAPS['cine_segmented_lax_4ch_proj_from_lax'] = TensorMap('cine_segmented_lax_4ch_proj_from_lax', (256, 256, 50), loss='logcosh',
                                                          tensor_from_file=_make_mri_projected_segmentation_from_file('cine_segmented_lax_4ch', MRI_LAX_SEGMENTED))
