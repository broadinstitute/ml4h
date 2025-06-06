import h5py
import numpy as np
import scipy
import logging

import biosppy
from typing import List, Tuple, Dict
from tensorflow.keras.utils import to_categorical

from ml4h.tensormap.ukb.demographics import age_in_years_tensor
from ml4h.normalizer import ZeroMeanStd1, Standardize, RandomStandardize
from ml4h.metrics import weighted_crossentropy, ignore_zeros_logcosh, mse_10x
from ml4h.TensorMap import TensorMap, Interpretation, no_nans, make_range_validator
from ml4h.defines import ECG_CHAR_2_IDX, ECG_REST_MGB_LEADS, ECG_REST_AMP_LEADS_UKB, ECG_BIKE_LEADS
from ml4h.defines import ECG_REST_LEADS, ECG_REST_MEDIAN_LEADS, ECG_REST_AMP_LEADS, ECG_SEGMENTED_CHANNEL_MAP
from ml4h.tensormap.general import tensor_path, pad_or_crop_array_to_shape, tensor_from_hd5, named_tensor_from_hd5
from ml4h.tensormap.general import get_tensor_at_first_date, normalized_first_date, pass_nan, build_tensor_from_file

_HRR_SENTINEL = -1000


# BIKE ECG
def _check_phase_full_len(hd5: h5py.File, phase: str):
    phase_len = get_tensor_at_first_date(hd5, 'ecg_bike', f'{phase}_duration')
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
    original = get_tensor_at_first_date(hd5, tm.path_prefix, tm.name)
    recovery = original[-tm.shape[0]:]
    return recovery.reshape(tm.shape)


def _first_date_bike_pretest(tm: TensorMap, hd5: h5py.File, dependents=None):
    _check_phase_full_len(hd5, 'pretest')
    original = get_tensor_at_first_date(hd5, tm.path_prefix, tm.name)
    pretest = original[:tm.shape[0]]
    return pretest.reshape(tm.shape)


def _first_date_hrr(tm: TensorMap, hd5: h5py.File, dependents=None):
    _check_phase_full_len(hd5, 'rest')
    last_hr = get_tensor_at_first_date(hd5, 'ecg_bike', 'trend_heartrate')[-1]
    max_hr = get_tensor_at_first_date(hd5, 'ecg_bike', 'max_hr')
    return max_hr - last_hr


def _healthy_check(hd5):
    for phase in ('pretest', 'exercise', 'rest'):
        _check_phase_full_len(hd5, phase)
    max_load = max(get_tensor_at_first_date(hd5, 'ecg_bike', 'trend_load'))
    if max_load < 60:
        raise ValueError('Max load not high enough')


def _healthy_bike(tm: TensorMap, hd5: h5py.File, dependents=None):
    _healthy_check(hd5)
    return normalized_first_date(tm, hd5)


def _healthy_hrr(tm: TensorMap, hd5: h5py.File, dependents=None):
    _healthy_check(hd5)
    return _first_date_hrr(tm, hd5)


def _median_pretest(tm: TensorMap, hd5: h5py.File, dependents=None):
    _healthy_check(hd5)
    times = get_tensor_at_first_date(hd5, 'ecg_bike', 'trend_time')
    tensor = np.abs(
        get_tensor_at_first_date(
        hd5, tm.path_prefix, 'float_array', tm.name,
        ),
    )
    return np.median(tensor[times <= 15])


def _new_hrr(tm: TensorMap, hd5: h5py.File, dependents=None):
    _check_phase_full_len(hd5, 'rest')
    hrs = get_tensor_at_first_date(hd5, 'ecg_bike', 'trend_heartrate')
    phases = get_tensor_at_first_date(hd5, 'ecg_bike', 'trend_phasename')
    min_hr = hrs[phases == 2].min()
    max_hr = get_tensor_at_first_date(hd5, 'ecg_bike', 'max_hr')
    max_pred = get_tensor_at_first_date(hd5, 'ecg_bike', 'max_pred_hr')
    hrr = max_hr - min_hr
    if max_hr / max_pred > 150:
        raise ValueError('Max hr / max pred hr too high.')
    if hrr > 80:
        raise ValueError('HRR too high.')
    return hrr


def _sentinel_hrr(tm: TensorMap, hd5: h5py.File, dependents=None):
    try:
        return _new_hrr(tm, hd5)
    except ValueError:
        return _HRR_SENTINEL


def _hr_achieved(tm: TensorMap, hd5: h5py.File, dependents=None):
    _check_phase_full_len(hd5, 'rest')
    max_hr = get_tensor_at_first_date(hd5, 'ecg_bike', 'max_hr')
    max_pred = get_tensor_at_first_date(hd5, 'ecg_bike', 'max_pred_hr')
    return max_hr / max_pred


def _warp_ecg(ecg):
    i = np.arange(ecg.shape[0])
    warped = i + (
        np.random.rand() * 100 * np.sin(i / (500 + np.random.rand() * 100))
        + np.random.rand() * 100 * np.cos(i / (500 + np.random.rand() * 100))
    )
    warped_ecg = np.zeros_like(ecg)
    for j in range(ecg.shape[1]):
        warped_ecg[:, j] = np.interp(i, warped, ecg[:, j])
    return warped_ecg


def _make_ecg_rest(
        instance: int = 0, downsample_steps: int = 0,
        short_time_nperseg: int = 0, short_time_noverlap: int = 0,
        skip_poor: bool = False, random_offset: int = 0,
):
    def ecg_rest_from_file(tm, hd5, dependents={}):
        ecg_interpretation = str(
            tm.hd5_first_dataset_in_group(
            hd5, 'ukb_ecg_rest/ecg_rest_text/',
            )[()],
        )
        if skip_poor and 'Poor data quality' in ecg_interpretation:
            raise ValueError(f'Poor data quality skipped by {tm.name}.')
        tensor = np.zeros(tm.shape, dtype=np.float32)
        for k in hd5[tm.path_prefix]:
            if k in tm.channel_map:
                data = tm.hd5_first_dataset_in_group(
                    hd5, f'{tm.path_prefix}/{k}/instance_{instance}',
                )
                if random_offset > 0:
                    offset = np.random.randint(random_offset)
                    data = data[offset:]
                if short_time_nperseg > 0 and short_time_noverlap > 0:
                    f, t, short_time_ft = scipy.signal.stft(
                        data, nperseg=short_time_nperseg, noverlap=short_time_noverlap,
                    )
                    tensor[..., tm.channel_map[k]] = short_time_ft
                elif downsample_steps > 1:
                    tensor[:, tm.channel_map[k]] = np.array(data, dtype=np.float32)[::downsample_steps]
                else:
                    tensor[:, tm.channel_map[k]] = pad_or_crop_array_to_shape((tm.shape[0],), data)
        return tensor
    return ecg_rest_from_file


def _get_lead_cm(length):
    lead_cm = {}
    lead_weights = []
    for i in range(length):
        wave_val = i - (length//2)
        lead_cm['w'+str(wave_val).replace('-', '_')] = i
        lead_weights.append((np.abs(wave_val+1)/(length/2)) + 1.0)
    return lead_cm, lead_weights


def _make_rhythm_tensor(skip_poor=True):
    def rhythm_tensor_from_file(tm, hd5, dependents={}):
        categorical_data = np.zeros(tm.shape, dtype=np.float32)
        ecg_interpretation = str(
            tm.hd5_first_dataset_in_group(
            hd5, 'ukb_ecg_rest/ecg_rest_text/',
            )[()],
        )
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


def label_from_ecg_interpretation_text(tm, hd5, dependents={}):
    categorical_data = np.zeros(tm.shape, dtype=np.float32)
    ecg_interpretation = str(
        tm.hd5_first_dataset_in_group(
        hd5, 'ukb_ecg_rest/ecg_rest_text/',
        )[()],
    )
    for channel in tm.channel_map:
        if channel.lower() in ecg_interpretation.lower():
            categorical_data[tm.channel_map[channel]] = 1.0
            return categorical_data
    if 'no_' + tm.name in tm.channel_map:
        categorical_data[tm.channel_map['no_' + tm.name]] = 1.0
        return categorical_data
    else:
        raise ValueError(
            f"ECG categorical interpretation could not find any of these keys: {tm.channel_map.keys()}",
        )


# Extract RAmplitude and SAmplitude for LVH criteria
def _make_ukb_ecg_rest(population_normalize: float = None):
    def ukb_ecg_rest_from_file(tm, hd5, dependents={}):
        if 'ukb_ecg_rest' not in hd5:
            raise ValueError(
                'Group with R and S amplitudes not present in hd5',
            )
        tensor = get_tensor_at_first_date(
            hd5, tm.path_prefix, tm.name, pass_nan,
        )
        try:
            if population_normalize is None:
                tensor = tm.zero_mean_std1(tensor)
            else:
                tensor /= population_normalize
        except:
            ValueError(f'Cannot normalize {tm.name}')
        return tensor
    return ukb_ecg_rest_from_file


def _make_ukb_ecg_rest_lvh():
    def ukb_ecg_rest_lvh_from_file(tm, hd5, dependents={}):
        # Lead order seems constant and standard throughout, but we could eventually tensorize it from XML
        lead_order = ECG_REST_AMP_LEADS
        avl_min = 1100.0
        sl_min = 3500.0
        cornell_female_min = 2000.0
        cornell_male_min = 2800.0
        if 'ukb_ecg_rest' not in hd5:
            raise ValueError(
                'Group with R and S amplitudes not present in hd5',
            )
        tensor_ramp = get_tensor_at_first_date(
            hd5, tm.path_prefix, 'ramplitude', pass_nan,
        )
        tensor_samp = get_tensor_at_first_date(
            hd5, tm.path_prefix, 'samplitude', pass_nan,
        )
        criteria_sleads = [lead_order[l] for l in ['V1', 'V3']]
        criteria_rleads = [lead_order[l] for l in ['aVL', 'V5', 'V6']]
        if np.any(np.isnan(np.union1d(tensor_ramp[criteria_rleads], tensor_samp[criteria_sleads]))):
            raise ValueError(
                'Missing some of the R and S amplitude readings needed to evaluate LVH criteria',
            )
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
            is_lvh = tensor_ramp[lead_order['aVL']] + \
                tensor_samp[lead_order['V3']]
            if is_female:
                is_lvh = is_lvh > cornell_female_min
            if is_male:
                is_lvh = is_lvh > cornell_male_min
        else:
            raise ValueError(
                f'{tm.name} criterion for LVH is not accounted for',
            )
        # Following convention from categorical TMAPS, positive has cmap index 1
        tensor = np.zeros(tm.shape, dtype=np.float32)
        index = 0
        if is_lvh:
            index = 1
        tensor[index] = 1.0
        return tensor
    return ukb_ecg_rest_lvh_from_file


def _ecg_rest_to_segment(population_normalize=None, hertz=500, random_offset_seconds=0):
    def ecg_rest_section_to_segment(tm, hd5, dependents={}):
        tensor = np.zeros(tm.shape, dtype=np.float32)
        segmented = tm.dependent_map.hd5_first_dataset_in_group(
            hd5, tm.dependent_map.hd5_key_guess(),
        )
        offset_seconds = float(segmented.attrs['offset_seconds'])
        random_offset_samples = 0
        if random_offset_seconds > 0:
            random_offset_begin = np.random.uniform(random_offset_seconds)
            offset_seconds += random_offset_begin
            random_offset_samples = int(random_offset_begin * hertz)
        offset_begin = int(offset_seconds * hertz)
        segment_index = np.array(
            segmented[random_offset_samples:random_offset_samples+tm.dependent_map.shape[0]], dtype=np.float32,
        )
        dependents[tm.dependent_map] = to_categorical(
            segment_index, tm.dependent_map.shape[-1],
        )
        for k in hd5[tm.path_prefix]:
            if k in tm.channel_map:
                tensor[:, tm.channel_map[k]] = np.array(hd5[tm.path_prefix][k], dtype=np.float32)[
                                                        offset_begin:offset_begin+tm.shape[0]
                ]
        if population_normalize is None:
            tm.normalization = {'zero_mean_std1': 1.0}
        else:
            tensor /= population_normalize
        return tensor
    return ecg_rest_section_to_segment


ecg_bike_hrr = TensorMap(
    'hrr', path_prefix='ecg_bike', loss='log_cosh', metrics=['mae'], shape=(1,),
    normalization={'mean': 30.55, 'std': 12.81},
    tensor_from_file=_first_date_hrr,
)
ecg_bike_healthy_max_hr = TensorMap(
    'max_hr', path_prefix='ecg_bike', loss='log_cosh', metrics=['mae'],
    normalization={'mean': 113.7, 'std': 13.3}, shape=(1,),
    tensor_from_file=_healthy_bike,
)
ecg_bike_healthy_hrr = TensorMap(
    'hrr', path_prefix='ecg_bike', loss='log_cosh', metrics=['mae'], shape=(1,),
    normalization={'mean': 30.47, 'std': 11.76},
    tensor_from_file=_healthy_hrr,
)
ecg_bike_healthy_resting = TensorMap(
    'resting_hr', path_prefix='ecg_bike', loss='log_cosh', metrics=['mae'], shape=(1,),
    normalization={'mean': 70.0, 'std': 11.62},
    tensor_from_file=_healthy_bike,
)
ecg_bike_med_pretest_hr = TensorMap(
    'trend_heartrate', path_prefix='ecg_bike', loss='log_cosh', metrics=['mae'], shape=(1,),
    normalization={'mean': 70., 'std': 11.},
    tensor_from_file=_median_pretest,
)
ecg_bike_med_pretest_stamp = TensorMap(
    'trend_stamplitude', path_prefix='ecg_bike', loss='log_cosh', metrics=['mae'], shape=(1,),
    normalization={'mean': .03, 'std': .03},
    tensor_from_file=_median_pretest,
)
ecg_bike_med_pretest_jpoint = TensorMap(
    'trend_jpointamplitude', path_prefix='ecg_bike', loss='log_cosh', metrics=['mae'], shape=(1,),
    normalization={'mean': .032, 'std': .46},
    tensor_from_file=_median_pretest,
)
ecg_bike_med_pretest_stamp20 = TensorMap(
    'trend_stamplitude20ms', path_prefix='ecg_bike', loss='log_cosh', metrics=['mae'], shape=(1,),
    normalization={'mean': .03, 'std': .03},
    tensor_from_file=_median_pretest,
)
ecg_bike_recovery = TensorMap(
    'full', shape=(30000, 1), path_prefix='ecg_bike', validator=no_nans,
    tensor_from_file=_first_date_bike_recovery,
)
ecg_bike_pretest = TensorMap(
    'full', shape=(500 * 15 - 4, 3), path_prefix='ecg_bike', validator=no_nans,
    normalization={
        'mean': np.array(
        [7, -7, 3.5],
        )[np.newaxis], 'std': np.array([31, 30, 16])[np.newaxis],
    },
    tensor_from_file=_first_date_bike_pretest,
)
ecg_bike_pretest_5k = TensorMap(
    'full', shape=(5000, 3), path_prefix='ecg_bike', validator=no_nans,
    normalization={
        'mean': np.array(
        [7, -7, 3.5],
        )[np.newaxis], 'std': np.array([31, 30, 16])[np.newaxis],
    },
    tensor_from_file=_first_date_bike_pretest,
)
ecg_bike_new_hrr = TensorMap(
    'hrr', path_prefix='ecg_bike', loss='log_cosh', metrics=['mae'], shape=(1,),
    normalization={'mean': 31, 'std': 12},
    tensor_from_file=_new_hrr,
)
ecg_bike_hrr_sentinel = TensorMap(
    'hrr', path_prefix='ecg_bike', metrics=['mae'], shape=(1,),
    normalization={'mean': 31, 'std': 12}, sentinel=_HRR_SENTINEL,
    tensor_from_file=_sentinel_hrr,
)
ecg_bike_hrr_student = TensorMap(
    'hrr', path_prefix='ecg_bike', metrics=['mae'], shape=(1,),
    normalization={'mean': 31, 'std': 12}, sentinel=_HRR_SENTINEL,
    tensor_from_file=build_tensor_from_file(
        'inference.tsv', 'ecg_bike_hrr-sentinel_prediction',
    ),
)
ecg_bike_hr_achieved = TensorMap(
    'hr_achieved', path_prefix='ecg_bike', loss='log_cosh', metrics=['mae'], shape=(1,),
    normalization={'mean': .68, 'std': .1},
    tensor_from_file=_hr_achieved,
)

ecg_rest_raw = TensorMap(
    'ecg_rest_raw', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
    channel_map=ECG_REST_LEADS, normalization=Standardize(mean=0, std=2000),
)

ecg_rest_raw_mv = TensorMap(
    'ecg_rest_raw_mv', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
    channel_map=ECG_REST_LEADS, normalization=Standardize(mean=0, std=1000),
)
ecg_rest_raw_mv_warp = TensorMap(
    'ecg_rest_raw_mv_warp', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
    channel_map=ECG_REST_LEADS, normalization=Standardize(mean=0, std=1000), augmentations=[_warp_ecg],
)

ecg_rest_raw_100 = TensorMap(
    'ecg_rest_raw_100', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
    channel_map=ECG_REST_LEADS, normalization=Standardize(mean=0, std=100),
)
ecg_rest_raw_8s = TensorMap(
    'ecg_rest_raw_8s', Interpretation.CONTINUOUS, shape=(4096, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
    channel_map=ECG_REST_LEADS, normalization=Standardize(mean=0, std=10),
)

ecg_rest_raw_10 = TensorMap(
    'ecg_rest_raw_10', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
    channel_map=ECG_REST_LEADS, normalization=Standardize(mean=0, std=10),
)


ecg_rest = TensorMap(
    'strip', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
    channel_map=ECG_REST_LEADS, normalization=ZeroMeanStd1(), loss='log_cosh', activation='linear', metrics=['mse', 'mae'],
)
ecg_rest_8s = TensorMap(
    'ecg_rest_8s', Interpretation.CONTINUOUS, shape=(4096, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
    channel_map=ECG_REST_LEADS, normalization=ZeroMeanStd1(),
)
ecg_rest_no_poor = TensorMap(
    'strip', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(skip_poor=True),
    channel_map=ECG_REST_LEADS, normalization=ZeroMeanStd1(),
)

ecg_rest_mgb = TensorMap(
    'ecg_5000_std', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
    channel_map=ECG_REST_AMP_LEADS_UKB, normalization=ZeroMeanStd1(),
)

ecg_rest_mgb_mv = TensorMap(
    'ecg_5000_mv', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
    channel_map=ECG_REST_AMP_LEADS_UKB, normalization=Standardize(mean=0, std=150),
)
mgb_ecg_rest_mv = TensorMap(
    'mgb_ecg_5000_mv', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
    channel_map=ECG_REST_AMP_LEADS_UKB, normalization=Standardize(mean=0, std=150),
)
ecg_rest_mgb_2500 = TensorMap(
    'ecg_2500_std', Interpretation.CONTINUOUS, shape=(2500, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(downsample_steps=2),
    channel_map=ECG_REST_AMP_LEADS_UKB, normalization=ZeroMeanStd1(),
)
mgb_ecg_rest_2500 = TensorMap(
    'mgb_ecg_2500_std', Interpretation.CONTINUOUS, shape=(2500, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(downsample_steps=2),
    channel_map=ECG_REST_AMP_LEADS_UKB, normalization=ZeroMeanStd1(),
)
partners_ecg_2500 = TensorMap(
    'partners_ecg_2500_newest', Interpretation.CONTINUOUS, shape=(2500, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
    channel_map=ECG_REST_AMP_LEADS_UKB, normalization=ZeroMeanStd1(),
)

ecg_rest_mgb_strip_I = TensorMap(
    'ecg_strip_I', Interpretation.CONTINUOUS, shape=(5000, 1), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
    channel_map={'strip_I': 0}, normalization=ZeroMeanStd1(),
)
ecg_rest_mgb_strip_II = TensorMap(
    'ecg_strip_II', Interpretation.CONTINUOUS, shape=(5000, 1), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
    channel_map={'strip_II': 0}, normalization=ZeroMeanStd1(),
)

rest_strip_I_random = TensorMap(
    'ecg_strip_I', Interpretation.CONTINUOUS, shape=(4096, 1), path_prefix='ukb_ecg_rest',
    tensor_from_file=_make_ecg_rest(random_offset=904),
    channel_map={'strip_I': 0}, normalization=ZeroMeanStd1(),
)
rest_strip_II_random = TensorMap(
    'ecg_strip_II', Interpretation.CONTINUOUS, shape=(4096, 1), path_prefix='ukb_ecg_rest',
    tensor_from_file=_make_ecg_rest(random_offset=904),
    channel_map={'strip_II': 0}, normalization=ZeroMeanStd1(),
)
rest_strip_I_random_1s = TensorMap(
    'rest_strip_I_random_1s', Interpretation.CONTINUOUS, shape=(500, 1), path_prefix='ukb_ecg_rest',
    tensor_from_file=_make_ecg_rest(random_offset=4500),
    channel_map={'strip_I': 0}, normalization=ZeroMeanStd1(),
)
rest_strip_I_random_2s = TensorMap(
    'rest_strip_I_random_2s', Interpretation.CONTINUOUS, shape=(1000, 1), path_prefix='ukb_ecg_rest',
    tensor_from_file=_make_ecg_rest(random_offset=4000),
    channel_map={'strip_I': 0}, normalization=ZeroMeanStd1(),
)
rest_strip_II_random_2s = TensorMap(
    'rest_strip_II_random_2s', Interpretation.CONTINUOUS, shape=(1000, 1), path_prefix='ukb_ecg_rest',
    tensor_from_file=_make_ecg_rest(random_offset=4000),
    channel_map={'strip_II': 0}, normalization=ZeroMeanStd1(),
)
ecg_rest_mgb_2500_test = TensorMap(
    'ecg_2500_std', Interpretation.CONTINUOUS, shape=(2500, 12), path_prefix='ukb_ecg_rest',
    tensor_from_file=_make_ecg_rest(downsample_steps=2), channel_map=ECG_REST_AMP_LEADS_UKB, normalization=ZeroMeanStd1(),
)

ecg_rest_mgb_2500_raw = TensorMap(
    'partners_ecg_2500_newest', Interpretation.CONTINUOUS, shape=(2500, 12), path_prefix='ukb_ecg_rest',
    tensor_from_file=_make_ecg_rest(downsample_steps=2), channel_map=ECG_REST_MGB_LEADS, normalization=Standardize(mean=0, std=2000),
)
ecg_rest_mse10x = TensorMap(
    'strip', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
    channel_map=ECG_REST_LEADS, normalization=ZeroMeanStd1(), loss=mse_10x,
)
ecg_rest_2500_ukb = TensorMap(
    'ecg_rest_2500', Interpretation.CONTINUOUS, shape=(2500, 12), path_prefix='ukb_ecg_rest', channel_map=ECG_REST_LEADS,
    tensor_from_file=_make_ecg_rest(downsample_steps=2), normalization=ZeroMeanStd1(),
)

ecg_rest_stft = TensorMap(
    'ecg_rest_stft', Interpretation.CONTINUOUS, shape=(52, 72, 12), path_prefix='ukb_ecg_rest', channel_map=ECG_REST_LEADS,
    tensor_from_file=_make_ecg_rest(short_time_nperseg=103, short_time_noverlap=32), normalization=ZeroMeanStd1(),
)
ecg_rest_stft_512 = TensorMap(
    'ecg_rest_stft_512', shape=(257, 314, 12), path_prefix='ukb_ecg_rest', channel_map=ECG_REST_LEADS,
    tensor_from_file=_make_ecg_rest(short_time_nperseg=512, short_time_noverlap=496), normalization=ZeroMeanStd1(),
)

ecg_rest_stack = TensorMap(
    'strip', Interpretation.CONTINUOUS, shape=(600, 12, 8), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
    channel_map=ECG_REST_LEADS, normalization=ZeroMeanStd1(),
)

ecg_rest_median_raw = TensorMap(
    'median', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(600, 12), loss='log_cosh', activation='linear', tensor_from_file=_make_ecg_rest(),
    metrics=['mse', 'mae'], channel_map=ECG_REST_MEDIAN_LEADS, normalization=Standardize(mean=0, std=2000),
)
ecg_rest_median_lead_I = TensorMap(
    'ecg_rest_median_lead_I', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest',
    shape=(600, 1), loss='log_cosh', activation='linear', tensor_from_file=_make_ecg_rest(),
    metrics=['mse', 'mae'], channel_map={'median_I': 0}, normalization=ZeroMeanStd1(),
)
ecg_rest_median_lead_II = TensorMap(
    'ecg_rest_median_lead_II', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest',
    shape=(600, 1), loss='log_cosh', activation='linear', tensor_from_file=_make_ecg_rest(),
    metrics=['mse', 'mae'], channel_map={'median_II': 0}, normalization=ZeroMeanStd1(),
)
ecg_rest_median_mv = TensorMap(
    'median', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(600, 12), loss='log_cosh', activation='linear', tensor_from_file=_make_ecg_rest(),
    metrics=['mse', 'mae'], channel_map=ECG_REST_MEDIAN_LEADS, normalization=Standardize(mean=0, std=1000),
)
ecg_rest_median_mv_warp = TensorMap(
    'median_warp', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(600, 12), loss='log_cosh', activation='linear', tensor_from_file=_make_ecg_rest(),
    metrics=['mse', 'mae'], channel_map=ECG_REST_MEDIAN_LEADS, normalization=Standardize(mean=0, std=1000), augmentations=[_warp_ecg],
)

ecg_rest_median_raw_10 = TensorMap(
    'ecg_rest_median_raw_10', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(600, 12), loss='log_cosh',
    activation='linear', tensor_from_file=_make_ecg_rest(),
    metrics=['mse', 'mae'], channel_map=ECG_REST_MEDIAN_LEADS, normalization=Standardize(mean=0, std=10),
)


ecg_rest_median_576 = TensorMap(
    'ecg_rest_median_576', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(576, 12), loss='log_cosh',
    activation='linear', tensor_from_file=_make_ecg_rest(), channel_map=ECG_REST_MEDIAN_LEADS,
    normalization=Standardize(mean=0, std=10),
)

ecg_rest_median_512 = TensorMap(
    'ecg_rest_median_512', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(512, 12), loss='logcosh',
    activation='linear', tensor_from_file=_make_ecg_rest(), channel_map=ECG_REST_MEDIAN_LEADS,
    normalization=ZeroMeanStd1(),
)

ecg_rest_median_raw_10_no_poor = TensorMap(
    'ecg_rest_median_raw_10', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(600, 12), loss='log_cosh', activation='linear',
    tensor_from_file=_make_ecg_rest(skip_poor=True), metrics=['mse', 'mae'], channel_map=ECG_REST_MEDIAN_LEADS, normalization=Standardize(mean=0, std=10),
)
ecg_rest_median_raw_10_random_norm = TensorMap(
    'ecg_rest_median_raw_10', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(600, 12), loss='log_cosh', activation='linear',
    tensor_from_file=_make_ecg_rest(skip_poor=True),
    metrics=['mse', 'mae', 'log_cosh'], channel_map=ECG_REST_MEDIAN_LEADS, normalization=RandomStandardize(mean=0, std=10),
)
ecg_rest_median = TensorMap(
    'median', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(600, 12), loss='log_cosh', activation='linear', tensor_from_file=_make_ecg_rest(),
    metrics=['mse', 'mae'], channel_map=ECG_REST_MEDIAN_LEADS, normalization=ZeroMeanStd1(),
)

ecg_rest_median_raw_10_prediction = TensorMap(
    'ecg_rest_median_raw_10', Interpretation.CONTINUOUS, shape=(600, 12), loss='log_cosh', activation='linear',
    normalization=ZeroMeanStd1(),
    tensor_from_file=named_tensor_from_hd5('ecg_rest_median_raw_10_prediction'), metrics=['mse', 'mae'],
    channel_map=ECG_REST_MEDIAN_LEADS,
)


def stretch_ecg(x, n=0):
    """
    stretches input ECG to n bpm
    """
    out = biosppy.signals.ecg.ecg(x.copy(), sampling_rate=500, show=False)
    hr = out[-1].mean()
    t = np.arange(len(x))
    if n == 0:
        tp = np.arange(len(x))
    else:
        tp = np.arange(len(x)) * n / hr
    stretched = np.interp(tp, t, x)
    out2 = biosppy.signals.ecg.ecg(stretched, show=False)
    return stretched, out2[2]


def make_biosspy_median(example, channel_map, median_size = 600, bpm = 0):
    medians = np.zeros((median_size, len(channel_map)))
    for lead in channel_map:
        waves = []

        stretched, peaks = stretch_ecg(example[:,channel_map[lead]], bpm)

        for j, p0 in enumerate(peaks[:-2]):
            p11 = peaks[j+1]
            middle = (p0+p11)//2
            waves.append(stretched[middle:middle+median_size])
        waves = np.array(waves)
        medians[:, channel_map[lead]] = np.median(waves, axis=0)

    return medians


def ecg_median_biosppy(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}) -> np.ndarray:
    tensor = np.zeros(tm.shape, dtype=np.float32)
    for lead in tm.channel_map:
        tensor[:, tm.channel_map[lead]] = hd5[f'{tm.path_prefix}{lead}']
    tensor = np.nan_to_num(tensor)
    return tensor


ecg_biosppy_median = TensorMap(
    'median', Interpretation.CONTINUOUS, path_prefix='median_', shape=(600, 12), tensor_from_file=ecg_median_biosppy,
    channel_map=ECG_REST_LEADS,
)

ecg_biosppy_median_60bpm = TensorMap(
    'median', Interpretation.CONTINUOUS, path_prefix='median_60bpm_', shape=(600, 12), tensor_from_file=ecg_median_biosppy,
    channel_map=ECG_REST_LEADS,
)

ecg_biosppy_median_60bpm_lead_I = TensorMap(
    'median', Interpretation.CONTINUOUS, path_prefix='median_60bpm_', shape=(600, 12), tensor_from_file=ecg_median_biosppy,
    channel_map={'strip_I': 0},
)

ecg_biosppy_median_mgb = TensorMap(
    'median', Interpretation.CONTINUOUS, path_prefix='median_', shape=(600, 12), tensor_from_file=ecg_median_biosppy,
    channel_map=ECG_REST_AMP_LEADS,
)

ecg_biosppy_median_60bpm_mgb = TensorMap(
    'median', Interpretation.CONTINUOUS, path_prefix='median_60bpm_', shape=(600, 12), tensor_from_file=ecg_median_biosppy,
    channel_map=ECG_REST_AMP_LEADS,
)
ecg_biosppy_median_60bpm_mgb_lead_I = TensorMap(
    'median', Interpretation.CONTINUOUS, path_prefix='median_60bpm_', shape=(600, 12),
    tensor_from_file=ecg_median_biosppy,
    channel_map={'I': 0},
)


def ecg_median_biosppy_on_the_fly(ecg_10s_shape=(5000, 12), bpm=0, instance=2):
    def _ecg_median_tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}) -> np.ndarray:
        ecg_10s = np.zeros(ecg_10s_shape, dtype=np.float32)
        for k in hd5[tm.path_prefix]:
            if k in tm.channel_map:
                data = tm.hd5_first_dataset_in_group(
                    hd5, f'{tm.path_prefix}/{k}/instance_{instance}',
                )
                ecg_10s[:, tm.channel_map[k]] = pad_or_crop_array_to_shape((ecg_10s_shape[0],), data)
        return make_biosspy_median(ecg_10s, tm.channel_map, median_size=tm.shape[0], bpm=bpm)
    return _ecg_median_tensor_from_file


ecg_biosppy_median_otf = TensorMap(
    'median', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(600, 12), tensor_from_file=ecg_median_biosppy_on_the_fly(),
    channel_map=ECG_REST_LEADS, normalization=ZeroMeanStd1(),
)
ecg_biosppy_median_60bpm_otf = TensorMap(
    'median', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(600, 12), tensor_from_file=ecg_median_biosppy_on_the_fly(bpm=60),
    channel_map=ECG_REST_LEADS, normalization=ZeroMeanStd1(),
)


def ecg_prediction_lead_from_hd5(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}) -> np.ndarray:
    ecg = np.array(hd5['ecg_rest_median_raw_10_prediction'], dtype=np.float32)
    tensor = np.zeros(tm.shape, dtype=np.float32)
    tensor[:, 0] = ecg[:, 0]
    return tensor


ecg_rest_median_raw_10_prediction_lead_I = TensorMap(
    'ecg_rest_median_raw_10_lead_I', Interpretation.CONTINUOUS, shape=(600, 1), loss='log_cosh', activation='linear',
    tensor_from_file=ecg_prediction_lead_from_hd5, metrics=['mse', 'mae'], channel_map={'median_I': 0},
)

ecg_rest_median_raw_10_truth = TensorMap(
    'ecg_rest_median_raw_10_truth', Interpretation.CONTINUOUS, shape=(600, 12), loss='log_cosh', activation='linear',
    tensor_from_file=tensor_from_hd5, metrics=['mse', 'mae'], channel_map=ECG_REST_MEDIAN_LEADS,
)

ecg_rest_median_stack = TensorMap(
    'median', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(600, 12, 1), activation='linear', tensor_from_file=_make_ecg_rest(),
    metrics=['mse', 'mae', 'log_cosh'], loss='log_cosh', loss_weight=1.0,
    channel_map=ECG_REST_MEDIAN_LEADS, normalization=ZeroMeanStd1(),
)

ecg_median_1lead = TensorMap(
    'median', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(600, 1), loss='log_cosh', loss_weight=10.0, tensor_from_file=_make_ecg_rest(),
    activation='linear', metrics=['mse', 'mae', 'log_cosh'], channel_map={'lead': 0}, normalization=ZeroMeanStd1(),
)

ecg_rest_1lead = TensorMap(
    'strip', Interpretation.CONTINUOUS, shape=(600, 8), path_prefix='ukb_ecg_rest', channel_map={'lead': 0}, tensor_from_file=_make_ecg_rest(),
    dependent_map=ecg_median_1lead, normalization=ZeroMeanStd1(),
)

ecg_median_1lead_categorical = TensorMap(
    'median',  Interpretation.CATEGORICAL, shape=(600, 32), activation='softmax', tensor_from_file=_make_ecg_rest(),
    channel_map=_get_lead_cm(32)[0],  normalization=ZeroMeanStd1(),
    loss=weighted_crossentropy(
        np.array(_get_lead_cm(32)[1]), 'ecg_median_categorical',
    ),
)

ecg_rest_1lead_categorical = TensorMap(
    'strip', shape=(600, 8), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
    normalization=ZeroMeanStd1(),
    channel_map={
        'window0': 0, 'window1': 1, 'window2': 2, 'window3': 3,
        'window4': 4, 'window5': 5, 'window6': 6, 'window7': 7,
    },
    dependent_map=ecg_median_1lead_categorical,
)

ecg_rhythm = TensorMap(
    'ecg_rhythm', Interpretation.CATEGORICAL, tensor_from_file=_make_rhythm_tensor(),
    loss=weighted_crossentropy([1.0, 2.0, 3.0, 3.0, 20.0, 20.0], 'ecg_rhythm'),
    channel_map={
        'Normal_sinus_rhythm': 0, 'Sinus_bradycardia': 1, 'Marked_sinus_bradycardia': 2,
        'Other_sinus_rhythm': 3, 'Atrial_fibrillation': 4, 'Other_rhythm': 5,
    },
)
ecg_rhythm_poor = TensorMap(
    'ecg_rhythm', Interpretation.CATEGORICAL, tensor_from_file=_make_rhythm_tensor(False),
    loss=weighted_crossentropy(
        [1.0, 2.0, 3.0, 3.0, 20.0, 20.0], 'ecg_rhythm_poor',
    ),
    channel_map={
        'Normal_sinus_rhythm': 0, 'Sinus_bradycardia': 1, 'Marked_sinus_bradycardia': 2,
        'Other_sinus_rhythm': 3, 'Atrial_fibrillation': 4, 'Other_rhythm': 5,
    },
)

ecg_rest_age = TensorMap(
    'ecg_rest_age', Interpretation.CONTINUOUS, tensor_from_file=age_in_years_tensor('ecg_rest_date'), loss='log_cosh',
    channel_map={'ecg_rest_age': 0}, validator=make_range_validator(0, 110), normalization={'mean': 65, 'std': 7.7},
)

acute_mi = TensorMap(
    'acute_mi', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text, channel_map={'no_acute_mi': 0, 'ACUTE MI': 1},
    loss=weighted_crossentropy([0.1, 10.0], 'acute_mi'),
)

anterior_blocks = TensorMap(
    'anterior_blocks', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text,
    channel_map={
        'no_anterior_blocks': 0, 'Left anterior fascicular block': 1,
        'Left posterior fascicular block': 2,
    },
    loss=weighted_crossentropy([0.1, 10.0, 10.0], 'anterior_blocks'),
)

av_block = TensorMap(
    'av_block', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text, channel_map={'no_av_block': 0, 'st degree AV block': 1},
    loss=weighted_crossentropy([0.1, 10.0], 'av_block'),
)

incomplete_right_bundle_branch_block = TensorMap(
    'incomplete_right_bundle_branch_block', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text,
    channel_map={
        'no_incomplete_right_bundle_branch_block': 0,
        'Incomplete right bundle branch block': 1,
    },
    loss=weighted_crossentropy(
        [0.1, 10.0], 'incomplete_right_bundle_branch_block',
    ),
)

infarcts = TensorMap(
    'infarcts', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text,
    channel_map={
        'no_infarcts': 0, 'Anterior infarct': 1, 'Anteroseptal infarct': 2,
        'Inferior infarct': 3, 'Lateral infarct': 4, 'Septal infarct': 5,
    },
    loss=weighted_crossentropy([0.1, 4.0, 6.0, 7.0, 6.0, 4.0], 'infarcts'),
)

left_atrial_enlargement = TensorMap(
    'left_atrial_enlargement', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text,
    channel_map={
        'no_left_atrial_enlargement': 0,
        'Left atrial enlargement': 1,
    },
    loss=weighted_crossentropy([0.1, 10.0], 'left_atrial_enlargement'),
)

left_ventricular_hypertrophy = TensorMap(
    'left_ventricular_hypertrophy', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text,
    channel_map={
        'no_left_ventricular_hypertrophy': 0,
        'Left ventricular hypertrophy': 1,
    },
    loss=weighted_crossentropy([0.1, 10.0], 'left_ventricular_hypertrophy'),
)

lvh_fine = TensorMap(
    'lvh_fine', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text, loss=weighted_crossentropy([0.5, 12.0, 16.0, 30.0, 36.0], 'lvh_fine'),
    channel_map={
        'no_lvh_fine': 0, 'Minimal voltage criteria for LVH may be normal variant': 1,
        'Moderate voltage criteria for LVH may be normal variant': 2, 'Voltage criteria for left ventricular hypertrophy': 3,
        'Left ventricular hypertrophy': 4,
    },
)



premature_atrial_complexes = TensorMap(
    'premature_atrial_complexes', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text,
    channel_map={
        'no_premature_atrial_complexes': 0,
        'premature atrial complexes': 1,
    },
    loss=weighted_crossentropy([0.1, 10.0], 'premature_atrial_complexes'),
)

premature_supraventricular_complexes = TensorMap(
    'premature_supraventricular_complexes', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text,
    channel_map={
        'no_premature_supraventricular_complexes': 0,
        'premature supraventricular complexes': 1,
    },
    loss=weighted_crossentropy(
        [0.1, 10.0], 'premature_supraventricular_complexes',
    ),
)

premature_ventricular_complexes = TensorMap(
    'premature_ventricular_complexes', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text,
    channel_map={
        'no_premature_ventricular_complexes': 0,
        'premature ventricular complexes': 1,
    },
    loss=weighted_crossentropy([0.1, 10.0], 'premature_ventricular_complexes'),
)

prolonged_qt = TensorMap(
    'prolonged_qt', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text, channel_map={'no_prolonged_qt': 0, 'Prolonged QT': 1},
    loss=weighted_crossentropy([0.1, 10.0], 'prolonged_qt'),
)


ecg_rest_ramplitude_raw = TensorMap(
    'ramplitude', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(12,), tensor_from_file=_make_ukb_ecg_rest(1.0),
    loss='log_cosh', metrics=['mse', 'mape', 'mae'], loss_weight=1.0,
)

ecg_rest_samplitude_raw = TensorMap(
    'samplitude', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(12,), tensor_from_file=_make_ukb_ecg_rest(1.0),
    loss='log_cosh', metrics=['mse', 'mape', 'mae'], loss_weight=1.0,
)

ecg_rest_ramplitude = TensorMap(
    'ramplitude', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(12,), tensor_from_file=_make_ukb_ecg_rest(),
    loss='log_cosh', metrics=['mse', 'mape', 'mae'], loss_weight=1.0,
)

ecg_rest_samplitude = TensorMap(
    'samplitude', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(12,), tensor_from_file=_make_ukb_ecg_rest(),
    loss='log_cosh', metrics=['mse', 'mape', 'mae'], loss_weight=1.0,
)


ecg_rest_lvh_avl = TensorMap(
    'avl_lvh', Interpretation.CATEGORICAL, path_prefix='ukb_ecg_rest', tensor_from_file=_make_ukb_ecg_rest_lvh(),
    channel_map={'no_avl_lvh': 0, 'aVL LVH': 1},
    loss=weighted_crossentropy([0.006, 1.0], 'avl_lvh'),
)

ecg_rest_lvh_sokolow_lyon = TensorMap(
    'sokolow_lyon_lvh', Interpretation.CATEGORICAL, path_prefix='ukb_ecg_rest', tensor_from_file=_make_ukb_ecg_rest_lvh(),
    channel_map={'no_sokolow_lyon_lvh': 0, 'Sokolow Lyon LVH': 1},
    loss=weighted_crossentropy([0.005, 1.0], 'sokolov_lyon_lvh'),
)

ecg_rest_lvh_cornell = TensorMap(
    'cornell_lvh', Interpretation.CATEGORICAL, path_prefix='ukb_ecg_rest', tensor_from_file=_make_ukb_ecg_rest_lvh(),
    channel_map={'no_cornell_lvh': 0, 'Cornell LVH': 1},
    loss=weighted_crossentropy([0.003, 1.0], 'cornell_lvh'),
)


ecg_segmented = TensorMap(
    'ecg_segmented', Interpretation.CATEGORICAL, shape=(1224, len(ECG_SEGMENTED_CHANNEL_MAP)), path_prefix='ecg_rest',
    cacheable=False, channel_map=ECG_SEGMENTED_CHANNEL_MAP,
)
ecg_section_to_segment = TensorMap(
    'ecg_section_to_segment', shape=(1224, 12), path_prefix='ecg_rest', dependent_map=ecg_segmented,
    channel_map=ECG_REST_LEADS, tensor_from_file=_ecg_rest_to_segment(),
)
ecg_section_to_segment_warp = TensorMap(
    'ecg_section_to_segment', shape=(1224, 12), path_prefix='ecg_rest', dependent_map=ecg_segmented,
    cacheable=False, channel_map=ECG_REST_LEADS, tensor_from_file=_ecg_rest_to_segment(),
    augmentations=[_warp_ecg],
)

ecg_segmented_second = TensorMap(
    'ecg_segmented', Interpretation.CATEGORICAL, shape=(496, len(ECG_SEGMENTED_CHANNEL_MAP)), path_prefix='ecg_rest',
    cacheable=False, channel_map=ECG_SEGMENTED_CHANNEL_MAP,
)
ecg_second_to_segment = TensorMap(
    'ecg_second_to_segment', shape=(496, 12), path_prefix='ecg_rest', dependent_map=ecg_segmented_second,
    cacheable=False, channel_map=ECG_REST_LEADS, tensor_from_file=_ecg_rest_to_segment(random_offset_seconds=1.5),
)
ecg_second_to_segment_warp = TensorMap(
    'ecg_second_to_segment', shape=(496, 12), path_prefix='ecg_rest', dependent_map=ecg_segmented_second,
    cacheable=False, channel_map=ECG_REST_LEADS, tensor_from_file=_ecg_rest_to_segment(random_offset_seconds=1.5),
    augmentations=[_warp_ecg],
)

poor_data_quality = TensorMap(
    'poor_data_quality', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text, channel_map={'no_poor_data_quality': 0, 'Poor data quality': 1},
    loss=weighted_crossentropy([0.1, 3.0], 'poor_data_quality'),
)

####
ecg_semi_coarse = TensorMap(
    'ecg_semi_coarse', Interpretation.CATEGORICAL, loss=weighted_crossentropy([1.0, 1.0, 2.0, 4.0, 16.0, 20.0], 'ecg_semi_coarse'),
    channel_map={'Normal_sinus_rhythm': 0, 'Sinus_bradycardia': 1, 'Marked_sinus_bradycardia': 2, 'Other_sinus_rhythm': 3, 'Atrial_fibrillation': 4, 'Other_rhythm': 5},
)


ecg_semi_coarse_with_poor = TensorMap(
    'ecg_semi_coarse_with_poor', Interpretation.CATEGORICAL, loss=weighted_crossentropy([1.0, 2.0, 3.0, 3.0, 20.0, 20.0], 'ecg_semi_coarse_with_poor'),
    channel_map={'Normal_sinus_rhythm': 0, 'Sinus_bradycardia': 1, 'Marked_sinus_bradycardia': 2, 'Other_sinus_rhythm': 3, 'Atrial_fibrillation': 4, 'Other_rhythm': 5},
)

normal = TensorMap(
    'normal', Interpretation.CATEGORICAL, loss=weighted_crossentropy([2.0, 3.0, 3.0, 3.0], 'ecg_normal'),
    tensor_from_file=label_from_ecg_interpretation_text,
    channel_map={'Normal ECG': 0, 'Abnormal ECG': 1, 'Borderline ECG': 2, 'Otherwise normal ECG': 3},
)
infarct = TensorMap(
    'infarct', Interpretation.CATEGORICAL, channel_map={'no_infarct': 0, 'infarct': 1},
    tensor_from_file=label_from_ecg_interpretation_text,
    loss=weighted_crossentropy([1.0, 8.0], 'ecg_infarct'),
)
ecg_poor_data = TensorMap(
    'ecg_poor_data', Interpretation.CATEGORICAL, channel_map={'no_poor_data_quality': 0, 'poor_data_quality': 1},
    tensor_from_file=label_from_ecg_interpretation_text,
    loss=weighted_crossentropy([1.0, 8.0], 'ecg_poor_data'),
)
block = TensorMap(
    'block', Interpretation.CATEGORICAL, channel_map={'no_block': 0, 'block': 1},
    tensor_from_file=label_from_ecg_interpretation_text,
    loss=weighted_crossentropy([1.0, 8.0], 'ecg_block'),
)

ecg_rest_next_char = TensorMap('ecg_rest_next_char', Interpretation.LANGUAGE, shape=(len(ECG_CHAR_2_IDX),), channel_map=ECG_CHAR_2_IDX, activation='softmax', loss='categorical_crossentropy', loss_weight=2.0)
ecg_rest_text = TensorMap('ecg_rest_text', Interpretation.LANGUAGE, shape=(100, len(ECG_CHAR_2_IDX)), path_prefix='ukb_ecg_rest', channel_map={'context': 0, 'alphabet': 1}, dependent_map=ecg_rest_next_char)

p_axis = TensorMap(
    'PAxis', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'PAxis': 0}, loss='log_cosh', validator=make_range_validator(-50, 130),
    normalization={'mean': 48.7, 'std': 23.1},
)
p_duration = TensorMap(
    'PDuration', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'PDuration': 0}, loss='log_cosh', validator=make_range_validator(30, 140),
    normalization={'mean': 96.1, 'std': 18.85},
)
p_offset = TensorMap(
    'POffset', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'POffset': 0}, loss='log_cosh', validator=make_range_validator(200, 500),
    normalization={'mean': 369.1, 'std': 28.42},
)
p_onset = TensorMap(
    'POnset', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'POnset': 0}, loss='log_cosh', validator=make_range_validator(120, 400),
    normalization={'mean': 275.1, 'std': 26.420},
)
pp_interval = TensorMap(
    'PPInterval', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'PPInterval': 0}, loss='log_cosh', validator=make_range_validator(300, 1800),
    normalization={'mean': 1036.1, 'std': 185.0},
)
pq_interval = TensorMap(
    'PQInterval', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'PQInterval': 0}, loss='log_cosh', validator=make_range_validator(70, 400),
    normalization={'mean': 165.9, 'std': 26.3},
)
q_offset = TensorMap(
    'QOffset', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'QOffset': 0}, loss='log_cosh', validator=make_range_validator(300, 600),
    normalization={'mean': 525.1, 'std': 13.52},
)
q_onset = TensorMap(
    'QOnset', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'QOnset': 0}, loss='log_cosh', validator=make_range_validator(370, 600),
    normalization={'mean': 435.1, 'std': 11.420},
)
qrs_complexes = TensorMap(
    'QRSComplexes', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'QRSComplexes': 0}, loss='log_cosh', validator=make_range_validator(0, 60),
    normalization={'mean': 8.0, 'std': 20.0},
)
qrs_duration = TensorMap(
    'QRSDuration', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'QRSDuration': 0}, loss='log_cosh', validator=make_range_validator(45, 175),
    normalization={'mean': 89.53, 'std': 12.21},
)
qrs_num = TensorMap(
    'QRSNum', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'QRSNum': 0}, loss='log_cosh', validator=make_range_validator(2, 30),
    normalization={'mean': 9.61, 'std': 1.64},
)
qt_interval = TensorMap(
    'QTInterval', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'QTInterval': 0}, loss='log_cosh', validator=make_range_validator(300, 600),
    normalization=Standardize(mean=426.1, std=32.24),
)
qt_interval_quintiles = TensorMap(
    'QTInterval', Interpretation.DISCRETIZED, path_prefix='ukb_ecg_rest',
    channel_map={'QTInterval': 0}, normalization={'mean': 426.1, 'std': 32.24},
    discretization_bounds=[-0.842, -0.253, 0.253, 0.842],
)
qtc_interval = TensorMap(
    'QTCInterval', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'QTCInterval': 0}, loss='log_cosh', validator=make_range_validator(300, 600),
    normalization={'mean': 419.1, 'std': 20.7},
)
r_axis = TensorMap(
    'RAxis', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'RAxis': 0}, loss='log_cosh', validator=make_range_validator(-100, 200),
    normalization={'mean': 25.7, 'std': 36.6},
)
rr_interval = TensorMap(
    'RRInterval', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'RRInterval': 0}, loss='log_cosh',
    validator=make_range_validator(400, 2000), normalization={'mean': 1040.61, 'std': 175.5}, metrics=[],
)
ventricular_rate = TensorMap(
    'VentricularRate', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'VentricularRate': 0}, validator=make_range_validator(30, 150),
    loss='log_cosh', normalization={'mean': 59.3, 'std': 10.6},
)
t_offset = TensorMap(
    'TOffset', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'TOffset': 0}, loss='log_cosh', validator=make_range_validator(700, 1000),
    normalization={'mean': 860.7, 'std': 32.52},
)
t_axis = TensorMap(
    'TAxis', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'TAxis': 0}, loss='log_cosh', validator=make_range_validator(-100, 200),
    normalization={'mean': 40.8, 'std': 32.6},
)

af_prs = TensorMap('AF_PRS_LDscore', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'AF_PRS_LDscore': 0}, normalization={'mean': -1.0, 'std': 0.4})
charge = TensorMap(
    'charge', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'charge': 0}, normalization={'mean': 12.0, 'std': 2.0},
    validator=make_range_validator(0, 20),
)

qtc_intervalp = TensorMap(
    'QTCInterval', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'QTCInterval': 0}, loss='log_cosh', validator=make_range_validator(100, 900),
    parents=[qt_interval, rr_interval], normalization={'mean': 419.1, 'std': 20.7},
)
qrs_durationpp = TensorMap(
    'QRSDuration', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'QRSDuration': 0}, loss='log_cosh', validator=make_range_validator(45, 175),
    normalization={'mean': 89.53, 'std': 12.21},
    parents=[qtc_intervalp],
)

p_axis_sentinel = TensorMap(
    'PAxis', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'PAxis': 0}, sentinel=0, metrics=['log_cosh'],
    normalization={'mean': 48.7, 'std': 23.1},
)
p_duration_sentinel = TensorMap(
    'PDuration', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'PDuration': 0}, sentinel=0, metrics=['log_cosh'],
    normalization={'mean': 96.1, 'std': 18.85},
)
p_offset_sentinel = TensorMap(
    'POffset', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'POffset': 0}, sentinel=0, metrics=['log_cosh'],
    normalization={'mean': 369.1, 'std': 28.42},
)
p_onset_sentinel = TensorMap(
    'POnset', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'POnset': 0}, sentinel=0, metrics=['log_cosh'],
    normalization={'mean': 275.1, 'std': 26.420},
)
pp_interval_sentinel = TensorMap(
    'PPInterval', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'PPInterval': 0}, sentinel=0, metrics=['log_cosh'],
    normalization={'mean': 1036.1, 'std': 185.0},
)
pq_interval_sentinel = TensorMap(
    'PQInterval', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'PQInterval': 0}, sentinel=0, metrics=['log_cosh'],
    normalization={'mean': 165.9, 'std': 26.3},
)
qrs_duration_sentinel = TensorMap(
    'QRSDuration', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'QRSDuration': 0}, sentinel=0,
    normalization={'mean': 89.53, 'std': 12.21},
)
qt_interval_sentinel = TensorMap(
    'QTInterval', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'QTInterval': 0}, sentinel=0,
    normalization={'mean': 426.1, 'std': 32.24},
)
qtc_interval_sentinel = TensorMap(
    'QTCInterval', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'QTCInterval': 0}, sentinel=0,
    normalization={'mean': 419.1, 'std': 20.7},
)
qtc_intervalp_sentinel = TensorMap(
    'QTCInterval', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'QTCInterval': 0}, sentinel=0,
    normalization={'mean': 419.1, 'std': 20.7},
    parents=[qt_interval, rr_interval],
)
qtc_intervalp_sentinel = TensorMap(
    'QTCInterval', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'QTCInterval': 0}, sentinel=0,
    normalization={'mean': 419.1, 'std': 20.7},
    parents=[qt_interval, rr_interval],
)
r_axis_sentinel = TensorMap('RAxis', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'RAxis': 0}, sentinel=0, normalization={'mean': 25.7, 'std': 36.6})
rr_interval_sentinel = TensorMap(
    'RRInterval', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'RRInterval': 0}, sentinel=0,
    normalization={'mean': 1040.61, 'std': 175.5},
)
t_axis_sentinel = TensorMap('TAxis', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', channel_map={'TAxis': 0}, sentinel=0, normalization={'mean': 40.8, 'std': 32.6})


bb_baseline = TensorMap(
    'bb_baseline', Interpretation.CATEGORICAL, channel_map={'no_bb_baseline': 0, 'bb_baseline': 1},
    loss=weighted_crossentropy([0.0453, 0.9547], 'bb_baseline'),
)
ccb_baseline = TensorMap(
    'ccb_baseline', Interpretation.CATEGORICAL, channel_map={'no_ccb_baseline': 0, 'ccb_baseline': 1},
    loss=weighted_crossentropy([0.0044, 0.9956], 'ccb_baseline'),
)
class1_baseline = TensorMap(
    'class1_baseline', Interpretation.CATEGORICAL, channel_map={'no_class1_baseline': 0, 'class1_baseline': 1},
    loss=weighted_crossentropy([0.0023, 0.9977], 'class1_baseline'),
)
class3_baseline = TensorMap(
    'class3_baseline', Interpretation.CATEGORICAL, channel_map={'no_class3_baseline': 0, 'class3_baseline': 1},
    loss=weighted_crossentropy([0.0011, 0.9989], 'class3_baseline'),
)
qtc_drug_def_baseline = TensorMap(
    'qtc_drug_def_baseline', Interpretation.CATEGORICAL,
    channel_map={'no_qtc_drug_def_baseline': 0, 'qtc_drug_def_baseline': 1},
    loss=weighted_crossentropy([0.0210, 0.9790], 'qtc_drug_def_baseline'),
)
qtc_drug_poss_baseline = TensorMap(
    'qtc_drug_poss_baseline', Interpretation.CATEGORICAL,
    channel_map={'no_qtc_drug_poss_baseline': 0, 'qtc_drug_poss_baseline': 1},
    loss=weighted_crossentropy([0.0189, 0.9811], 'qtc_drug_poss_baseline'),
)
combined_qtc_drug_baseline = TensorMap(
    'combined_qtc_drug_baseline', Interpretation.CATEGORICAL,
    channel_map={'no_combined_qtc_drug_baseline': 0, 'combined_qtc_drug_baseline': 1},
    loss=weighted_crossentropy([0.0389, 0.9611], 'combined_qtc_drug_baseline'),
)

class1_baseline = TensorMap('class1_baseline', Interpretation.CATEGORICAL, channel_map={'no_class1_baseline': 0, 'class1_baseline': 1}, loss_weight=100.0, loss=weighted_crossentropy([0.0023, 0.9977], 'class1_baseline'))
bb_baseline = TensorMap('bb_baseline', Interpretation.CATEGORICAL, channel_map={'no_bb_baseline': 0, 'bb_baseline': 1}, loss_weight=100.0, loss=weighted_crossentropy([0.0453, 0.9547], 'bb_baseline'))
class3_baseline = TensorMap('class3_baseline', Interpretation.CATEGORICAL, channel_map={'no_class3_baseline': 0, 'class3_baseline': 1}, loss_weight=100.0, loss=weighted_crossentropy([0.0011, 0.9989], 'class3_baseline'))
ccb_baseline = TensorMap('ccb_baseline', Interpretation.CATEGORICAL, channel_map={'no_ccb_baseline': 0, 'ccb_baseline': 1}, loss_weight=100.0, loss=weighted_crossentropy([0.0044, 0.9956], 'ccb_baseline'))
qtc_drug_def_baseline = TensorMap('qtc_drug_def_baseline', Interpretation.CATEGORICAL, channel_map={'no_qtc_drug_def_baseline': 0, 'qtc_drug_def_baseline': 1}, loss_weight=100.0, loss=weighted_crossentropy([0.0210, 0.9790], 'qtc_drug_def_baseline'))
qtc_drug_poss_baseline = TensorMap('qtc_drug_poss_baseline', Interpretation.CATEGORICAL, channel_map={'no_qtc_drug_poss_baseline': 0, 'qtc_drug_poss_baseline': 1}, loss_weight=100.0, loss=weighted_crossentropy([0.0189, 0.9811], 'qtc_drug_poss_baseline'))
class1_fu = TensorMap('class1_fu', Interpretation.CATEGORICAL, channel_map={'no_class1_fu': 0, 'class1_fu': 1}, loss_weight=100.0, loss=weighted_crossentropy([0.0018, 0.9982], 'class1_fu'))
bb_fu = TensorMap('bb_fu', Interpretation.CATEGORICAL, channel_map={'no_bb_fu': 0, 'bb_fu': 1}, loss_weight=100.0, loss=weighted_crossentropy([0.0306, 0.9694], 'bb_fu'))
class3_fu = TensorMap('class3_fu', Interpretation.CATEGORICAL, channel_map={'no_class3_fu': 0, 'class3_fu': 1}, loss_weight=100.0, loss=weighted_crossentropy([0.0006, 0.9994], 'class3_fu'))
ccb_fu = TensorMap('ccb_fu', Interpretation.CATEGORICAL, channel_map={'no_ccb_fu': 0, 'ccb_fu': 1}, loss_weight=100.0, loss=weighted_crossentropy([0.0035, 0.9965], 'ccb_fu'))
qtc_drug_def_fu = TensorMap('qtc_drug_def_fu', Interpretation.CATEGORICAL, channel_map={'no_qtc_drug_def_fu': 0, 'qtc_drug_def_fu': 1}, loss_weight=100.0, loss=weighted_crossentropy([0.0140, 0.9860], 'qtc_drug_def_fu'))
qtc_drug_poss_fu = TensorMap('qtc_drug_poss_fu', Interpretation.CATEGORICAL, channel_map={'no_qtc_drug_poss_fu': 0, 'qtc_drug_poss_fu': 1}, loss_weight=100.0, loss=weighted_crossentropy([0.0127, 0.9873], 'qtc_drug_poss_fu'))
qtc_drug_def_any = TensorMap('qtc_drug_def_any', Interpretation.CATEGORICAL, channel_map={'no_qtc_drug_def_any': 0, 'qtc_drug_def_any': 1}, loss_weight=100.0, loss=weighted_crossentropy([0.0302, 0.9698], 'qtc_drug_def_any'))
qtc_drug_poss_any = TensorMap('qtc_drug_poss_any', Interpretation.CATEGORICAL, channel_map={'no_qtc_drug_poss_any': 0, 'qtc_drug_poss_any': 1}, loss_weight=100.0, loss=weighted_crossentropy([0.0267, 0.9733], 'qtc_drug_poss_any'))
any_class1 = TensorMap('any_class1', Interpretation.CATEGORICAL, channel_map={'no_any_class1': 0, 'any_class1': 1}, loss_weight=100.0, loss=weighted_crossentropy([0.0031, 0.9969], 'any_class1'))
any_bb = TensorMap('any_bb', Interpretation.CATEGORICAL, channel_map={'no_any_bb': 0, 'any_bb': 1}, loss_weight=100.0, loss=weighted_crossentropy([0.0602, 0.9398], 'any_bb'))
any_class3 = TensorMap('any_class3', Interpretation.CATEGORICAL, channel_map={'no_any_class3': 0, 'any_class3': 1}, loss_weight=100.0, loss=weighted_crossentropy([0.0013, 0.9987], 'any_class3'))
any_ccb = TensorMap('any_ccb', Interpretation.CATEGORICAL, channel_map={'no_any_ccb': 0, 'any_ccb': 1}, loss_weight=100.0, loss=weighted_crossentropy([0.0062, 0.9938], 'any_ccb'))
combined_qtc_drug_baseline = TensorMap('combined_qtc_drug_baseline', Interpretation.CATEGORICAL, channel_map={'no_combined_qtc_drug_baseline': 0, 'combined_qtc_drug_baseline': 1}, loss_weight=100.0, loss=weighted_crossentropy([0.0389, 0.9611], 'combined_qtc_drug_baseline'))
combined_qtc_drug_fu = TensorMap('combined_qtc_drug_fu', Interpretation.CATEGORICAL, channel_map={'no_combined_qtc_drug_fu': 0, 'combined_qtc_drug_fu': 1}, loss_weight=100.0, loss=weighted_crossentropy([0.0260, 0.9740], 'combined_qtc_drug_fu'))
combined_qtc_drug_any = TensorMap('combined_qtc_drug_any', Interpretation.CATEGORICAL, channel_map={'no_combined_qtc_drug_any': 0, 'combined_qtc_drug_any': 1}, loss_weight=100.0, loss=weighted_crossentropy([0.0546, 0.9454], 'combined_qtc_drug_any'))

ecg_bike_max_hr_no0 = TensorMap(
    'bike_max_hr', Interpretation.CONTINUOUS, channel_map={'bike_max_hr': 0},
    loss=ignore_zeros_logcosh, metrics=['log_cosh'], normalization={'mean': 110.03, 'std': 20.04},
)
ecg_bike_resting_hr_no0 = TensorMap(
    'bike_resting_hr', Interpretation.CONTINUOUS, channel_map={'bike_resting_hr': 0},
    loss=ignore_zeros_logcosh, metrics=['log_cosh'], normalization={'mean': 71.2, 'std': 12.57},
)
ecg_bike_max_pred_hr_no0 = TensorMap(
    'bike_max_pred_hr', Interpretation.CONTINUOUS, channel_map={'bike_max_pred_hr': 0},
    loss=ignore_zeros_logcosh, metrics=['log_cosh'], normalization={'mean': 167.5, 'std': 5.78},
)
ecg_bike_raw_trend_hr = TensorMap(
    'trend_heartrate', Interpretation.CONTINUOUS, shape=(87,), path_prefix='ukb_ecg_bike',
    tensor_from_file=normalized_first_date,
)
ecg_bike_raw_trend_load = TensorMap(
    'trend_load', Interpretation.CONTINUOUS, shape=(87,), path_prefix='ukb_ecg_bike',
    tensor_from_file=normalized_first_date,
)
ecg_bike_raw_trend_grade = TensorMap(
    'trend_grade', Interpretation.CONTINUOUS, shape=(87,), path_prefix='ukb_ecg_bike',
    tensor_from_file=normalized_first_date,
)
ecg_bike_raw_trend_artifact = TensorMap(
    'trend_artifact', Interpretation.CONTINUOUS, shape=(87,), path_prefix='ukb_ecg_bike',
    tensor_from_file=normalized_first_date,
)
ecg_bike_raw_trend_mets = TensorMap(
    'trend_mets', Interpretation.CONTINUOUS, shape=(87,), path_prefix='ukb_ecg_bike',
    tensor_from_file=normalized_first_date,
)
ecg_bike_raw_trend_pacecount = TensorMap(
    'trend_pacecount', Interpretation.CONTINUOUS, shape=(87,), path_prefix='ukb_ecg_bike',
    tensor_from_file=normalized_first_date,
)
ecg_bike_raw_trend_phasename = TensorMap(
    'trend_phasename', Interpretation.CONTINUOUS, shape=(87,), path_prefix='ukb_ecg_bike',
    tensor_from_file=normalized_first_date,
)
ecg_bike_raw_trend_phasetime = TensorMap(
    'trend_phasetime', Interpretation.CONTINUOUS, shape=(87,), path_prefix='ukb_ecg_bike',
    tensor_from_file=normalized_first_date,
)
ecg_bike_raw_trend_time = TensorMap(
    'trend_time', Interpretation.CONTINUOUS, shape=(87,), path_prefix='ukb_ecg_bike',
    tensor_from_file=normalized_first_date,
)
ecg_bike_raw_trend_vecount = TensorMap(
    'trend_vecount', Interpretation.CONTINUOUS, shape=(87,), path_prefix='ukb_ecg_bike',
    tensor_from_file=normalized_first_date,
)
ecg_bike_raw_full = TensorMap(
    'full', Interpretation.CONTINUOUS, shape=(216500, 3), path_prefix='ukb_ecg_bike',
    tensor_from_file=normalized_first_date,
)
ecg_bike_median = TensorMap(
    'median', Interpretation.CONTINUOUS, shape=(5500, 3), path_prefix='ukb_ecg_bike',
    tensor_from_file=normalized_first_date, normalization=ZeroMeanStd1(), channel_map=ECG_BIKE_LEADS,
)
ecg_bike_strip = TensorMap(
    'strip', Interpretation.CONTINUOUS, shape=(5000, 3), path_prefix='ukb_ecg_bike',
    tensor_from_file=normalized_first_date,
)

def get_instance_0(tm: TensorMap, hd5: h5py.File, dependents=None):
    tensor = np.array(
        hd5[f"{tm.path_prefix}/{tm.name}/instance_0"],
        dtype=np.float32,
    )
    return pad_or_crop_array_to_shape(tm.shape, tensor)


ecg_bike_median_instance_0 = TensorMap(
    'median', Interpretation.CONTINUOUS, shape=(5500, 3), path_prefix='ukb_ecg_bike',
    tensor_from_file=get_instance_0, normalization=ZeroMeanStd1(), channel_map=ECG_BIKE_LEADS,
)

def ppg_from_hd5(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}) -> np.ndarray:
    ppg = np.zeros(tm.shape,  dtype=np.float32)
    ppg[:, 0] = hd5[tm.name]
    return ppg

ppg_0 = TensorMap(
    'ppg_0', shape=(100, 1), tensor_from_file=ppg_from_hd5, channel_map={'ppg_0': 0},
    normalization=Standardize(mean=4824.6, std=3235.8),
)

ppg_2 = TensorMap(
    'ppg_2', shape=(100, 1), tensor_from_file=ppg_from_hd5, channel_map={'ppg_2': 0},
    normalization=Standardize(mean=4824.6, std=3235.8),
)

def uw_ecg_from_hd5(tm, hd5, dependents={}):
    tensor = np.array(hd5[tm.path_prefix], dtype=np.float32)
    return tensor

ecg_median_uw = TensorMap('median',
                          Interpretation.CONTINUOUS,
                          shape=(600, 12),
                          path_prefix='ecg.ecg_rest_median_raw_10',
                          channel_map=ECG_REST_MEDIAN_LEADS,
                          tensor_from_file=uw_ecg_from_hd5)
