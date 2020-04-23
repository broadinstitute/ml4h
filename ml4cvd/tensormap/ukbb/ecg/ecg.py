import h5py
import numpy as np
import scipy
from typing import List, Tuple
from tensorflow.keras.utils import to_categorical
from ml4cvd.tensor_writer_ukbb import tensor_path
from ml4cvd.TensorMap import TensorMap, Interpretation, no_nans, make_range_validator
from ml4cvd.defines import ECG_REST_LEADS, ECG_REST_MEDIAN_LEADS, ECG_REST_AMP_LEADS, ECG_SEGMENTED_CHANNEL_MAP
from ml4cvd.tensormap.ukbb.general import get_tensor_at_first_date, normalized_first_date, pass_nan, build_tensor_from_file
from ml4cvd.metrics import weighted_crossentropy
from ml4cvd.tensormap.ukbb.demographics.demographics import age_in_years_tensor

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
    tensor = np.abs(get_tensor_at_first_date(
        hd5, tm.path_prefix, 'float_array', tm.name))
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
    population_normalize: float = None, random_roll: bool = False, warp: bool = False, downsample_steps: int = 0,
    short_time_nperseg: int = 0, short_time_noverlap: int = 0,
):
    def ecg_rest_from_file(tm, hd5, dependents={}):
        tensor = np.zeros(tm.shape, dtype=np.float32)
        if random_roll:
            roll = np.random.randint(2500)
        if tm.dependent_map is not None:
            dependents[tm.dependent_map] = np.zeros(
                tm.dependent_map.shape, dtype=np.float32)
            key_choices = [k for k in hd5[tm.path_prefix] if tm.name in k]
            lead_idx = np.random.choice(key_choices)
            tensor = np.reshape(
                hd5[tm.path_prefix][lead_idx][: tensor.shape[0] * tensor.shape[1]], tensor.shape, order='F')
            dependents[tm.dependent_map][:, 0] = np.array(
                hd5[tm.path_prefix][lead_idx.replace(tm.name, tm.dependent_map.name)])
            dependents[tm.dependent_map] = tm.zero_mean_std1(
                dependents[tm.dependent_map])
        else:
            for k in hd5[tm.path_prefix]:
                if k in tm.channel_map:
                    data = tm.hd5_first_dataset_in_group(
                        hd5, f'{tm.path_prefix}/{k}/')
                    if short_time_nperseg > 0 and short_time_noverlap > 0:
                        f, t, short_time_ft = scipy.signal.stft(
                            data, nperseg=short_time_nperseg, noverlap=short_time_noverlap)
                        tensor[..., tm.channel_map[k]] = short_time_ft
                    elif downsample_steps > 1:
                        tensor[:, tm.channel_map[k]] = np.array(data, dtype=np.float32)[
                                                                ::downsample_steps]
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
        ecg_interpretation = str(tm.hd5_first_dataset_in_group(
            hd5, 'ukb_ecg_rest/ecg_rest_text/')[()])
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
    ecg_interpretation = str(tm.hd5_first_dataset_in_group(
        hd5, 'ukb_ecg_rest/ecg_rest_text/')[()])
    for channel in tm.channel_map:
        if channel in ecg_interpretation:
            categorical_data[tm.channel_map[channel]] = 1.0
            return categorical_data
    if 'no_' + tm.name in tm.channel_map:
        categorical_data[tm.channel_map['no_' + tm.name]] = 1.0
        return categorical_data
    else:
        raise ValueError(
            f"ECG categorical interpretation could not find any of these keys: {tm.channel_map.keys()}")


# Extract RAmplitude and SAmplitude for LVH criteria
def _make_ukb_ecg_rest(population_normalize: float = None):
    def ukb_ecg_rest_from_file(tm, hd5, dependents={}):
        if 'ukb_ecg_rest' not in hd5:
            raise ValueError(
                'Group with R and S amplitudes not present in hd5')
        tensor = get_tensor_at_first_date(
            hd5, tm.path_prefix, tm.name, pass_nan)
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
                'Group with R and S amplitudes not present in hd5')
        tensor_ramp = get_tensor_at_first_date(
            hd5, tm.path_prefix, 'ramplitude', pass_nan)
        tensor_samp = get_tensor_at_first_date(
            hd5, tm.path_prefix, 'samplitude', pass_nan)
        criteria_sleads = [lead_order[l] for l in ['V1', 'V3']]
        criteria_rleads = [lead_order[l] for l in ['aVL', 'V5', 'V6']]
        if np.any(np.isnan(np.union1d(tensor_ramp[criteria_rleads], tensor_samp[criteria_sleads]))):
            raise ValueError(
                'Missing some of the R and S amplitude readings needed to evaluate LVH criteria')
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
                f'{tm.name} criterion for LVH is not accounted for')
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
            hd5, tm.dependent_map.hd5_key_guess())
        offset_seconds = float(segmented.attrs['offset_seconds'])
        random_offset_samples = 0
        if random_offset_seconds > 0:
            random_offset_begin = np.random.uniform(random_offset_seconds)
            offset_seconds += random_offset_begin
            random_offset_samples = int(random_offset_begin * hertz)
        offset_begin = int(offset_seconds * hertz)
        segment_index = np.array(
            segmented[random_offset_samples:random_offset_samples+tm.dependent_map.shape[0]], dtype=np.float32)
        dependents[tm.dependent_map] = to_categorical(
            segment_index, tm.dependent_map.shape[-1])
        for k in hd5[tm.path_prefix]:
            if k in tm.channel_map:
                tensor[:, tm.channel_map[k]] = np.array(hd5[tm.path_prefix][k], dtype=np.float32)[
                                                        offset_begin:offset_begin+tm.shape[0]]
        if population_normalize is None:
            tm.normalization = {'zero_mean_std1': 1.0}
        else:
            tensor /= population_normalize
        return tensor
    return ecg_rest_section_to_segment


ecg_bike_hrr = TensorMap(
    'hrr', path_prefix='ecg_bike', loss='logcosh', metrics=['mae'], shape=(1,),
    normalization={'mean': 30.55, 'std': 12.81},
    tensor_from_file=_first_date_hrr,
)
ecg_bike_healthy_max_hr = TensorMap(
    'max_hr', path_prefix='ecg_bike', loss='logcosh', metrics=['mae'],
    normalization={'mean': 113.7, 'std': 13.3}, shape=(1,),
    tensor_from_file=_healthy_bike,
)
ecg_bike_healthy_hrr = TensorMap(
    'hrr', path_prefix='ecg_bike', loss='logcosh', metrics=['mae'], shape=(1,),
    normalization={'mean': 30.47, 'std': 11.76},
    tensor_from_file=_healthy_hrr,
)
ecg_bike_healthy_resting = TensorMap(
    'resting_hr', path_prefix='ecg_bike', loss='logcosh', metrics=['mae'], shape=(1,),
    normalization={'mean': 70.0, 'std': 11.62},
    tensor_from_file=_healthy_bike,
)
ecg_bike_med_pretest_hr = TensorMap(
    'trend_heartrate', path_prefix='ecg_bike', loss='logcosh', metrics=['mae'], shape=(1,),
    normalization={'mean': 70., 'std': 11.},
    tensor_from_file=_median_pretest,
)
ecg_bike_med_pretest_stamp = TensorMap(
    'trend_stamplitude', path_prefix='ecg_bike', loss='logcosh', metrics=['mae'], shape=(1,),
    normalization={'mean': .03, 'std': .03},
    tensor_from_file=_median_pretest,
)
ecg_bike_med_pretest_jpoint = TensorMap(
    'trend_jpointamplitude', path_prefix='ecg_bike', loss='logcosh', metrics=['mae'], shape=(1,),
    normalization={'mean': .032, 'std': .46},
    tensor_from_file=_median_pretest,
)
ecg_bike_med_pretest_stamp20 = TensorMap(
    'trend_stamplitude20ms', path_prefix='ecg_bike', loss='logcosh', metrics=['mae'], shape=(1,),
    normalization={'mean': .03, 'std': .03},
    tensor_from_file=_median_pretest,
)
ecg_bike_recovery = TensorMap(
    'full', shape=(30000, 1), path_prefix='ecg_bike', validator=no_nans,
    tensor_from_file=_first_date_bike_recovery,
)
ecg_bike_pretest = TensorMap(
    'full', shape=(500 * 15 - 4, 3), path_prefix='ecg_bike', validator=no_nans,
    normalization={'mean': np.array(
        [7, -7, 3.5])[np.newaxis], 'std': np.array([31, 30, 16])[np.newaxis]},
    tensor_from_file=_first_date_bike_pretest,
)
ecg_bike_pretest_5k = TensorMap(
    'full', shape=(5000, 3), path_prefix='ecg_bike', validator=no_nans,
    normalization={'mean': np.array(
        [7, -7, 3.5])[np.newaxis], 'std': np.array([31, 30, 16])[np.newaxis]},
    tensor_from_file=_first_date_bike_pretest,
)
ecg_bike_new_hrr = TensorMap(
    'hrr', path_prefix='ecg_bike', loss='logcosh', metrics=['mae'], shape=(1,),
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
        'inference.tsv', 'ecg_bike_hrr-sentinel_prediction'),
)
ecg_bike_hr_achieved = TensorMap(
    'hr_achieved', path_prefix='ecg_bike', loss='logcosh', metrics=['mae'], shape=(1,),
    normalization={'mean': .68, 'std': .1},
    tensor_from_file=_hr_achieved,
)

ecg_rest_raw = TensorMap(
    'ecg_rest_raw', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(population_normalize=2000.0),
    channel_map=ECG_REST_LEADS,
)

ecg_rest_raw_roll = TensorMap(
    'ecg_rest_raw', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(population_normalize=2000.0, random_roll=True),
    channel_map=ECG_REST_LEADS, cacheable=False,
)
ecg_rest_raw_warp = TensorMap(
    'ecg_rest_raw', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(population_normalize=2000.0, warp=True),
    channel_map=ECG_REST_LEADS, cacheable=False,
)
ecg_rest_raw_warp_n_roll = TensorMap(
    'ecg_rest_raw', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(population_normalize=2000.0, random_roll=True, warp=True),
    channel_map=ECG_REST_LEADS, cacheable=False,
)
ecg_rest_raw_100 = TensorMap(
    'ecg_rest_raw_100', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(population_normalize=100.0),
    channel_map=ECG_REST_LEADS,
)

ecg_rest = TensorMap(
    'strip', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
    channel_map=ECG_REST_LEADS, normalization={'zero_mean_std1': 1.0},
)
ecg_rest_2500_ukb = TensorMap(
    'ecg_rest_2500', Interpretation.CONTINUOUS, shape=(2500, 12), path_prefix='ukb_ecg_rest', channel_map=ECG_REST_LEADS,
    tensor_from_file=_make_ecg_rest(downsample_steps=2), normalization={'zero_mean_std1': 1.0},
)

ecg_rest_stft = TensorMap(
    'ecg_rest_stft', Interpretation.CONTINUOUS, shape=(33, 158, 12), path_prefix='ukb_ecg_rest', channel_map=ECG_REST_LEADS,
    tensor_from_file=_make_ecg_rest(short_time_nperseg=64, short_time_noverlap=32), normalization={'zero_mean_std1': 1.0},
)
ecg_rest_stft_512 = TensorMap(
    'ecg_rest_stft_512', shape=(257, 314, 12), path_prefix='ukb_ecg_rest', channel_map=ECG_REST_LEADS,
    tensor_from_file=_make_ecg_rest(short_time_nperseg=512, short_time_noverlap=496), normalization={'zero_mean_std1': 1.0},
)

ecg_rest = TensorMap(
    'strip', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
    channel_map=ECG_REST_LEADS, normalization={'zero_mean_std1': 1.0},
)

ecg_rest_stack = TensorMap(
    'strip', Interpretation.CONTINUOUS, shape=(600, 12, 8), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
    channel_map=ECG_REST_LEADS, normalization={'zero_mean_std1': 1.0},
)

ecg_rest_median_raw = TensorMap(
    'median', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(600, 12), loss='logcosh', activation='linear', tensor_from_file=_make_ecg_rest(population_normalize=2000.0),
    metrics=['mse', 'mae', 'logcosh'], channel_map=ECG_REST_MEDIAN_LEADS,
)

ecg_rest_median = TensorMap(
    'median', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(600, 12), loss='logcosh', activation='linear', tensor_from_file=_make_ecg_rest(),
    metrics=['mse', 'mae', 'logcosh'], channel_map=ECG_REST_MEDIAN_LEADS, normalization={'zero_mean_std1': 1.0},
)

ecg_rest_median_stack = TensorMap(
    'median', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(600, 12, 1), activation='linear', tensor_from_file=_make_ecg_rest(),
    metrics=['mse', 'mae', 'logcosh'], loss='logcosh', loss_weight=1.0,
    channel_map=ECG_REST_MEDIAN_LEADS, normalization={'zero_mean_std1': 1.0},
)

ecg_median_1lead = TensorMap(
    'median', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(600, 1), loss='logcosh', loss_weight=10.0, tensor_from_file=_make_ecg_rest(),
    activation='linear', metrics=['mse', 'mae', 'logcosh'], channel_map={'lead': 0}, normalization={'zero_mean_std1': 1.0},
)

ecg_rest_1lead = TensorMap(
    'strip', Interpretation.CONTINUOUS, shape=(600, 8), path_prefix='ukb_ecg_rest', channel_map={'lead': 0}, tensor_from_file=_make_ecg_rest(),
    dependent_map=ecg_median_1lead, normalization={
        'zero_mean_std1': 1.0},
)

ecg_rest_raw = TensorMap('ecg_rest_raw', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(population_normalize=2000.0),
                                 channel_map=ECG_REST_LEADS)

ecg_rest_raw_roll = TensorMap('ecg_rest_raw', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(population_normalize=2000.0, random_roll=True),
                                       channel_map=ECG_REST_LEADS, cacheable=False)
ecg_rest_raw_warp = TensorMap('ecg_rest_raw', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(population_normalize=2000.0, warp=True),
                                       channel_map=ECG_REST_LEADS, cacheable=False)
ecg_rest_raw_warp_n_roll = TensorMap('ecg_rest_raw', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(population_normalize=2000.0, random_roll=True, warp=True),
                                              channel_map=ECG_REST_LEADS, cacheable=False)
ecg_rest_raw_100 = TensorMap('ecg_rest_raw_100', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(population_normalize=100.0),
                                      channel_map=ECG_REST_LEADS)

ecg_rest = TensorMap('strip', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
                              channel_map=ECG_REST_LEADS, normalization={'zero_mean_std1': 1.0})
ecg_rest_2500_ukb = TensorMap('ecg_rest_2500', Interpretation.CONTINUOUS, shape=(2500, 12), path_prefix='ukb_ecg_rest', channel_map=ECG_REST_LEADS,
                                       tensor_from_file=_make_ecg_rest(downsample_steps=2), normalization={'zero_mean_std1': 1.0})

ecg_rest_stft = TensorMap('ecg_rest_stft', Interpretation.CONTINUOUS, shape=(33, 158, 12), path_prefix='ukb_ecg_rest', channel_map=ECG_REST_LEADS,
                                   tensor_from_file=_make_ecg_rest(short_time_nperseg=64, short_time_noverlap=32), normalization={'zero_mean_std1': 1.0})
ecg_rest_stft_512 = TensorMap('ecg_rest_stft_512', shape=(257, 314, 12), path_prefix='ukb_ecg_rest', channel_map=ECG_REST_LEADS,
                                       tensor_from_file=_make_ecg_rest(short_time_nperseg=512, short_time_noverlap=496), normalization={'zero_mean_std1': 1.0})

ecg_rest = TensorMap('strip', Interpretation.CONTINUOUS, shape=(5000, 12), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
                              channel_map=ECG_REST_LEADS, normalization={'zero_mean_std1': 1.0})

ecg_rest_stack = TensorMap('strip', Interpretation.CONTINUOUS, shape=(600, 12, 8), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
                                    channel_map=ECG_REST_LEADS, normalization={'zero_mean_std1': 1.0})

ecg_rest_median_raw = TensorMap('median', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(600, 12), loss='logcosh', activation='linear', tensor_from_file=_make_ecg_rest(population_normalize=2000.0),
                                         metrics=['mse', 'mae', 'logcosh'], channel_map=ECG_REST_MEDIAN_LEADS)

ecg_rest_median = TensorMap('median', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(600, 12), loss='logcosh', activation='linear', tensor_from_file=_make_ecg_rest(),
                                     metrics=['mse', 'mae', 'logcosh'], channel_map=ECG_REST_MEDIAN_LEADS, normalization={'zero_mean_std1': 1.0})

ecg_rest_median_stack = TensorMap('median', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(600, 12, 1), activation='linear', tensor_from_file=_make_ecg_rest(),
                                           metrics=['mse', 'mae', 'logcosh'], loss='logcosh', loss_weight=1.0,
                                           channel_map=ECG_REST_MEDIAN_LEADS, normalization={'zero_mean_std1': 1.0})

ecg_median_1lead = TensorMap('median', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(600, 1), loss='logcosh', loss_weight=10.0, tensor_from_file=_make_ecg_rest(),
                                      activation='linear', metrics=['mse', 'mae', 'logcosh'], channel_map={'lead': 0}, normalization={'zero_mean_std1': 1.0})

ecg_rest_1lead = TensorMap('strip', Interpretation.CONTINUOUS, shape=(600, 8), path_prefix='ukb_ecg_rest', channel_map={'lead': 0}, tensor_from_file=_make_ecg_rest(),
                                    dependent_map=ecg_median_1lead, normalization={'zero_mean_std1': 1.0})


ecg_median_1lead_categorical = TensorMap(
    'median',  Interpretation.CATEGORICAL, shape=(600, 32), activation='softmax', tensor_from_file=_make_ecg_rest(),
    channel_map=_get_lead_cm(32)[0], normalization={'zero_mean_std1': 1.0},
    loss=weighted_crossentropy(
        np.array(_get_lead_cm(32)[1]), 'ecg_median_categorical'),
)

ecg_rest_1lead_categorical = TensorMap(
    'strip', shape=(600, 8), path_prefix='ukb_ecg_rest', tensor_from_file=_make_ecg_rest(),
    normalization={'zero_mean_std1': 1.0},
    channel_map={
        'window0': 0, 'window1': 1, 'window2': 2, 'window3': 3,
        'window4': 4, 'window5': 5, 'window6': 6, 'window7': 7,
    },
    dependent_map=ecg_median_1lead_categorical,
)

ecg_rhythm = TensorMap(
    'ecg_rhythm', Interpretation.CATEGORICAL, tensor_from_file=_make_rhythm_tensor(),
    loss=weighted_crossentropy([1.0, 2.0, 3.0, 3.0, 20.0, 20.0], 'ecg_rhythm'),
    channel_map={'Normal_sinus_rhythm': 0, 'Sinus_bradycardia': 1, 'Marked_sinus_bradycardia': 2,
        'Other_sinus_rhythm': 3, 'Atrial_fibrillation': 4, 'Other_rhythm': 5},
)
ecg_rhythm_poor = TensorMap(
    'ecg_rhythm', Interpretation.CATEGORICAL, tensor_from_file=_make_rhythm_tensor(False),
    loss=weighted_crossentropy(
        [1.0, 2.0, 3.0, 3.0, 20.0, 20.0], 'ecg_rhythm_poor'),
    channel_map={'Normal_sinus_rhythm': 0, 'Sinus_bradycardia': 1, 'Marked_sinus_bradycardia': 2,
        'Other_sinus_rhythm': 3, 'Atrial_fibrillation': 4, 'Other_rhythm': 5},
)

ecg_rest_age = TensorMap(
    'ecg_rest_age', Interpretation.CONTINUOUS, tensor_from_file=age_in_years_tensor('ecg_rest_date'), loss='logcosh',
    channel_map={'ecg_rest_age': 0}, validator=make_range_validator(0, 110), normalization={'mean': 65, 'std': 7.7},
)

acute_mi = TensorMap(
    'acute_mi', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text, channel_map={'no_acute_mi': 0, 'ACUTE MI': 1},
    loss=weighted_crossentropy([0.1, 10.0], 'acute_mi'),
)

anterior_blocks = TensorMap(
    'anterior_blocks', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text,
    channel_map={'no_anterior_blocks': 0, 'Left anterior fascicular block': 1,
        'Left posterior fascicular block': 2},
    loss=weighted_crossentropy([0.1, 10.0, 10.0], 'anterior_blocks'),
)

av_block = TensorMap(
    'av_block', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text, channel_map={'no_av_block': 0, 'st degree AV block': 1},
    loss=weighted_crossentropy([0.1, 10.0], 'av_block'),
)

incomplete_right_bundle_branch_block = TensorMap(
    'incomplete_right_bundle_branch_block', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text,
    channel_map={'no_incomplete_right_bundle_branch_block': 0,
        'Incomplete right bundle branch block': 1},
    loss=weighted_crossentropy(
        [0.1, 10.0], 'incomplete_right_bundle_branch_block'),
)

infarcts = TensorMap(
    'infarcts', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text,
    channel_map={'no_infarcts': 0, 'Anterior infarct': 1, 'Anteroseptal infarct': 2,
        'Inferior infarct': 3, 'Lateral infarct': 4, 'Septal infarct': 5},
    loss=weighted_crossentropy([0.1, 4.0, 6.0, 7.0, 6.0, 4.0], 'infarcts'),
)

left_atrial_enlargement = TensorMap(
    'left_atrial_enlargement', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text,
    channel_map={'no_left_atrial_enlargement': 0,
        'Left atrial enlargement': 1},
    loss=weighted_crossentropy([0.1, 10.0], 'left_atrial_enlargement'),
)

left_ventricular_hypertrophy = TensorMap(
    'left_ventricular_hypertrophy', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text,
    channel_map={'no_left_ventricular_hypertrophy': 0,
        'Left ventricular hypertrophy': 1},
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
    channel_map={'no_premature_atrial_complexes': 0,
        'premature atrial complexes': 1},
    loss=weighted_crossentropy([0.1, 10.0], 'premature_atrial_complexes'),
)

premature_supraventricular_complexes = TensorMap(
    'premature_supraventricular_complexes', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text,
    channel_map={'no_premature_supraventricular_complexes': 0,
        'premature supraventricular complexes': 1},
    loss=weighted_crossentropy(
        [0.1, 10.0], 'premature_supraventricular_complexes'),
)

premature_ventricular_complexes = TensorMap(
    'premature_ventricular_complexes', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text,
    channel_map={'no_premature_ventricular_complexes': 0,
        'premature ventricular complexes': 1},
    loss=weighted_crossentropy([0.1, 10.0], 'premature_ventricular_complexes'),
)

prolonged_qt = TensorMap(
    'prolonged_qt', Interpretation.CATEGORICAL, tensor_from_file=label_from_ecg_interpretation_text, channel_map={'no_prolonged_qt': 0, 'Prolonged QT': 1},
    loss=weighted_crossentropy([0.1, 10.0], 'prolonged_qt'),
)


ecg_rest_ramplitude_raw = TensorMap(
    'ramplitude', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(12,), tensor_from_file=_make_ukb_ecg_rest(1.0),
    loss='logcosh', metrics=['mse', 'mape', 'mae'], loss_weight=1.0,
)

ecg_rest_samplitude_raw = TensorMap(
    'samplitude', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(12,), tensor_from_file=_make_ukb_ecg_rest(1.0),
    loss='logcosh', metrics=['mse', 'mape', 'mae'], loss_weight=1.0,
)

ecg_rest_ramplitude = TensorMap(
    'ramplitude', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(12,), tensor_from_file=_make_ukb_ecg_rest(),
    loss='logcosh', metrics=['mse', 'mape', 'mae'], loss_weight=1.0,
)

ecg_rest_samplitude = TensorMap(
    'samplitude', Interpretation.CONTINUOUS, path_prefix='ukb_ecg_rest', shape=(12,), tensor_from_file=_make_ukb_ecg_rest(),
    loss='logcosh', metrics=['mse', 'mape', 'mae'], loss_weight=1.0,
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