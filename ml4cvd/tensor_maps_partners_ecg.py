import os
import csv
import copy
import h5py
import logging
import datetime
import numpy as np
import pandas as pd
from itertools import product
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Union

from ml4cvd.tensor_maps_by_hand import TMAPS
from ml4cvd.normalizer import Standardize, ZeroMeanStd1
from ml4cvd.defines import ECG_REST_UKB_LEADS, ECG_REST_AMP_LEADS, STOP_CHAR
from ml4cvd.defines import PARTNERS_DATE_FORMAT, PARTNERS_DATETIME_FORMAT, CARDIAC_SURGERY_DATE_FORMAT
from ml4cvd.TensorMap import TensorMap, str2date, Interpretation, make_range_validator, decompress_data, TimeSeriesOrder


YEAR_DAYS = 365.26
INCIDENCE_CSV = '/media/erisone_snf13/c3po_mgh_patient_outcomes_05152020.csv'
CARDIAC_SURGERY_OUTCOMES_CSV = '/data/sts-data/mgh-preop-ecg-outcome-labels.csv'
PARTNERS_PREFIX = 'partners_ecg_rest'


def _hd5_filename_to_mrn_int(filename: str) -> int:
    return int(os.path.basename(filename).split('.')[0])


def _get_ecg_dates(tm, hd5):
    if not hasattr(_get_ecg_dates, 'mrn_lookup'):
        _get_ecg_dates.mrn_lookup = dict()
    mrn = _hd5_filename_to_mrn_int(hd5.filename)
    if mrn in _get_ecg_dates.mrn_lookup:
        return _get_ecg_dates.mrn_lookup[mrn]

    dates = list(hd5[tm.path_prefix])
    if tm.time_series_lookup is not None:
        start, end = tm.time_series_lookup[mrn]
        dates = [date for date in dates if start < date < end]
    if tm.time_series_order == TimeSeriesOrder.NEWEST:
        dates.sort()
    elif tm.time_series_order == TimeSeriesOrder.OLDEST:
        dates.sort(reverse=True)
    elif tm.time_series_order == TimeSeriesOrder.RANDOM:
        np.random.shuffle(dates)
    else:
        raise NotImplementedError(f'Unknown option "{tm.time_series_order}" passed for which tensors to use in multi tensor HD5')
    start_idx = tm.time_series_limit if tm.time_series_limit is not None else 1
    dates = dates[-start_idx:]  # If num_tensors is 0, get all tensors
    dates.sort(reverse=True)
    _get_ecg_dates.mrn_lookup[mrn] = dates
    return dates


def validator_no_empty(tm: TensorMap, tensor: np.ndarray, hd5: h5py.File):
    if any(tensor == ''):
        raise ValueError(f'TensorMap {tm.name} failed empty string check.')


def validator_no_negative(tm: TensorMap, tensor: np.ndarray, hd5: h5py.File):
    if any(tensor < 0):
        raise ValueError(f'TensorMap {tm.name} failed non-negative check')


def validator_not_all_zero(tm: TensorMap, tensor: np.ndarray, hd5: h5py.File):
    if np.count_nonzero(tensor) == 0:
        raise ValueError(f'TensorMap {tm.name} failed all-zero check')


def _is_dynamic_shape(tm: TensorMap, num_ecgs: int) -> Tuple[bool, Tuple[int, ...]]:
    if tm.shape[0] is None:
        return True, (num_ecgs,) + tm.shape[1:]
    return False, tm.shape


def _make_hd5_path(tm, ecg_date, value_key):
    return f'{tm.path_prefix}/{ecg_date}/{value_key}'


def _resample_voltage(voltage, desired_samples):
    if len(voltage) != 5000:
        raise ValueError(f'skipping not 5ks')
    if len(voltage) == desired_samples:
        return voltage
    elif len(voltage) == 2500 and desired_samples == 5000:
        x = np.arange(2500)
        x_interp = np.linspace(0, 2500, 5000)
        return np.interp(x_interp, x, voltage)
    elif len(voltage) == 5000 and desired_samples == 2500:
        return voltage[::2]
    else:
        raise ValueError(f'Voltage length {len(voltage)} is not desired {desired_samples} and re-sampling method is unknown.')


def _resample_voltage_with_rate(voltage, desired_samples, rate, desired_rate):
    if len(voltage) == desired_samples and rate == desired_rate:
        return voltage
    elif desired_samples / len(voltage) == 2 and desired_rate / rate == 2:
        x = np.arange(len(voltage))
        x_interp = np.linspace(0, len(voltage), desired_samples)
        return np.interp(x_interp, x, voltage)
    elif desired_samples / len(voltage) == 0.5 and desired_rate / rate == 0.5:
        return voltage[::2]
    elif desired_samples / len(voltage) == 2 and desired_rate == rate:
        return np.pad(voltage, (0, len(voltage)))
    elif desired_samples / len(voltage) == 0.5 and desired_rate == rate:
        return voltage[:len(voltage)//2]
    else:
        raise ValueError(f'Voltage length {len(voltage)} is not desired {desired_samples} with desired rate {desired_rate} and rate {rate}.')


def make_voltage(exact_length = False):
    def get_voltage_from_file(tm, hd5, dependents={}):
        ecg_dates = _get_ecg_dates(tm, hd5)
        dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
        voltage_length = shape[1] if dynamic else shape[0]
        tensor = np.zeros(shape, dtype=np.float32)
        for i, ecg_date in enumerate(ecg_dates):
            for cm in tm.channel_map:
                try:
                    path = _make_hd5_path(tm, ecg_date, cm)
                    voltage = decompress_data(data_compressed=hd5[path][()], dtype=hd5[path].attrs['dtype'])
                    if exact_length:
                        assert len(voltage) == voltage_length
                    voltage = _resample_voltage(voltage, voltage_length)
                    slices = (i, ..., tm.channel_map[cm]) if dynamic else (..., tm.channel_map[cm])
                    tensor[slices] = voltage
                except (KeyError, AssertionError, ValueError):
                    logging.debug(f'Could not get voltage for lead {cm} with {voltage_length} samples in {hd5.filename}')
        return tensor
    return get_voltage_from_file

# Creates 12 TMaps:
# partners_ecg_2500      partners_ecg_2500_exact      partners_ecg_5000      partners_ecg_5000_exact
# partners_ecg_2500_std  partners_ecg_2500_std_exact  partners_ecg_5000_std  partners_ecg_5000_std_exact
# partners_ecg_2500_raw  partners_ecg_2500_raw_exact  partners_ecg_5000_raw  partners_ecg_5000_raw_exact
#
# default normalizes with ZeroMeanStd1 and resamples
# _std normalizes with Standardize mean = 0, std = 2000
# _raw does not normalize
# _exact does not resample
length_options = [2500, 5000]
exact_options = [True, False]
normalize_options = [ZeroMeanStd1(), Standardize(mean=0, std=2000), None]
for length, exact_length, normalization in product(length_options, exact_options, normalize_options):
    norm = '' if isinstance(normalization, ZeroMeanStd1) else '_std' if isinstance(normalization, Standardize) else '_raw'
    exact = '_exact' if exact_length else ''
    name = f'partners_ecg_{length}{norm}{exact}'
    TMAPS[name] = TensorMap(
        name,
        shape=(None, length, 12),
        path_prefix=PARTNERS_PREFIX,
        tensor_from_file=make_voltage(exact_length),
        normalization=normalization,
        channel_map=ECG_REST_AMP_LEADS,
        time_series_limit=0,
        validator=validator_not_all_zero,
    )


def voltage_stat(tm, hd5, dependents={}):
    ecg_dates = _get_ecg_dates(tm, hd5)
    dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
    tensor = np.zeros(shape, dtype=np.float32)
    for i, ecg_date in enumerate(ecg_dates):
        try:
            slices = lambda stat: (i, tm.channel_map[stat]) if dynamic else (tm.channel_map[stat],)
            path = lambda lead: _make_hd5_path(tm, ecg_date, lead)
            voltages = np.array([decompress_data(data_compressed=hd5[path(lead)][()], dtype='int16') for lead in ECG_REST_AMP_LEADS])
            tensor[slices('mean')] = np.mean(voltages)
            tensor[slices('std')] = np.std(voltages)
            tensor[slices('min')] = np.min(voltages)
            tensor[slices('max')] = np.max(voltages)
            tensor[slices('median')] = np.median(voltages)
        except KeyError:
            logging.warning(f'Could not get voltage stats for ECG at {hd5.filename}')
    return tensor


TMAPS['partners_ecg_voltage_stats'] = TensorMap(
    'partners_ecg_voltage_stats',
    shape=(None, 5),
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=voltage_stat,
    channel_map={'mean': 0, 'std': 1, 'min': 2, 'max': 3, 'median': 4},
    time_series_limit=0,
)


def make_voltage_attr(volt_attr: str = ""):
    def get_voltage_attr_from_file(tm, hd5, dependents={}):
        ecg_dates = _get_ecg_dates(tm, hd5)
        dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
        tensor = np.zeros(shape, dtype=np.float32)
        for i, ecg_date in enumerate(ecg_dates):
            for cm in tm.channel_map:
                try:
                    path = _make_hd5_path(tm, ecg_date, cm)
                    slices = (i, tm.channel_map[cm]) if dynamic else (tm.channel_map[cm],)
                    tensor[slices] = hd5[path].attrs[volt_attr]
                except KeyError:
                    pass
        return tensor
    return get_voltage_attr_from_file


TMAPS["voltage_len"] = TensorMap(
    "voltage_len",
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_voltage_attr(volt_attr="len"),
    shape=(None, 12),
    channel_map=ECG_REST_AMP_LEADS,
    time_series_limit=0,
)


def make_partners_ecg_label(keys: Union[str, List[str]] = "read_md_clean", dict_of_list: Dict = dict(), not_found_key: str = "unspecified"):
    if type(keys) == str:
        keys = [keys]

    def get_partners_ecg_label(tm, hd5, dependents={}):
        ecg_dates = _get_ecg_dates(tm, hd5)
        dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
        label_array = np.zeros(shape, dtype=np.float32)
        for i, ecg_date in enumerate(ecg_dates):
            found = False
            for channel, idx in sorted(tm.channel_map.items(), key=lambda cm: cm[1]):
                if channel not in dict_of_list:
                    continue
                for key in keys:
                    path = _make_hd5_path(tm, ecg_date, key)
                    if path not in hd5:
                        continue
                    read = decompress_data(data_compressed=hd5[path][()], dtype=hd5[path].attrs['dtype'])
                    for string in dict_of_list[channel]:
                        if string not in read:
                            continue
                        slices = (i, idx) if dynamic else (idx,)
                        label_array[slices] = 1
                        found = True
                        break
                    if found:
                        break
                if found:
                    break

            if not found:
                slices = (i, tm.channel_map[not_found_key]) if dynamic else (tm.channel_map[not_found_key],)
                label_array[slices] = 1
        return label_array
    return get_partners_ecg_label


def partners_ecg_datetime(tm, hd5, dependents={}):
    ecg_dates = _get_ecg_dates(tm, hd5)
    dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
    tensor = np.full(shape , '', dtype=f'<U19')
    for i, ecg_date in enumerate(ecg_dates):
        tensor[i] = ecg_date
    return tensor


task = "partners_ecg_datetime"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=partners_ecg_datetime,
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


def make_voltage_len_categorical_tmap(lead, channel_prefix = '_', channel_unknown = 'other'):
    def _tensor_from_file(tm, hd5, dependents = {}):
        ecg_dates = _get_ecg_dates(tm, hd5)
        dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
        tensor = np.zeros(shape, dtype=float)
        for i, ecg_date in enumerate(ecg_dates):
            path = _make_hd5_path(tm, ecg_date, lead)
            try:
                lead_len = hd5[path].attrs['len']
                lead_len = f'{channel_prefix}{lead_len}'
                matched = False
                for cm in tm.channel_map:
                    if lead_len.lower() == cm.lower():
                        slices = (i, tm.channel_map[cm]) if dynamic else (tm.channel_map[cm],)
                        tensor[slices] = 1.0
                        matched = True
                        break
                if not matched:
                    slices = (i, tm.channel_map[channel_unknown]) if dynamic else (tm.channel_map[channel_unknown],)
                    tensor[slices] = 1.0
            except KeyError:
                logging.debug(f'Could not get voltage length for lead {lead} from ECG on {ecg_date} in {hd5.filename}')
        return tensor
    return _tensor_from_file


for lead in ECG_REST_AMP_LEADS:
    tmap_name = f'lead_{lead}_len'
    TMAPS[tmap_name] = TensorMap(
        tmap_name,
        interpretation=Interpretation.CATEGORICAL,
        path_prefix=PARTNERS_PREFIX,
        tensor_from_file=make_voltage_len_categorical_tmap(lead=lead),
        channel_map={'_2500': 0, '_5000': 1, 'other': 2},
        time_series_limit=0,
        validator=validator_not_all_zero,
    )


def make_partners_ecg_tensor(key: str, fill: float = 0, channel_prefix: str = '', channel_unknown: str = 'other'):
    def get_partners_ecg_tensor(tm, hd5, dependents={}):
        ecg_dates = _get_ecg_dates(tm, hd5)
        dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
        if tm.interpretation == Interpretation.LANGUAGE:
            tensor = np.full(shape, '', dtype=object)
        elif tm.interpretation == Interpretation.CONTINUOUS:
            tensor = np.zeros(shape, dtype=np.float32) if fill == 0 else np.full(shape, fill, dtype=np.float32)
        elif tm.interpretation == Interpretation.CATEGORICAL:
            tensor = np.zeros(shape, dtype=float)
        else:
            raise NotImplementedError(f'unsupported interpretation for partners tmaps: {tm.interpretation}')

        for i, ecg_date in enumerate(ecg_dates):
            if i >= shape[0]:
                break
            path = _make_hd5_path(tm, ecg_date, key)
            try:
                data = decompress_data(data_compressed=hd5[path][()], dtype='str')
                if tm.interpretation == Interpretation.CATEGORICAL:
                    matched = False
                    data = f'{channel_prefix}{data}'
                    for cm in tm.channel_map:
                        if data.lower() == cm.lower():
                            slices = (i, tm.channel_map[cm]) if dynamic else (tm.channel_map[cm],)
                            tensor[slices] = 1.0
                            matched = True
                            break
                    if not matched:
                        slices = (i, tm.channel_map[channel_unknown]) if dynamic else (tm.channel_map[channel_unknown],)
                        tensor[slices] = 1.0
                else:
                    tensor[i] = data
            except (KeyError, ValueError):
                logging.debug(f'Could not obtain tensor {tm.name} from ECG on {ecg_date} in {hd5.filename}')

        if tm.interpretation == Interpretation.LANGUAGE:
            tensor = tensor.astype(str)
        return tensor
    return get_partners_ecg_tensor


def make_partners_language_tensor(key: str):
    def language_tensor(tm, hd5, dependents={}):
        words = str(decompress_data(data_compressed=hd5[key][()], dtype=hd5[key].attrs['dtype']))
        tensor = np.zeros(tm.shape, dtype=np.float32)
        for i, c in enumerate(words):
            if i >= tm.shape[0]:
                logging.debug(f'Text {words} is longer than {tm.name} can store in shape:{tm.shape}, truncating...')
                break
            tensor[i, tm.channel_map[c]] = 1.0
        tensor[min(tm.shape[0]-1, i+1), tm.channel_map[STOP_CHAR]] = 1.0
        return tensor
    return language_tensor


task = "partners_ecg_read_md"
TMAPS[task] = TensorMap(
    task,
    #annotation_units=128,
    #channel_map=PARTNERS_CHAR_2_IDX,
    interpretation=Interpretation.LANGUAGE,
    #shape=(512, len(PARTNERS_CHAR_2_IDX)),
    path_prefix=PARTNERS_PREFIX,
    #tensor_from_file=make_partners_language_tensor(key="read_md_clean"),
    tensor_from_file=make_partners_ecg_tensor(key="read_md_clean"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


task = "partners_ecg_read_pc"
TMAPS[task] = TensorMap(
    task,
    #annotation_units=128,
    #channel_map=PARTNERS_CHAR_2_IDX,
    interpretation=Interpretation.LANGUAGE,
    #tensor_from_file=make_partners_language_tensor(key="read_pc_clean"),
    #shape=(512, len(PARTNERS_CHAR_2_IDX)),
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_partners_ecg_tensor(key="read_pc_clean"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


task = "partners_ecg_patientid"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_partners_ecg_tensor(key="patientid"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


def validator_clean_mrn(tm: TensorMap, tensor: np.ndarray, hd5: h5py.File):
    int(tensor)


task = "partners_ecg_patientid_clean"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_partners_ecg_tensor(key="patientid_clean"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_clean_mrn,
)


task = "partners_ecg_firstname"
TMAPS[task] = TensorMap(
    task,
    #channel_map=PARTNERS_CHAR_2_IDX,
    interpretation=Interpretation.LANGUAGE,
    #tensor_from_file=make_partners_language_tensor(key="patientfirstname"),
    #shape=(512, len(PARTNERS_CHAR_2_IDX)),
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_partners_ecg_tensor(key="patientfirstname"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


task = "partners_ecg_lastname"
TMAPS[task] = TensorMap(
    task,
    #channel_map=PARTNERS_CHAR_2_IDX,
    interpretation=Interpretation.LANGUAGE,
    #tensor_from_file=make_partners_language_tensor(key="patientlastname"),
    #shape=(512, len(PARTNERS_CHAR_2_IDX)),
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_partners_ecg_tensor(key="patientlastname"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


task = "partners_ecg_sex"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_partners_ecg_tensor(key="gender"),
    channel_map={'female': 0, 'male': 1},
    shape=(2,),
    validator=validator_not_all_zero,
)

task = "partners_ecg_date"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_partners_ecg_tensor(key="acquisitiondate"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


task = "partners_ecg_time"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_partners_ecg_tensor(key="acquisitiontime"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


task = "partners_ecg_sitename"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_partners_ecg_tensor(key="sitename"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


task = "partners_ecg_location"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_partners_ecg_tensor(key="location"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


task = "partners_ecg_dob"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_partners_ecg_tensor(key="dateofbirth"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


def make_sampling_frequency_from_file(lead: str = "I", duration: int = 10, channel_prefix: str = "_", channel_unknown: str = "other", fill: int = -1):
    def sampling_frequency_from_file(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}):
        ecg_dates = _get_ecg_dates(tm, hd5)
        dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
        if tm.interpretation == Interpretation.CATEGORICAL:
            tensor = np.zeros(shape, dtype=np.float32)
        else:
            tensor = np.full(shape, fill, dtype=np.float32)
        for i, ecg_date in enumerate(ecg_dates):
            path = _make_hd5_path(tm, ecg_date, lead)
            lead_length = hd5[path].attrs["len"]
            sampling_frequency = lead_length / duration
            try:
                if tm.interpretation == Interpretation.CATEGORICAL:
                    matched = False
                    sampling_frequency = f'{channel_prefix}{sampling_frequency}'
                    for cm in tm.channel_map:
                        if sampling_frequency.lower() == cm.lower():
                            slices = (i, tm.channel_map[cm]) if dynamic else (tm.channel_map[cm],)
                            tensor[slices] = 1.0
                            matched = True
                            break
                    if not matched:
                        slices = (i, tm.channel_map[channel_unknown]) if dynamic else (tm.channel_map[channel_unknown],)
                        tensor[slices] = 1.0
                else:
                    tensor[i] = sampling_frequency
            except (KeyError, ValueError):
                logging.debug(f'Could not calculate sampling frequency from ECG on {ecg_date} in {hd5.filename}')
        return tensor
    return sampling_frequency_from_file


task = "partners_ecg_sampling_frequency"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_sampling_frequency_from_file(),
    channel_map={'_250': 0, '_500': 1, 'other': 2},
    time_series_limit=0,
    validator=validator_not_all_zero,
)


task = "partners_ecg_sampling_frequency_pc"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_partners_ecg_tensor(key="ecgsamplebase_pc", channel_prefix='_'),
    channel_map={'_0': 0, '_250': 1, '_500': 2, 'other': 3},
    time_series_limit=0,
    validator=validator_not_all_zero,
)


task = "partners_ecg_sampling_frequency_md"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_partners_ecg_tensor(key="ecgsamplebase_md", channel_prefix='_'),
    channel_map={'_0': 0, '_250': 1, '_500': 2, 'other': 3},
    time_series_limit=0,
    validator=validator_not_all_zero,
)


task = "partners_ecg_sampling_frequency_wv"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_partners_ecg_tensor(key="waveform_samplebase", channel_prefix='_'),
    channel_map={'_0': 0, '_240': 1, '_250': 2, '_500': 3, 'other': 4},
    time_series_limit=0,
    validator=validator_not_all_zero,
)


task = "partners_ecg_sampling_frequency_continuous"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_sampling_frequency_from_file(),
    time_series_limit=0,
    shape=(None, 1),
    validator=validator_no_negative,
)


task = "partners_ecg_sampling_frequency_pc_continuous"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_partners_ecg_tensor(key="ecgsamplebase_pc", fill=-1),
    time_series_limit=0,
    shape=(None, 1),
    validator=validator_no_negative,
)


task = "partners_ecg_sampling_frequency_md_continuous"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_partners_ecg_tensor(key="ecgsamplebase_md", fill=-1),
    time_series_limit=0,
    shape=(None, 1),
    validator=validator_no_negative,
)


task = "partners_ecg_sampling_frequency_wv_continuous"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_partners_ecg_tensor(key="waveform_samplebase", fill=-1),
    time_series_limit=0,
    shape=(None, 1),
    validator=validator_no_negative,
)


task = "partners_ecg_time_resolution"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_partners_ecg_tensor(key="intervalmeasurementtimeresolution", channel_prefix='_'),
    channel_map={'_25': 0, '_50': 1, '_100': 2, 'other': 3},
    time_series_limit=0,
    validator=validator_not_all_zero,
)


task = "partners_ecg_amplitude_resolution"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_partners_ecg_tensor(key="intervalmeasurementamplituderesolution", channel_prefix='_'),
    channel_map={'_10': 0, '_20': 1, '_40': 2, 'other': 3},
    time_series_limit=0,
    validator=validator_not_all_zero,
)


task = "partners_ecg_measurement_filter"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_partners_ecg_tensor(key="intervalmeasurementfilter", channel_prefix='_'),
    time_series_limit=0,
    channel_map={'_None': 0, '_40': 1, '_80': 2, 'other': 3},
    validator=validator_not_all_zero,
)


task = "partners_ecg_high_pass_filter"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_partners_ecg_tensor(key="waveform_highpassfilter", fill=-1),
    time_series_limit=0,
    shape=(None, 1),
    validator=validator_no_negative,
)


task = "partners_ecg_low_pass_filter"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_partners_ecg_tensor(key="waveform_lowpassfilter", fill=-1),
    time_series_limit=0,
    shape=(None, 1),
    validator=validator_no_negative,
)


task = "partners_ecg_ac_filter"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=make_partners_ecg_tensor(key="waveform_acfilter", channel_prefix='_'),
    time_series_limit=0,
    channel_map={'_None': 0, '_50': 1, '_60': 2, 'other': 3},
    validator=validator_not_all_zero,
)


task = "partners_ecg_rate_pc"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    loss='logcosh',
    tensor_from_file=make_partners_ecg_tensor(key="ventricularrate_pc"),
    shape=(None, 1),
    normalization=Standardize(mean=59.3, std=10.6),
    time_series_limit=0,
    validator=make_range_validator(10, 200),
)


task = "partners_ecg_rate_md"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    loss='logcosh',
    tensor_from_file=make_partners_ecg_tensor(key="ventricularrate_md"),
    shape=(None, 1),
    time_series_limit=0,
    validator=make_range_validator(10, 200),
)


task = "partners_ecg_qrs_pc"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    loss='logcosh',
    tensor_from_file=make_partners_ecg_tensor(key="qrsduration_pc"),
    shape=(None, 1),
    time_series_limit=0,
    validator=make_range_validator(20, 400),
)


task = "partners_ecg_qrs_md"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    loss='logcosh',
    tensor_from_file=make_partners_ecg_tensor(key="qrsduration_md"),
    shape=(None, 1),
    time_series_limit=0,
    validator=make_range_validator(20, 400),
)


task = "partners_ecg_pr_pc"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    loss='logcosh',
    tensor_from_file=make_partners_ecg_tensor(key="printerval_pc"),
    shape=(None, 1),
    time_series_limit=0,
    validator=make_range_validator(50, 500),
)


task = "partners_ecg_pr_md"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    loss='logcosh',
    tensor_from_file=make_partners_ecg_tensor(key="printerval_md"),
    shape=(None, 1),
    time_series_limit=0,
    validator=make_range_validator(50, 500),
)


task = "partners_ecg_qt_pc"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    loss='logcosh',
    tensor_from_file=make_partners_ecg_tensor(key="qtinterval_pc"),
    shape=(None, 1),
    time_series_limit=0,
    validator=make_range_validator(100, 800),
)


task = "partners_ecg_qt_md"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    loss='logcosh',
    tensor_from_file=make_partners_ecg_tensor(key="qtinterval_md"),
    shape=(None, 1),
    time_series_limit=0,
    validator=make_range_validator(100, 800),
)


task = "partners_ecg_qtc_pc"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    loss='logcosh',
    tensor_from_file=make_partners_ecg_tensor(key="qtcorrected_pc"),
    shape=(None, 1),
    time_series_limit=0,
    validator=make_range_validator(100, 800),
)


task = "partners_ecg_qtc_md"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    loss='logcosh',
    tensor_from_file=make_partners_ecg_tensor(key="qtcorrected_md"),
    shape=(None, 1),
    time_series_limit=0,
    validator=make_range_validator(100, 800),
)


task = "partners_ecg_paxis_pc"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    loss='logcosh',
    tensor_from_file=make_partners_ecg_tensor(key="paxis_pc", fill=999),
    shape=(None, 1),
    time_series_limit=0,
    validator=make_range_validator(-180, 180),
)


task = "partners_ecg_paxis_md"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    loss='logcosh',
    tensor_from_file=make_partners_ecg_tensor(key="paxis_md", fill=999),
    shape=(None, 1),
    time_series_limit=0,
    validator=make_range_validator(-180, 180),
)


task = "partners_ecg_raxis_pc"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    loss='logcosh',
    tensor_from_file=make_partners_ecg_tensor(key="raxis_pc", fill=999),
    shape=(None, 1),
    time_series_limit=0,
    validator=make_range_validator(-180, 180),
)


task = "partners_ecg_raxis_md"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    loss='logcosh',
    tensor_from_file=make_partners_ecg_tensor(key="raxis_md", fill=999),
    shape=(None, 1),
    time_series_limit=0,
    validator=make_range_validator(-180, 180),
)


task = "partners_ecg_taxis_pc"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    loss='logcosh',
    tensor_from_file=make_partners_ecg_tensor(key="taxis_pc", fill=999),
    shape=(None, 1),
    time_series_limit=0,
    validator=make_range_validator(-180, 180),
)


task = "partners_ecg_taxis_md"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    loss='logcosh',
    tensor_from_file=make_partners_ecg_tensor(key="taxis_md", fill=999),
    shape=(None, 1),
    time_series_limit=0,
    validator=make_range_validator(-180, 180),
)


task = "partners_ecg_weight_lbs"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    loss='logcosh',
    tensor_from_file=make_partners_ecg_tensor(key="weightlbs"),
    shape=(None, 1),
    time_series_limit=0,
    validator=make_range_validator(100, 800),
)


def partners_ecg_age(tm, hd5, dependents={}):
    ecg_dates = _get_ecg_dates(tm, hd5)
    dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
    tensor = np.zeros(shape, dtype=float)
    for i, ecg_date in enumerate(ecg_dates):
        if i >= shape[0]:
            break
        path = lambda key: _make_hd5_path(tm, ecg_date, key)
        try:
            birthday = decompress_data(data_compressed=hd5[path('dateofbirth')][()], dtype='str')
            acquisition = decompress_data(data_compressed=hd5[path('acquisitiondate')][()], dtype='str')
            delta = _partners_str2date(acquisition) - _partners_str2date(birthday)
            years = delta.days / YEAR_DAYS
            tensor[i] = years
        except KeyError:
            try:
                tensor[i] = decompress_data(data_compressed=hd5[path('patientage')][()], dtype='str')
            except KeyError:
                raise KeyError(f'Could not get patient date of birth or age from ECG')
    return tensor


TMAPS['partners_ecg_age'] = TensorMap('partners_ecg_age', path_prefix=PARTNERS_PREFIX, loss='logcosh', tensor_from_file=partners_ecg_age, shape=(None, 1), time_series_limit=0)


def partners_ecg_acquisition_year(tm, hd5, dependents={}):
    ecg_dates = _get_ecg_dates(tm, hd5)
    dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
    tensor = np.zeros(shape, dtype=int)
    for i, ecg_date in enumerate(ecg_dates):
        path = _make_hd5_path(tm, ecg_date, 'acquisitiondate')
        try:
            acquisition = decompress_data(data_compressed=hd5[path][()], dtype='str')
            tensor[i] = _partners_str2date(acquisition).year
        except KeyError:
            pass
    return tensor


TMAPS['partners_ecg_acquisition_year'] = TensorMap('partners_ecg_acquisition_year', path_prefix=PARTNERS_PREFIX, loss='logcosh',  tensor_from_file=partners_ecg_acquisition_year, shape=(None, 1), time_series_limit=0)


def partners_bmi(tm, hd5, dependents={}):
    ecg_dates = _get_ecg_dates(tm, hd5)
    dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
    tensor = np.zeros(shape, dtype=float)
    for i, ecg_date in enumerate(ecg_dates):
        path = lambda key: _make_hd5_path(tm, ecg_date, key)
        try:
            weight_lbs = decompress_data(data_compressed=hd5[path('weightlbs')][()], dtype='str')
            weight_kg = 0.453592 * float(weight_lbs)
            height_in = decompress_data(data_compressed=hd5[path('heightin')][()], dtype='str')
            height_m = 0.0254 * float(height_in)
            if height_m < 0.2 or height_m > 2.1 or weight_kg < 3 or weight_kg > 300:
                raise ValueError('Height or weight from ECG were not reasonable, can not compute BMI')
            bmi = weight_kg / (height_m*height_m)
            logging.debug(f' Height was {height_in} weight: {weight_lbs} bmi is {bmi}')
            tensor[i] = bmi
        except KeyError:
            raise ValueError('Missing Height or weight from ECG can not compute BMI.')
    return tensor


TMAPS['partners_ecg_bmi'] = TensorMap('partners_ecg_bmi', path_prefix=PARTNERS_PREFIX, channel_map={'bmi': 0}, tensor_from_file=partners_bmi, time_series_limit=0)


def partners_channel_string(hd5_key, race_synonyms={}, unspecified_key=None):
    def tensor_from_string(tm, hd5, dependents={}):
        ecg_dates = _get_ecg_dates(tm, hd5)
        dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
        tensor = np.zeros(shape, dtype=np.float32)
        for i, ecg_date in enumerate(ecg_dates):
            path = _make_hd5_path(tm, ecg_date, hd5_key)
            found = False
            try:
                hd5_string = decompress_data(data_compressed=hd5[path][()], dtype='str')
                for key in tm.channel_map:
                    slices = (i, tm.channel_map[key]) if dynamic else (tm.channel_map[key],)
                    if hd5_string.lower() == key.lower():
                        tensor[slices] = 1.0
                        found = True
                        break
                    if key in race_synonyms:
                        for synonym in race_synonyms[key]:
                            if hd5_string.lower() == synonym.lower():
                                tensor[slices] = 1.0
                                found = True
                            if found: break
                        if found: break
            except KeyError:
                pass
            if not found:
                if unspecified_key is None:
                    # TODO Do we want to try to continue to get tensors for other ECGs in HD5?
                    raise ValueError(f'No channel keys found in {hd5_key} for {tm.name} with channel map {tm.channel_map}.')
                slices = (i, tm.channel_map[unspecified_key]) if dynamic else (tm.channel_map[unspecified_key],)
                tensor[slices] = 1.0
        return tensor
    return tensor_from_string


race_synonyms = {'asian': ['oriental'], 'hispanic': ['latino'], 'white': ['caucasian']}
TMAPS['partners_ecg_race'] = TensorMap(
    'partners_ecg_race', interpretation=Interpretation.CATEGORICAL, path_prefix=PARTNERS_PREFIX, channel_map={'asian': 0, 'black': 1, 'hispanic': 2, 'white': 3, 'unknown': 4},
    tensor_from_file=partners_channel_string('race', race_synonyms), time_series_limit=0,
)


def _partners_adult(hd5_key, minimum_age=18):
    def tensor_from_string(tm, hd5, dependents={}):
        ecg_dates = _get_ecg_dates(tm, hd5)
        dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
        tensor = np.zeros(shape, dtype=np.float32)
        for i, ecg_date in enumerate(ecg_dates):
            path = lambda key: _make_hd5_path(tm, ecg_date, key)
            birthday = decompress_data(data_compressed=hd5[path('dateofbirth')][()], dtype='str')
            acquisition = decompress_data(data_compressed=hd5[path('acquisitiondate')][()], dtype='str')
            delta = _partners_str2date(acquisition) - _partners_str2date(birthday)
            years = delta.days / YEAR_DAYS
            if years < minimum_age:
                raise ValueError(f'ECG taken on patient below age cutoff.')
            hd5_string = decompress_data(data_compressed=hd5[path(hd5_key)][()], dtype=hd5[path(hd5_key)].attrs['dtype'])
            found = False
            for key in tm.channel_map:
                if hd5_string.lower() == key.lower():
                    slices = (i, tm.channel_map[key]) if dynamic else (tm.channel_map[key],)
                    tensor[slices] = 1.0
                    found = True
                    break
            if not found:
                # TODO Do we want to try to continue to get tensors for other ECGs in HD5?
                raise ValueError(f'No channel keys found in {hd5_string} for {tm.name} with channel map {tm.channel_map}.')
        return tensor
    return tensor_from_string


TMAPS['partners_adult_sex'] = TensorMap(
    'adult_sex', interpretation=Interpretation.CATEGORICAL, path_prefix=PARTNERS_PREFIX, channel_map={'female': 0, 'male': 1},
    tensor_from_file=_partners_adult('gender'), time_series_limit=0,
)


def voltage_zeros(tm, hd5, dependents={}):
    ecg_dates = _get_ecg_dates(tm, hd5)
    dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
    tensor = np.zeros(shape, dtype=np.float32)
    for i, ecg_date in enumerate(ecg_dates):
        for cm in tm.channel_map:
            path = _make_hd5_path(tm, ecg_date, cm)
            voltage = decompress_data(data_compressed=hd5[path][()], dtype=hd5[path].attrs['dtype'])
            slices = (i, tm.channel_map[cm]) if dynamic else (tm.channel_map[cm],)
            tensor[slices] = np.count_nonzero(voltage == 0)
    return tensor


TMAPS["voltage_zeros"] = TensorMap(
    "voltage_zeros",
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=PARTNERS_PREFIX,
    tensor_from_file=voltage_zeros,
    shape=(None, 12),
    channel_map=ECG_REST_AMP_LEADS,
    time_series_limit=0,
)


def v6_zeros_validator(tm: TensorMap, tensor: np.ndarray, hd5: h5py.File):
    voltage = decompress_data(data_compressed=hd5['V6'][()], dtype=hd5['V6'].attrs['dtype'])
    if np.count_nonzero(voltage == 0) > 10:
        raise ValueError(f'TensorMap {tm.name} has too many zeros in V6.')


def build_partners_time_series_tensor_maps(
        needed_tensor_maps: List[str],
        time_series_limit: int = 1,
) -> Dict[str, TensorMap]:
    name2tensormap: Dict[str:TensorMap] = {}

    for needed_name in needed_tensor_maps:
        if needed_name.endswith('_newest'):
            base_split = '_newest'
            time_series_order = TimeSeriesOrder.NEWEST
        elif needed_name.endswith('_oldest'):
            base_split = '_oldest'
            time_series_order = TimeSeriesOrder.OLDEST
        elif needed_name.endswith('_random'):
            base_split = '_random'
            time_series_order = TimeSeriesOrder.RANDOM
        else:
            continue

        base_name = needed_name.split(base_split)[0]
        if base_name not in TMAPS:
            continue

        time_tmap = copy.deepcopy(TMAPS[base_name])
        time_tmap.name = needed_name
        time_tmap.shape = time_tmap.shape[1:]
        time_tmap.time_series_limit = time_series_limit
        time_tmap.time_series_order = time_series_order
        time_tmap.metrics = None
        time_tmap.infer_metrics()

        name2tensormap[needed_name] = time_tmap
    return name2tensormap


# Date formatting
def _partners_str2date(d) -> datetime.date:
    return datetime.datetime.strptime(d, PARTNERS_DATE_FORMAT).date()


def _loyalty_str2date(date_string: str) -> datetime.date:
    return str2date(date_string.split(' ')[0])


def _cardiac_surgery_str2date(input_date: str, date_format: str = CARDIAC_SURGERY_DATE_FORMAT) -> datetime.datetime:
    return datetime.datetime.strptime(input_date, date_format)


def _ecg_tensor_from_date(tm: TensorMap, hd5: h5py.File, ecg_date: str, population_normalize: int = None):
    tensor = np.zeros(tm.shape, dtype=np.float32)
    for cm in tm.channel_map:
        path = _make_hd5_path(tm, ecg_date, cm)
        voltage = decompress_data(data_compressed=hd5[path][()], dtype=hd5[path].attrs['dtype'])
        voltage = _resample_voltage(voltage, tm.shape[0])
        tensor[..., tm.channel_map[cm]] = voltage
    if population_normalize is not None:
        tensor /= population_normalize
    return tensor


def _date_from_dates(ecg_dates, target_date=None, earliest_date=None):
    if target_date:
        if target_date and earliest_date:
            incident_dates = [d for d in ecg_dates if earliest_date < datetime.datetime.strptime(d, PARTNERS_DATETIME_FORMAT).date() < target_date]
        else:
            incident_dates = [d for d in ecg_dates if datetime.datetime.strptime(d, PARTNERS_DATETIME_FORMAT).date() < target_date]
        if len(incident_dates) == 0:
            raise ValueError('No ECGs prior to target were found.')
        return np.random.choice(incident_dates)
    return np.random.choice(ecg_dates)


def _field_to_index_from_map(value: str, field_map: Dict[str, float] = {'Female': 0, 'Male': 1}) -> float:
    return field_map[value]


def _date_field_to_age(value: str, follow_up_start: datetime.datetime) -> float:
    return (_loyalty_str2date(follow_up_start) - _loyalty_str2date(value)).days / YEAR_DAYS


def csv_field_tensor_from_file(
    file_name: str, patient_column: str = 'MGH_MRN', value_column: str = 'birth_date', value_transform: Callable = _date_field_to_age,
    follow_up_start_column: str = None, delimiter: str = ','
) -> Callable:
    """Build a tensor_from_file function for future (and prior) diagnoses given a TSV of patients and diagnosis dates.

    The tensor_from_file function returned here should be used
    with CATEGORICAL TensorMaps to classify patients by disease state.

    :param file_name: CSV or TSV file with header of patient IDs (MRNs) dates of enrollment and dates of diagnosis
    :param patient_column: The header name of the column of patient ids
    :param value_column: The header name of the column of values to load
    :param value_transform: A function that transforms the string found in the CSV to the desired value
    :param follow_up_start_column: The header name of the column of enrollment dates
    :param delimiter: The delimiter separating columns of the TSV or CSV
    :return: The tensor_from_file function to provide to TensorMap constructors
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader)
        if follow_up_start_column is not None:
            follow_up_start_index = header.index(follow_up_start_column)
        patient_index = header.index(patient_column)
        value_index = header.index(value_column)
        value_table = {}
        for row in reader:
            try:
                patient_key = int(row[patient_index])
                if follow_up_start_column is not None:
                    value_table[patient_key] = value_transform(row[value_index], row[follow_up_start_index])
                else:
                    value_table[patient_key] = value_transform(row[value_index])
            except ValueError as e:
                logging.warning(f'val err {e}')
        logging.info(f'Done processing {value_column} Got {len(value_table)} patient rows. Last value was: {value_table[patient_key]}')

    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        mrn_int = _hd5_filename_to_mrn_int(hd5.filename)
        if mrn_int not in value_table:
            raise KeyError(f'{tm.name} mrn not in csv')

        tensor = np.zeros(tm.shape, dtype=np.float32)
        if tm.is_categorical():
            tensor[value_table[mrn_int]] = 1.0
        elif tm.is_continuous():
            tensor[0] = value_table[mrn_int]
        else:
            raise ValueError(f'{tm.name} has no way to make tensor from csv value.')
        return tensor
    return tensor_from_file


def csv_time_to_event(
    file_name: str, incidence_only: bool = False, patient_column: str = 'Mrn',
    follow_up_start_column: str = 'start_fu', follow_up_total_column: str = 'total_fu',
    diagnosis_column: str = 'first_stroke', delimiter: str = ',',
):
    """Build a tensor_from_file function for modeling relative time to event of diagnoses given a TSV of patients and dates.

    The tensor_from_file function returned here should be used
    with TIME_TO_EVENT TensorMaps to model relative time free from a diagnosis for a given disease.

    :param file_name: CSV or TSV file with header of patient IDs (MRNs) dates of enrollment and dates of diagnosis
    :param incidence_only: Flag to skip patients whose diagnosis date is prior to acquisition date of input data
    :param patient_column: The header name of the column of patient ids
    :param follow_up_start_column: The header name of the column of enrollment dates
    :param follow_up_total_column: The header name of the column with total enrollment time (in years)
    :param diagnosis_column: The header name of the column of disease diagnosis dates
    :param delimiter: The delimiter separating columns of the TSV or CSV
    :return: The tensor_from_file function to provide to TensorMap constructors
    """
    disease_dicts = defaultdict(dict)
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader)
        follow_up_start_index = header.index(follow_up_start_column)
        follow_up_total_index = header.index(follow_up_total_column)
        patient_index = header.index(patient_column)
        date_index = header.index(diagnosis_column)
        for row in reader:
            try:
                patient_key = int(row[patient_index])
                disease_dicts['follow_up_start'][patient_key] = _loyalty_str2date(row[follow_up_start_index])
                disease_dicts['follow_up_total'][patient_key] = float(row[follow_up_total_index])
                if row[date_index] == '' or row[date_index] == 'NULL':
                    continue
                disease_dicts['diagnosis_dates'][patient_key] = _loyalty_str2date(row[date_index])
                if len(disease_dicts['follow_up_start']) % 2000 == 0:
                    logging.debug(f"Processed: {len(disease_dicts['follow_up_start'])} patient rows.")
            except ValueError as e:
                logging.warning(f'val err {e}')
        logging.info(f"Done processing {diagnosis_column} Got {len(disease_dicts['follow_up_start'])} patient rows and {len(disease_dicts['diagnosis_dates'])} events.")

    def _cox_tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        mrn_int = _hd5_filename_to_mrn_int(hd5.filename)
        if mrn_int not in disease_dicts['follow_up_start']:
            raise KeyError(f'{diagnosis_column} did not contain MRN for TensorMap:{tm.name}')

        follow_up_days = YEAR_DAYS * disease_dicts['follow_up_total'][mrn_int]
        if follow_up_days > tm.days_window: # Censor outside window
            has_disease = 0
            follow_up = tm.days_window + 1
        elif mrn_int not in disease_dicts['diagnosis_dates']:
            has_disease = 0
            follow_up = follow_up_days
        else:
            has_disease = 1
            follow_up = (disease_dicts['diagnosis_dates'][mrn_int] - disease_dicts['follow_up_start'][mrn_int]).days

        if incidence_only and has_disease and disease_dicts['diagnosis_dates'][mrn_int] <= disease_dicts['follow_up_start'][mrn_int]:
            raise ValueError(f'{tm.name} only considers incident diagnoses')

        tensor = np.zeros(tm.shape, dtype=np.float32)
        tensor[0] = has_disease
        tensor[1] = follow_up
        return tensor
    return _cox_tensor_from_file


def build_legacy_ecg(
    file_name: str, patient_column: str = 'MGH_MRN', birth_column: str = 'dob', start_column: str = 'start_fu_age',
    delimiter: str = ',', check_birthday: bool = True, population_normalize: int = 2000,
) -> Callable:
    """Build a tensor_from_file function for ECGs in the legacy cohort.

    :param file_name: CSV or TSV file with header of patient IDs (MRNs) dates of enrollment and dates of diagnosis
    :param patient_column: The header name of the column of patient ids
    :param birth_column: The header name of the column of dates of birth
    :param start_column: The header name of the column of enrollment dates
    :param delimiter: The delimiter separating columns of the TSV or CSV
    :return: The tensor_from_file function to provide to TensorMap constructors
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader)
        patient_index = header.index(patient_column)
        birth_index = header.index(birth_column)
        start_index = header.index(start_column)
        birth_table = {}
        patient_table = {}
        earliest_table = {}
        for row in reader:
            try:
                patient_key = int(row[patient_index])
                birth_table[patient_key] = _loyalty_str2date(row[birth_index])
                patient_table[patient_key] = birth_table[patient_key] + datetime.timedelta(days=float(row[start_index])*YEAR_DAYS)
                earliest_table[patient_key] = patient_table[patient_key] - datetime.timedelta(days=3*YEAR_DAYS)
            except ValueError as e:
                logging.debug(f'val err {e}')
        logging.info(f'Done processing. Got {len(patient_table)} patient rows.')

    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        mrn_int = _hd5_filename_to_mrn_int(hd5.filename)
        if mrn_int not in patient_table:
            raise KeyError(f'{tm.name} mrn not in legacy csv.')

        ecg_dates = list(hd5[tm.path_prefix])
        ecg_date_key = _date_from_dates(ecg_dates)#, patient_table[patient_key], earliest_table[patient_key])

        if check_birthday:
            path = _make_hd5_path(tm, ecg_date_key, 'dateofbirth')
            birth_date = _partners_str2date(decompress_data(data_compressed=hd5[path][()], dtype=hd5[path].attrs['dtype']))
            if birth_date != birth_table[mrn_int]:
                raise ValueError(f'Birth dates do not match CSV had {birth_table[patient_key]}!')  # CSV had {birth_table[patient_key]} but HD5 has {birth_date}')

        return _ecg_tensor_from_date(tm, hd5, ecg_date_key, population_normalize)
    return tensor_from_file


def build_ukb_ecg2(
    file_name: str, patient_column: str = 'MGH_MRN', birth_column: str = 'dob', start_column: str = 'start_fu_age',
    delimiter: str = ',', check_birthday: bool = True, population_normalize: int = 2000,
) -> Callable:
    """Build a tensor_from_file function for ECGs in the legacy cohort.

    :param file_name: CSV or TSV file with header of patient IDs (MRNs) dates of enrollment and dates of diagnosis
    :param patient_column: The header name of the column of patient ids
    :param birth_column: The header name of the column of dates of birth
    :param start_column: The header name of the column of enrollment dates
    :param delimiter: The delimiter separating columns of the TSV or CSV
    :return: The tensor_from_file function to provide to TensorMap constructors
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader)
        patient_ecg_date = header.index(patient_column)
        birth_table = {}
        patient_table = {}
        earliest_table = {}
        for row in reader:
            try:
                patient_key = int(row[header.index(patient_column)])
                patient_table[patient_key] = birth_table[patient_key] + datetime.timedelta(days=float(row[start_index])*YEAR_DAYS)
                earliest_table[patient_key] = patient_table[patient_key] - datetime.timedelta(days=3*YEAR_DAYS)
            except ValueError as e:
                logging.debug(f'val err {e}')
        logging.info(f'Done processing. Got {len(patient_table)} patient rows.')

    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        mrn_int = _hd5_filename_to_mrn_int(hd5.filename)
        if mrn_int not in patient_table:
            raise KeyError(f'{tm.name} mrn not in legacy csv.')

        ecg_dates = list(hd5[tm.path_prefix])
        ecg_date_key = _date_from_dates(ecg_dates)
        return _ecg_tensor_from_date(tm, hd5, ecg_date_key, population_normalize)
    return tensor_from_file


def build_incidence_tensor_from_file(
    file_name: str, patient_column: str = 'Mrn', birth_column: str = 'birth_date',
    diagnosis_column: str = 'first_stroke', start_column: str = 'start_fu',
    delimiter: str = ',', incidence_only: bool = False, check_birthday: bool = True, population_normalize: int = None, dependent: bool = False,
) -> Callable:
    """Build a tensor_from_file function for future (and prior) diagnoses given a TSV of patients and diagnosis dates.

    The tensor_from_file function returned here should be used
    with CATEGORICAL TensorMaps to classify patients by disease state.

    :param file_name: CSV or TSV file with header of patient IDs (MRNs) dates of enrollment and dates of diagnosis
    :param patient_column: The header name of the column of patient ids
    :param diagnosis_column: The header name of the column of disease diagnosis dates
    :param start_column: The header name of the column of enrollment dates
    :param delimiter: The delimiter separating columns of the TSV or CSV
    :param incidence_only: Flag to skip patients whose diagnosis date is prior to acquisition date of input data
    :return: The tensor_from_file function to provide to TensorMap constructors
    """
    error = None
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader)
            patient_index = header.index(patient_column)
            birth_index = header.index(birth_column)
            start_index = header.index(start_column)
            date_index = header.index(diagnosis_column)
            date_table = {}
            birth_table = {}
            patient_table = {}
            for row in reader:
                try:
                    patient_key = int(row[patient_index])
                    patient_table[patient_key] = _loyalty_str2date(row[start_index])
                    birth_table[patient_key] = _loyalty_str2date(row[birth_index])
                    if row[date_index] == '' or row[date_index] == 'NULL':
                        continue
                    date_table[patient_key] = _loyalty_str2date(row[date_index])
                    if len(patient_table) % 2000 == 0:
                        logging.debug(f'Processed: {len(patient_table)} patient rows.')
                except ValueError as e:
                    logging.warning(f'val err {e}')
            logging.info(f'Done processing {diagnosis_column} Got {len(patient_table)} patient rows and {len(date_table)} events.')
    except FileNotFoundError as e:
        error = e

    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        if error:
            raise error

        mrn_int = _hd5_filename_to_mrn_int(hd5.filename)
        if mrn_int not in patient_table:
            raise KeyError(f'{tm.name} mrn not in incidence csv')

        disease_date = date_table[mrn_int] if mrn_int in date_table else None
        ecg_dates = list(hd5[tm.path_prefix])
        if dependent:
            ecg_date_key = _date_from_dates(ecg_dates, disease_date if incidence_only else None)
        else:
            ecg_dates.sort()
            ecg_date_key = ecg_dates[-1]

        if check_birthday:
            path = _make_hd5_path(tm, ecg_date_key, 'dateofbirth')
            birth_date = _partners_str2date(decompress_data(data_compressed=hd5[path][()], dtype=hd5[path].attrs['dtype']))
            if birth_date != birth_table[mrn_int]:
                raise ValueError(f'Birth dates do not match CSV had {birth_table[patient_key]}!')  # CSV had {birth_table[patient_key]} but HD5 has {birth_date}')

        ecg_date = datetime.datetime.strptime(ecg_date_key, PARTNERS_DATETIME_FORMAT).date()
        if ecg_date > patient_table[mrn_int]:
            raise ValueError(f'Assessed after enrollment.')

        if mrn_int not in date_table:
            index = 0
        else:
            if incidence_only and disease_date > ecg_date:
                index = 1
            elif incidence_only:
                raise ValueError(f'{tm.name} is skipping prevalent cases.')
            else:
                index = 1 if disease_date < ecg_date else 2

        if dependent:
            tensor = _ecg_tensor_from_date(tm, hd5, ecg_date_key, population_normalize)
            for dtm in tm.dependent_map:
                dependents[tm.dependent_map[dtm]] = np.zeros(tm.dependent_map[dtm].shape, dtype=np.float32)
                dependents[tm.dependent_map[dtm]][index] = 1.0
            logging.debug(f'mrn: {mrn_int}  Got disease_date: {disease_date} assess  {ecg_date_key} index {index}. dtm: {dependents[tm.dependent_map[dtm]]}')
        else:
            tensor = np.zeros(tm.shape, dtype=np.float32)
            tensor[index] = 1.0

        return tensor
    return tensor_from_file


def _diagnosis_channels(disease: str, incidence_only: bool = False):
    if incidence_only:
        return {f'no_{disease}': 0,  f'future_{disease}': 1}
    return {f'no_{disease}': 0, f'prior_{disease}': 1, f'future_{disease}': 2}


def _outcome_channels(outcome: str):
    return {f'no_{outcome}': 0,  f'{outcome}': 1}


def loyalty_time_to_event(
    file_name: str, incidence_only: bool = False, patient_column: str = 'Mrn',
    follow_up_start_column: str = 'start_fu', follow_up_total_column: str = 'total_fu',
    diagnosis_column: str = 'first_stroke', delimiter: str = ',', population_normalize: int = None, dependent: bool = True,
):
    """Build a tensor_from_file function for modeling relative time to event of diagnoses given a TSV of patients and dates.

    The tensor_from_file function returned here should be used
    with TIME_TO_EVENT TensorMaps to model relative time free from a diagnosis for a given disease.

    :param file_name: CSV or TSV file with header of patient IDs (MRNs) dates of enrollment and dates of diagnosis
    :param incidence_only: Flag to skip patients whose diagnosis date is prior to acquisition date of input data
    :param patient_column: The header name of the column of patient ids
    :param follow_up_start_column: The header name of the column of enrollment dates
    :param follow_up_total_column: The header name of the column with total enrollment time (in years)
    :param diagnosis_column: The header name of the column of disease diagnosis dates
    :param delimiter: The delimiter separating columns of the TSV or CSV
    :return: The tensor_from_file function to provide to TensorMap constructors
    """
    error = None
    disease_dicts = defaultdict(dict)
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader)
            follow_up_start_index = header.index(follow_up_start_column)
            follow_up_total_index = header.index(follow_up_total_column)
            patient_index = header.index(patient_column)
            date_index = header.index(diagnosis_column)
            for row in reader:
                try:
                    patient_key = int(row[patient_index])
                    disease_dicts['follow_up_start'][patient_key] = _loyalty_str2date(row[follow_up_start_index])
                    disease_dicts['follow_up_total'][patient_key] = float(row[follow_up_total_index])
                    if row[date_index] == '' or row[date_index] == 'NULL':
                        continue
                    disease_dicts['diagnosis_dates'][patient_key] = _loyalty_str2date(row[date_index])
                    if len(disease_dicts['follow_up_start']) % 2000 == 0:
                        logging.debug(f"Processed: {len(disease_dicts['follow_up_start'])} patient rows.")
                except ValueError as e:
                    logging.warning(f'val err {e}')
            logging.info(f"Done processing {diagnosis_column} Got {len(disease_dicts['follow_up_start'])} patient rows and {len(disease_dicts['diagnosis_dates'])} events.")
    except FileNotFoundError as e:
        error = e

    def _cox_tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        if error:
            raise error
        mrn_int = _hd5_filename_to_mrn_int(hd5.filename)
        if mrn_int not in disease_dicts['follow_up_start']:
            raise KeyError(f'{diagnosis_column} did not contain MRN for TensorMap:{tm.name}')
        ecg_dates = list(hd5[tm.path_prefix])
        disease_date = disease_dicts['diagnosis_dates'][mrn_int] if mrn_int in disease_dicts['diagnosis_dates'] else None
        if dependent:
            ecg_date_key = _date_from_dates(ecg_dates, disease_date if incidence_only else None)
        else:
            ecg_dates.sort()
            ecg_date_key = ecg_dates[-1]
        ecg_date = datetime.datetime.strptime(ecg_date_key, PARTNERS_DATETIME_FORMAT).date()

        if ecg_date > disease_dicts['follow_up_start'][mrn_int]:
            raise ValueError(f'Assessed after enrollment.')
        if (disease_dicts['follow_up_start'][mrn_int] - ecg_date).days > 365*3:
            raise ValueError(f'Assessed 3 years or more before enrollment.')

        if mrn_int not in disease_dicts['diagnosis_dates']:
            has_disease = 0
            censor_date = disease_dicts['follow_up_start'][mrn_int]
            censor_date += datetime.timedelta(days=YEAR_DAYS * disease_dicts['follow_up_total'][mrn_int])
        else:
            has_disease = 1
            censor_date = disease_dicts['diagnosis_dates'][mrn_int]

        if incidence_only and censor_date <= ecg_date and has_disease:
            raise ValueError(f'{tm.name} only considers incident diagnoses')

        if dependent:
            tensor = _ecg_tensor_from_date(tm, hd5, ecg_date_key, population_normalize)
            for dtm in tm.dependent_map:
                dependents[tm.dependent_map[dtm]] = np.zeros(tm.dependent_map[dtm].shape, dtype=np.float32)
                dependents[tm.dependent_map[dtm]][0] = has_disease
                dependents[tm.dependent_map[dtm]][1] = (censor_date - ecg_date).days
        else:
            tensor = np.zeros(tm.shape, dtype=np.float32)
            tensor[0] = has_disease
            tensor[1] = (censor_date - ecg_date).days
        return tensor
    return _cox_tensor_from_file


def _survival_from_file(
    day_window: int, file_name: str, incidence_only: bool = False, patient_column: str = 'Mrn',
    follow_up_start_column: str = 'start_fu', follow_up_total_column: str = 'total_fu',
    diagnosis_column: str = 'first_stroke', delimiter: str = ',', population_normalize: int = None,
) -> Callable:
    """Build a tensor_from_file function for modeling survival curves of diagnoses given a TSV of patients and dates.

    The tensor_from_file function returned here should be used
    with SURVIVAL_CURVE TensorMaps to model survival curves of patients for a given disease.

    :param day_window: Total number of days of follow up the length of the survival curves to learn.
    :param file_name: CSV or TSV file with header of patient IDs (MRNs) dates of enrollment and dates of diagnosis
    :param incidence_only: Flag to skip patients whose diagnosis date is prior to acquisition date of input data
    :param patient_column: The header name of the column of patient ids
    :param follow_up_start_column: The header name of the column of enrollment dates
    :param follow_up_total_column: The header name of the column with total enrollment time (in years)
    :param diagnosis_column: The header name of the column of disease diagnosis dates
    :param delimiter: The delimiter separating columns of the TSV or CSV
    :return: The tensor_from_file function to provide to TensorMap constructors
    """
    error = None
    disease_dicts = defaultdict(dict)
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader)
            follow_up_start_index = header.index(follow_up_start_column)
            follow_up_total_index = header.index(follow_up_total_column)
            patient_index = header.index(patient_column)
            date_index = header.index(diagnosis_column)
            for row in reader:
                try:
                    patient_key = int(row[patient_index])
                    disease_dicts['follow_up_start'][patient_key] = _loyalty_str2date(row[follow_up_start_index])
                    disease_dicts['follow_up_total'][patient_key] = float(row[follow_up_total_index])
                    if row[date_index] == '' or row[date_index] == 'NULL':
                        continue
                    disease_dicts['diagnosis_dates'][patient_key] = _loyalty_str2date(row[date_index])
                    if len(disease_dicts['follow_up_start']) % 2000 == 0:
                        logging.debug(f"Processed: {len(disease_dicts['follow_up_start'])} patient rows.")
                except ValueError as e:
                    logging.warning(f'val err {e}')
            logging.info(f"Done processing {diagnosis_column} Got {len(disease_dicts['follow_up_start'])} patient rows and {len(disease_dicts['diagnosis_dates'])} events.")
    except FileNotFoundError as e:
        error = e

    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        if error:
            raise error
        mrn_int = _hd5_filename_to_mrn_int(hd5.filename)
        if mrn_int not in disease_dicts['follow_up_start']:
            raise KeyError(f'{diagnosis_column} did not contain MRN for TensorMap:{tm.name}')
        ecg_dates = list(hd5[tm.path_prefix])
        disease_date = disease_dicts['diagnosis_dates'][mrn_int] if mrn_int in disease_dicts['diagnosis_dates'] else None
        ecg_date_key = _date_from_dates(ecg_dates, disease_date if incidence_only else None)
        ecg_date = datetime.datetime.strptime(ecg_date_key, PARTNERS_DATETIME_FORMAT).date()
        tensor = _ecg_tensor_from_date(tm, hd5, ecg_date_key, population_normalize)

        if mrn_int not in disease_dicts['follow_up_start']:
            raise KeyError(f'{tm.name} mrn not in incidence csv')
        if ecg_date > disease_dicts['follow_up_start'][mrn_int]:
            raise ValueError(f'Assessed after enrollment.')

        for dtm in tm.dependent_map:
            survival_then_censor = np.zeros(tm.dependent_map[dtm].shape, dtype=np.float32)
            if mrn_int not in disease_dicts['diagnosis_dates']:
                has_disease = 0
                censor_date = disease_dicts['follow_up_start'][mrn_int] + datetime.timedelta(days=YEAR_DAYS*disease_dicts['follow_up_total'][mrn_int])
            else:
                has_disease = 1
                censor_date = disease_dicts['diagnosis_dates'][mrn_int]

            intervals = tm.dependent_map[dtm].shape[0] // 2
            days_per_interval = day_window / intervals
            for i, day_delta in enumerate(np.arange(0, day_window, days_per_interval)):
                cur_date = ecg_date + datetime.timedelta(days=day_delta)
                survival_then_censor[i] = float(cur_date < censor_date)
                survival_then_censor[intervals+i] = has_disease * float(censor_date <= cur_date < censor_date + datetime.timedelta(days=days_per_interval))
                if i == 0 and censor_date <= cur_date:  # Handle prevalent diseases
                    survival_then_censor[intervals] = has_disease
                    if has_disease and incidence_only:
                        raise ValueError(f'{tm.name} is skipping prevalent cases.')
            dependents[tm.dependent_map[dtm]] = survival_then_censor
            logging.debug(
                f"Got survival disease {has_disease}, censor: {censor_date}, assess {ecg_date}, fu start {disease_dicts['follow_up_start'][mrn_int]} "
                f"fu total {disease_dicts['follow_up_total'][mrn_int]} tensor:{survival_then_censor[:4]} mid tense: {survival_then_censor[intervals:intervals+4]} ",
            )
        return tensor
    return tensor_from_file


def build_partners_tensor_maps(needed_tensor_maps: List[str]) -> Dict[str, TensorMap]:
    name2tensormap: Dict[str, TensorMap] = {}
    diagnosis2column = {
        'atrial_fibrillation': 'first_af', 'blood_pressure_medication': 'first_bpmed',
        'coronary_artery_disease': 'first_cad', 'cardiovascular_disease': 'first_cvd',
        'death': 'death_date', 'diabetes_mellitus': 'first_dm', 'heart_failure': 'first_hf',
        'hypertension': 'first_htn', 'left_ventricular_hypertrophy': 'first_lvh',
        'myocardial_infarction': 'first_mi', 'pulmonary_artery_disease': 'first_pad',
        'stroke': 'first_stroke', 'valvular_disease': 'first_valvular_disease',
    }
    days_window = 1825
    logging.info(f'needed name {needed_tensor_maps}')
    legacy_csv = '/home/sam/ml/legacy_cohort_overlap.csv'
    for needed_name in needed_tensor_maps:
        if needed_name == 'age_from_csv':
            name2tensormap[needed_name] = TensorMap(needed_name, shape=(1,), tensor_from_file=csv_field_tensor_from_file(INCIDENCE_CSV))
        elif needed_name == 'sex_from_csv':
            csv_tff = csv_field_tensor_from_file(INCIDENCE_CSV, value_column='sex', value_transform=_field_to_index_from_map)
            name2tensormap[needed_name] = TensorMap(needed_name, Interpretation.CATEGORICAL, channel_map={'Female': 0, 'Male': 1}, tensor_from_file=csv_tff)
        elif needed_name == 'age_from_csv_ukb':
            csv_tff = csv_field_tensor_from_file(legacy_csv, value_column='start_fu_age', value_transform=float)
            name2tensormap[needed_name] = TensorMap('21003_Age-when-attended-assessment-centre_2_0', Interpretation.CONTINUOUS, shape=(1,), tensor_from_file=csv_tff,
                                                    normalization={'mean': 63.35798891483556, 'std': 7.554638350423902},
                                                    channel_map={'21003_Age-when-attended-assessment-centre_2_0': 0})
        elif needed_name == 'sex_from_csv_ukb':
            csv_tff = csv_field_tensor_from_file(legacy_csv, value_column='Gender', value_transform=_field_to_index_from_map)
            name2tensormap[needed_name] = TensorMap('Sex_Male_0_0', Interpretation.CATEGORICAL, annotation_units=2, tensor_from_file=csv_tff,
                                                    channel_map={'Sex_Female_0_0': 0, 'Sex_Male_0_0': 1})
        elif needed_name == 'bmi_from_csv_ukb':
            csv_tff = csv_field_tensor_from_file(legacy_csv, value_column='bmi_atStartFu', value_transform=float)
            name2tensormap[needed_name] = TensorMap('21001_Body-mass-index-BMI_0_0', Interpretation.CONTINUOUS, shape=(1,), channel_map={'21001_Body-mass-index-BMI_0_0': 0},
                                                    annotation_units=1,  normalization={'mean': 27.3397, 'std': 4.77216}, tensor_from_file=csv_tff)
        elif needed_name == 'ecg_5000_legacy':
            tff = build_legacy_ecg(legacy_csv)
            name2tensormap[needed_name] = TensorMap('ecg_rest_raw', shape=(5000, 12), path_prefix=PARTNERS_PREFIX, tensor_from_file=tff, channel_map=ECG_REST_UKB_LEADS)
        if 'survival' not in needed_name:
            continue
        potential_day_string = needed_name.split('_')[-1]
        try:
            days_window = int(potential_day_string)
        except ValueError:
            pass

    for diagnosis in diagnosis2column:
        name = f'csv_incident_cox_{diagnosis}'
        if name in needed_tensor_maps:
            tensor_from_file_fxn = csv_time_to_event(INCIDENCE_CSV, incidence_only=True, diagnosis_column=diagnosis2column[diagnosis])
            name2tensormap[name] = TensorMap(name, Interpretation.TIME_TO_EVENT, path_prefix=PARTNERS_PREFIX, tensor_from_file=tensor_from_file_fxn)

        # Build diagnosis classification TensorMaps
        name = f'ecg_2500_to_diagnosis_{diagnosis}'
        if name in needed_tensor_maps:
            tensor_from_file_fxn = build_incidence_tensor_from_file(INCIDENCE_CSV, diagnosis_column=diagnosis2column[diagnosis], population_normalize=2000)
            name2tensormap[f'diagnosis_{diagnosis}'] = TensorMap(f'diagnosis_{diagnosis}', Interpretation.CATEGORICAL, cacheable=False, channel_map=_diagnosis_channels(diagnosis))
            name2tensormap[name] = TensorMap(name, shape=(2500, 12), path_prefix=PARTNERS_PREFIX, channel_map=ECG_REST_AMP_LEADS, cacheable=False,
                                             dependent_map={f'diagnosis_{diagnosis}': name2tensormap[f'diagnosis_{diagnosis}']}, tensor_from_file=tensor_from_file_fxn)
        name = f'ecg_2500_to_incident_{diagnosis}'
        if name in needed_tensor_maps:
            tensor_from_file_fxn = build_incidence_tensor_from_file(INCIDENCE_CSV, diagnosis_column=diagnosis2column[diagnosis], incidence_only=True, population_normalize=2000)
            name2tensormap[f'incident_{diagnosis}'] = TensorMap(f'incident_{diagnosis}', Interpretation.CATEGORICAL, channel_map=_diagnosis_channels(diagnosis, True), cacheable=False)
            name2tensormap[name] = TensorMap(name, shape=(2500, 12), path_prefix=PARTNERS_PREFIX, channel_map=ECG_REST_AMP_LEADS, cacheable=False,
                                             dependent_map={f'incident_{diagnosis}': name2tensormap[f'incident_{diagnosis}']}, tensor_from_file=tensor_from_file_fxn)
        name = f'diagnosis_{diagnosis}_newest'
        if name in needed_tensor_maps:
            tensor_from_file_fxn = build_incidence_tensor_from_file(INCIDENCE_CSV, diagnosis_column=diagnosis2column[diagnosis], dependent=False)
            name2tensormap[name] = TensorMap(name, Interpretation.CATEGORICAL, path_prefix=PARTNERS_PREFIX, channel_map=_diagnosis_channels(diagnosis), tensor_from_file=tensor_from_file_fxn)

        name = f'incident_{diagnosis}_newest'
        if name in needed_tensor_maps:
            tensor_from_file_fxn = build_incidence_tensor_from_file(INCIDENCE_CSV, diagnosis_column=diagnosis2column[diagnosis], incidence_only=True, dependent=False)
            name2tensormap[name] = TensorMap(name,  Interpretation.CATEGORICAL, path_prefix=PARTNERS_PREFIX, channel_map=_diagnosis_channels(diagnosis, True), tensor_from_file=tensor_from_file_fxn)

        # Build time to event TensorMaps
        name = f'ecg_2500_to_cox_{diagnosis}'
        if name in needed_tensor_maps:
            tensor_from_file_fxn = loyalty_time_to_event(INCIDENCE_CSV, diagnosis_column=diagnosis2column[diagnosis], population_normalize=2000)
            name2tensormap[f'cox_{diagnosis}'] = TensorMap(f'cox_{diagnosis}', Interpretation.TIME_TO_EVENT, cacheable=False)
            name2tensormap[name] = TensorMap(name, shape=(2500, 12), path_prefix=PARTNERS_PREFIX, channel_map=ECG_REST_AMP_LEADS, cacheable=False,
                                             dependent_map={f'cox_{diagnosis}': name2tensormap[f'cox_{diagnosis}']}, tensor_from_file=tensor_from_file_fxn)
        name = f'ecg_2500_to_incident_cox_{diagnosis}'
        if name in needed_tensor_maps:
            tensor_from_file_fxn = loyalty_time_to_event(INCIDENCE_CSV, diagnosis_column=diagnosis2column[diagnosis], incidence_only=True, population_normalize=2000)
            name2tensormap[f'incident_cox_{diagnosis}'] = TensorMap(f'incident_cox_{diagnosis}', Interpretation.TIME_TO_EVENT, cacheable=False)
            name2tensormap[name] = TensorMap(name, shape=(2500, 12), path_prefix=PARTNERS_PREFIX, channel_map=ECG_REST_AMP_LEADS, cacheable=False,
                                             dependent_map={f'incident_cox_{diagnosis}': name2tensormap[f'incident_cox_{diagnosis}']}, tensor_from_file=tensor_from_file_fxn)
        name = f'cox_{diagnosis}_newest'
        if name in needed_tensor_maps:
            tensor_from_file_fxn = loyalty_time_to_event(INCIDENCE_CSV, diagnosis_column=diagnosis2column[diagnosis], dependent=False)
            name2tensormap[name] = TensorMap(name,  Interpretation.TIME_TO_EVENT, path_prefix=PARTNERS_PREFIX, tensor_from_file=tensor_from_file_fxn)
        name = f'incident_cox_{diagnosis}_newest'
        if name in needed_tensor_maps:
            tensor_from_file_fxn = loyalty_time_to_event(INCIDENCE_CSV, diagnosis_column=diagnosis2column[diagnosis], incidence_only=True, dependent=False)
            name2tensormap[name] = TensorMap(name,  Interpretation.TIME_TO_EVENT, path_prefix=PARTNERS_PREFIX, tensor_from_file=tensor_from_file_fxn)

        # Build survival curve TensorMaps
        name = f'ecg_2500_to_survival_{diagnosis}_{days_window}'
        if name in needed_tensor_maps:
            tensor_from_file_fxn = _survival_from_file(days_window, INCIDENCE_CSV, diagnosis_column=diagnosis2column[diagnosis], population_normalize=2000)
            name2tensormap[f'survival_{diagnosis}'] = TensorMap(f'survival_{diagnosis}', Interpretation.SURVIVAL_CURVE, shape=(50,), cacheable=False, days_window=days_window)
            name2tensormap[name] = TensorMap(name, shape=(2500, 12), path_prefix=PARTNERS_PREFIX, channel_map=ECG_REST_AMP_LEADS, cacheable=False,
                                             dependent_map={f'survival_{diagnosis}': name2tensormap[f'survival_{diagnosis}']}, tensor_from_file=tensor_from_file_fxn)
        name = f'ecg_2500_to_incident_survival_{diagnosis}_{days_window}'
        if name in needed_tensor_maps:
            tensor_from_file_fxn = _survival_from_file(days_window, INCIDENCE_CSV, diagnosis_column=diagnosis2column[diagnosis], incidence_only=True, population_normalize=2000)
            name2tensormap[f'incident_survival_{diagnosis}'] = TensorMap(f'incident_survival_{diagnosis}', Interpretation.SURVIVAL_CURVE, shape=(50,), cacheable=False, days_window=days_window)
            name2tensormap[name] = TensorMap(name, shape=(2500, 12), path_prefix=PARTNERS_PREFIX, channel_map=ECG_REST_AMP_LEADS, cacheable=False,
                                             dependent_map={f'incident_survival_{diagnosis}': name2tensormap[f'incident_survival_{diagnosis}']}, tensor_from_file=tensor_from_file_fxn)

    logging.info(f'return names {list(name2tensormap.keys())}')
    return name2tensormap


def build_cardiac_surgery_dict(
    filename: str = CARDIAC_SURGERY_OUTCOMES_CSV,
    patient_column: str = 'medrecn',
    date_column: str = 'surgdt',
    additional_columns: List[str] = [],
) -> Dict[int, Dict[str, Union[int, str]]]:
    keys = [date_column] + additional_columns
    cardiac_surgery_dict = {}
    df = pd.read_csv(
        filename,
        low_memory=False,
        usecols=[patient_column]+keys,
    ).sort_values(by=[patient_column, date_column])
    # sort dataframe such that newest surgery per patient appears later and is used in lookup table
    for row in df.itertuples():
        patient_key = getattr(row, patient_column)
        cardiac_surgery_dict[patient_key] = {key: getattr(row, key) for key in keys}
    return cardiac_surgery_dict


def build_date_interval_lookup(
    cardiac_surgery_dict: Dict[int, Dict[str, Union[int, str]]],
    start_column: str = 'surgdt',
    start_offset: int = -30,
    end_column: str = 'surgdt',
    end_offset: int = 0,
) -> Dict[int, Tuple[str, str]]:
    date_interval_lookup = {}
    for mrn in cardiac_surgery_dict:
        start_date = (_cardiac_surgery_str2date(cardiac_surgery_dict[mrn][start_column], PARTNERS_DATETIME_FORMAT.replace('T', ' ')) + datetime.timedelta(days=start_offset)).strftime(PARTNERS_DATETIME_FORMAT)
        end_date = (_cardiac_surgery_str2date(cardiac_surgery_dict[mrn][end_column], PARTNERS_DATETIME_FORMAT.replace('T', ' ')) + datetime.timedelta(days=end_offset)).strftime(PARTNERS_DATETIME_FORMAT)
        date_interval_lookup[mrn] = (start_date, end_date)
    return date_interval_lookup


def make_cardiac_surgery_outcome_tensor_from_file(
    cardiac_surgery_dict: Dict[int, Dict[str, Union[int, str]]],
    outcome_column: str,
) -> Callable:
    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}):
        mrn = _hd5_filename_to_mrn_int(hd5.filename)
        tensor = np.zeros(tm.shape, dtype=np.float32)
        outcome = cardiac_surgery_dict[mrn][outcome_column]

        if type(outcome) is float and not outcome.is_integer():
            raise ValueError(f'Cardiac Surgery categorical outcome {tm.name} ({outcome_column}) got non-discrete value: {outcome}')

        # ensure binary outcome
        if outcome != 0 and outcome != 1:
            raise ValueError(f'Cardiac Surgery categorical outcome {tm.name} ({outcome_column}) got non-binary value: {outcome}')

        tensor[outcome] = 1
        return tensor
    return tensor_from_file


def build_cardiac_surgery_tensor_maps(
    needed_tensor_maps: List[str],
) -> Dict[str, TensorMap]:
    name2tensormap: Dict[str, TensorMap] = {}
    outcome2column = {
        "sts_death": "mtopd",
        "sts_stroke": "cnstrokp",
        "sts_renal_failure": "crenfail",
        "sts_prolonged_ventilation": "cpvntlng",
        "sts_dsw_infection": "deepsterninf",
        "sts_reoperation": "reop",
        "sts_any_morbidity": "anymorbidity",
        "sts_long_stay": "llos",
    }

    cardiac_surgery_dict = None
    date_interval_lookup = None
    for needed_name in needed_tensor_maps:
        if needed_name in outcome2column:
            if cardiac_surgery_dict is None:
                cardiac_surgery_dict = build_cardiac_surgery_dict(additional_columns=[column for outcome, column in outcome2column.items() if outcome in needed_tensor_maps])
            channel_map = _outcome_channels(needed_name)
            sts_tmap = TensorMap(
                needed_name,
                Interpretation.CATEGORICAL,
                path_prefix=PARTNERS_PREFIX,
                tensor_from_file=make_cardiac_surgery_outcome_tensor_from_file(cardiac_surgery_dict, outcome2column[needed_name]),
                channel_map=channel_map,
                validator=validator_not_all_zero,
            )
        else:
            if not needed_name.endswith('_sts'):
                continue

            base_name = needed_name.split('_sts')[0]
            if base_name not in TMAPS:
                TMAPS.update(build_partners_time_series_tensor_maps([base_name]))
                if base_name not in TMAPS:
                    continue

            if cardiac_surgery_dict is None:
                cardiac_surgery_dict = build_cardiac_surgery_dict(additional_columns=[column for outcome, column in outcome2column.items() if outcome in needed_tensor_maps])
            if date_interval_lookup is None:
                date_interval_lookup = build_date_interval_lookup(cardiac_surgery_dict)
            sts_tmap = copy.deepcopy(TMAPS[base_name])
            sts_tmap.name = needed_name
            sts_tmap.time_series_lookup = date_interval_lookup

        name2tensormap[needed_name] = sts_tmap

    return name2tensormap
