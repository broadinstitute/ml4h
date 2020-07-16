# Imports: standard library
import os
import csv
import copy
import logging
import datetime
from typing import Dict, List, Tuple, Union, Callable
from itertools import product, combinations
from collections import defaultdict

# Imports: third party
import h5py
import numpy as np
import pandas as pd

# Imports: first party
from ml4cvd.defines import (
    STOP_CHAR,
    ECG_DATE_FORMAT,
    ECG_REST_AMP_LEADS,
    ECG_DATETIME_FORMAT,
    ECG_REST_INDEPENDENT_LEADS,
    CARDIAC_SURGERY_DATE_FORMAT,
)
from ml4cvd.metrics import weighted_crossentropy
from ml4cvd.TensorMap import (
    TensorMap,
    Interpretation,
    TimeSeriesOrder,
    str2date,
    decompress_data,
    make_range_validator,
)
from ml4cvd.normalizer import Standardize, ZeroMeanStd1

YEAR_DAYS = 365.26
CARDIAC_SURGERY_OUTCOMES_CSV = "/data/sts-data/mgh-preop-ecg-outcome-labels.csv"
ECG_PREFIX = "partners_ecg_rest"
TMAPS = dict()


def _hd5_filename_to_mrn_int(filename: str) -> int:
    return int(os.path.basename(filename).split(".")[0])


def _get_ecg_dates(tm, hd5):
    if not hasattr(_get_ecg_dates, "mrn_lookup"):
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
        raise NotImplementedError(
            f'Unknown option "{tm.time_series_order}" passed for which tensors to use'
            " in multi tensor HD5",
        )
    start_idx = tm.time_series_limit if tm.time_series_limit is not None else 1
    dates = dates[-start_idx:]  # If num_tensors is 0, get all tensors
    dates.sort(reverse=True)
    _get_ecg_dates.mrn_lookup[mrn] = dates
    return dates


def validator_no_empty(tm: TensorMap, tensor: np.ndarray, hd5: h5py.File):
    if any(tensor == ""):
        raise ValueError(f"TensorMap {tm.name} failed empty string check.")


def validator_no_negative(tm: TensorMap, tensor: np.ndarray, hd5: h5py.File):
    if any(tensor < 0):
        raise ValueError(f"TensorMap {tm.name} failed non-negative check")


def validator_not_all_zero(tm: TensorMap, tensor: np.ndarray, hd5: h5py.File):
    if np.count_nonzero(tensor) == 0:
        raise ValueError(f"TensorMap {tm.name} failed all-zero check")


def _is_dynamic_shape(tm: TensorMap, num_ecgs: int) -> Tuple[bool, Tuple[int, ...]]:
    if tm.shape[0] is None:
        return True, (num_ecgs,) + tm.shape[1:]
    return False, tm.shape


def _make_hd5_path(tm, ecg_date, value_key):
    return f"{tm.path_prefix}/{ecg_date}/{value_key}"


def _resample_voltage(voltage, desired_samples):
    length_mismatch = False
    if len(voltage) == desired_samples:
        return voltage
    # Upsample
    elif len(voltage) < desired_samples:
        if desired_samples % len(voltage) == 0:
            x = np.arange(len(voltage))
            x_interp = np.linspace(0, len(voltage), len(desired_samples))
            return np.interp(x_interp, x, voltage)
        else:
            length_mismatch = True
    # Downsample
    else:  # len(voltage) > desired_samples
        if len(voltage) % desired_samples == 0:
            return voltage[:: int(len(voltage) / desired_samples)]
        else:
            length_mismatch = True

    if length_mismatch:
        raise ValueError(
            f"Cannot cleanly resample voltage length {len(voltage)} to desired samples"
            f" {desired_samples}",
        )


def make_voltage(exact_length=False):
    def get_voltage_from_file(tm, hd5, dependents={}):
        ecg_dates = _get_ecg_dates(tm, hd5)
        dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
        voltage_length = shape[1] if dynamic else shape[0]
        tensor = np.zeros(shape, dtype=np.float32)
        for i, ecg_date in enumerate(ecg_dates):
            for cm in tm.channel_map:
                try:
                    path = _make_hd5_path(tm, ecg_date, cm)
                    voltage = decompress_data(
                        data_compressed=hd5[path][()], dtype=hd5[path].attrs["dtype"],
                    )
                    if exact_length:
                        assert len(voltage) == voltage_length
                    voltage = _resample_voltage(voltage, voltage_length)
                    slices = (
                        (i, ..., tm.channel_map[cm])
                        if dynamic
                        else (..., tm.channel_map[cm])
                    )
                    tensor[slices] = voltage
                except (KeyError, AssertionError, ValueError):
                    logging.debug(
                        f"Could not get voltage for lead {cm} with {voltage_length}"
                        f" samples in {hd5.filename}",
                    )
        return tensor

    return get_voltage_from_file


# ECG augmentations
def _crop_ecg(ecg: np.array):
    cropped_ecg = ecg.copy()
    for j in range(ecg.shape[1]):
        crop_len = np.random.randint(len(ecg)) // 3
        crop_start = max(0, np.random.randint(-crop_len, len(ecg)))
        cropped_ecg[:, j][crop_start : crop_start + crop_len] = np.random.randn()
    return cropped_ecg


def _noise_ecg(ecg: np.array):
    noise_frac = np.random.rand() * 0.1  # max of 10% noise
    return ecg + noise_frac * ecg.mean(axis=0) * np.random.randn(
        ecg.shape[0], ecg.shape[1],
    )


def _warp_ecg(ecg: np.array):
    warp_strength = 0.02
    i = np.linspace(0, 1, len(ecg))
    envelope = warp_strength * (0.5 - np.abs(0.5 - i))
    warped = i + envelope * (
        np.sin(np.random.rand() * 5 + np.random.randn() * 5)
        + np.cos(np.random.rand() * 5 + np.random.randn() * 5)
    )
    warped_ecg = np.zeros_like(ecg)
    for j in range(ecg.shape[1]):
        warped_ecg[:, j] = np.interp(i, warped, ecg[:, j])
    return warped_ecg


# Dict[str, Callable]
augmentations = {
    "crop": _crop_ecg,
    "noise": _noise_ecg,
    "warp": _warp_ecg,
}


def _make_augmentation_combinations(
    augmentations: Dict[str, Callable],
) -> Dict[str, List[Callable]]:
    """Iterate through all combinations of augmentations, and return a dictionary
    of callables that are keyed by the concatenated names"""
    # Initialize dict in which to store augmentation combos
    unique_augmentations = dict()

    # Iterate through all possible lengths, e.g. 1, 2, 3-augment combos
    for r in range(1, len(augmentations) + 1):

        # Iterate through unique augmentation sets
        for aug_set in list(combinations(augmentations, r)):

            # Get list of augmentation functions for this set
            aug_funcs = [augmentations[aug] for aug in aug_set if aug in augmentations]

            # Get single string of name to append to TMap name
            aug_name = "_" + "_".join(sorted(aug_set))

            unique_augmentations[aug_name] = aug_funcs

    # Add empty entry for no augmentation
    unique_augmentations[""] = None

    return unique_augmentations


# Generates a diverse family of ECG TMaps with name modifications:
#
# ecg_625     ecg_625_std     ecg_625_std_exact     ecg_625_raw     ecg_625_raw_exact
# ecg_1250    ecg_1250_std    ecg_1250_std_exact    ecg_1250_raw    ecg_1250_raw_exact
# ecg_2500    ecg_2500_std    ecg_2500_std_exact    ecg_2500_raw    ecg_2500_raw_exact
# ecg_5000    ecg_5000_std    ecg_5000_std_exact    ecg_5000_raw    ecg_5000_raw_exact
#
# plus all permutations of above with 1-3 augmentations per TMap (see below).
#
# Generated TMaps have modifications in order:
#     [12_lead_]ecg_length[_normalization][_exact][_augmentations]
#
# length: number of samples in ECG voltage array
#    625
#    1250
#    2500
#    5000
#
# normalization (which also handles resampling):
#    None:      resample input to length, and normalize with ZeroMeanStd1
#    exact:     only return array with len(array) == length
#    std:       resample input to length; normalize with Standardize mean = 0, std = 2000
#    std_exact: normalize with Standardize mean = 0, std = 2000;
#               only return array with len(array) == length0
#    raw:       resample input to length; original voltage values; does not normalize
#    raw_exact: does not normalize; only return array with len(array) == length
#
# augmentations: modifications to arrays
#    crop:  isolate a random subsection of the ECG
#    noise: add Gaussian noise
#    warp:  add bendiness
#
# When using these generated TMaps, you must use the exact name;
#     valid:     ecg_2500_std_exact_crop_noise
#     not valid: ecg_2500_std_exact_noise_crop
#     not valid: ecg_2500_exact_std_noise_crop
#
length_options = [625, 1250, 2500, 5000]
exact_options = [True, False]
augmentation_options = _make_augmentation_combinations(augmentations)
normalize_options = [ZeroMeanStd1(), Standardize(mean=0, std=2000), None]
leads_options = [ECG_REST_INDEPENDENT_LEADS, ECG_REST_AMP_LEADS]

for leads, length, exact_length, normalization, augmentation in product(
    leads_options,
    length_options,
    exact_options,
    normalize_options,
    augmentation_options,
):
    norm = (
        ""
        if isinstance(normalization, ZeroMeanStd1)
        else "_std"
        if isinstance(normalization, Standardize)
        else "_raw"
    )
    exact = "_exact" if exact_length else ""

    num_leads = ""
    if len(leads) != 8:
        num_leads = f"{len(leads)}_lead_"
    name = f"{num_leads}ecg_{length}{norm}{exact}{augmentation}"
    TMAPS[name] = TensorMap(
        name,
        shape=(None, length, len(leads)),
        path_prefix=ECG_PREFIX,
        tensor_from_file=make_voltage(exact_length),
        normalization=normalization,
        channel_map=leads,
        time_series_limit=0,
        validator=validator_not_all_zero,
        augmentations=augmentation_options[augmentation],
    )


def voltage_stat(tm, hd5, dependents={}):
    ecg_dates = _get_ecg_dates(tm, hd5)
    dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
    tensor = np.zeros(shape, dtype=np.float32)
    for i, ecg_date in enumerate(ecg_dates):
        try:
            slices = (
                lambda stat: (i, tm.channel_map[stat])
                if dynamic
                else (tm.channel_map[stat],)
            )
            path = lambda lead: _make_hd5_path(tm, ecg_date, lead)
            voltages = np.array(
                [
                    decompress_data(data_compressed=hd5[path(lead)][()], dtype="int16")
                    for lead in ECG_REST_AMP_LEADS
                ],
            )
            tensor[slices("mean")] = np.mean(voltages)
            tensor[slices("std")] = np.std(voltages)
            tensor[slices("min")] = np.min(voltages)
            tensor[slices("max")] = np.max(voltages)
            tensor[slices("median")] = np.median(voltages)
        except KeyError:
            logging.warning(f"Could not get voltage stats for ECG at {hd5.filename}")
    return tensor


TMAPS["ecg_voltage_stats"] = TensorMap(
    "ecg_voltage_stats",
    shape=(None, 5),
    path_prefix=ECG_PREFIX,
    tensor_from_file=voltage_stat,
    channel_map={"mean": 0, "std": 1, "min": 2, "max": 3, "median": 4},
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
                    slices = (
                        (i, tm.channel_map[cm]) if dynamic else (tm.channel_map[cm],)
                    )
                    tensor[slices] = hd5[path].attrs[volt_attr]
                except KeyError:
                    pass
        return tensor

    return get_voltage_attr_from_file


TMAPS["voltage_len"] = TensorMap(
    "voltage_len",
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_voltage_attr(volt_attr="len"),
    shape=(None, 12),
    channel_map=ECG_REST_AMP_LEADS,
    time_series_limit=0,
)


def make_ecg_label(
    keys: Union[str, List[str]] = "read_md_clean",
    dict_of_list: Dict = dict(),
    not_found_key: str = "unspecified",
):
    if type(keys) == str:
        keys = [keys]

    def get_ecg_label(tm, hd5, dependents={}):
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
                    read = decompress_data(
                        data_compressed=hd5[path][()], dtype=hd5[path].attrs["dtype"],
                    )
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
                slices = (
                    (i, tm.channel_map[not_found_key])
                    if dynamic
                    else (tm.channel_map[not_found_key],)
                )
                label_array[slices] = 1
        return label_array

    return get_ecg_label


def ecg_datetime(tm, hd5, dependents={}):
    ecg_dates = _get_ecg_dates(tm, hd5)
    dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
    tensor = np.full(shape, "", dtype=f"<U19")
    for i, ecg_date in enumerate(ecg_dates):
        tensor[i] = ecg_date
    return tensor


task = "ecg_datetime"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=ecg_datetime,
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


def make_voltage_len_categorical_tmap(
    lead, channel_prefix="_", channel_unknown="other",
):
    def _tensor_from_file(tm, hd5, dependents={}):
        ecg_dates = _get_ecg_dates(tm, hd5)
        dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
        tensor = np.zeros(shape, dtype=float)
        for i, ecg_date in enumerate(ecg_dates):
            path = _make_hd5_path(tm, ecg_date, lead)
            try:
                lead_len = hd5[path].attrs["len"]
                lead_len = f"{channel_prefix}{lead_len}"
                matched = False
                for cm in tm.channel_map:
                    if lead_len.lower() == cm.lower():
                        slices = (
                            (i, tm.channel_map[cm])
                            if dynamic
                            else (tm.channel_map[cm],)
                        )
                        tensor[slices] = 1.0
                        matched = True
                        break
                if not matched:
                    slices = (
                        (i, tm.channel_map[channel_unknown])
                        if dynamic
                        else (tm.channel_map[channel_unknown],)
                    )
                    tensor[slices] = 1.0
            except KeyError:
                logging.debug(
                    f"Could not get voltage length for lead {lead} from ECG on"
                    f" {ecg_date} in {hd5.filename}",
                )
        return tensor

    return _tensor_from_file


for lead in ECG_REST_AMP_LEADS:
    tmap_name = f"lead_{lead}_len"
    TMAPS[tmap_name] = TensorMap(
        tmap_name,
        interpretation=Interpretation.CATEGORICAL,
        path_prefix=ECG_PREFIX,
        tensor_from_file=make_voltage_len_categorical_tmap(lead=lead),
        channel_map={"_2500": 0, "_5000": 1, "other": 2},
        time_series_limit=0,
        validator=validator_not_all_zero,
    )


def make_ecg_tensor(
    key: str, fill: float = 0, channel_prefix: str = "", channel_unknown: str = "other",
):
    def get_ecg_tensor(tm, hd5, dependents={}):
        ecg_dates = _get_ecg_dates(tm, hd5)
        dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
        if tm.interpretation == Interpretation.LANGUAGE:
            tensor = np.full(shape, "", dtype=object)
        elif tm.interpretation == Interpretation.CONTINUOUS:
            tensor = (
                np.zeros(shape, dtype=np.float32)
                if fill == 0
                else np.full(shape, fill, dtype=np.float32)
            )
        elif tm.interpretation == Interpretation.CATEGORICAL:
            tensor = np.zeros(shape, dtype=float)
        else:
            raise NotImplementedError(
                f"unsupported interpretation for ecg tmaps: {tm.interpretation}",
            )

        for i, ecg_date in enumerate(ecg_dates):
            path = _make_hd5_path(tm, ecg_date, key)
            try:
                data = decompress_data(data_compressed=hd5[path][()], dtype="str")
                if tm.interpretation == Interpretation.CATEGORICAL:
                    matched = False
                    data = f"{channel_prefix}{data}"
                    for cm in tm.channel_map:
                        if data.lower() == cm.lower():
                            slices = (
                                (i, tm.channel_map[cm])
                                if dynamic
                                else (tm.channel_map[cm],)
                            )
                            tensor[slices] = 1.0
                            matched = True
                            break
                    if not matched:
                        slices = (
                            (i, tm.channel_map[channel_unknown])
                            if dynamic
                            else (tm.channel_map[channel_unknown],)
                        )
                        tensor[slices] = 1.0
                else:
                    tensor[i] = data
            except (KeyError, ValueError):
                logging.debug(
                    f"Could not obtain tensor {tm.name} from ECG on {ecg_date} in"
                    f" {hd5.filename}",
                )

        if tm.interpretation == Interpretation.LANGUAGE:
            tensor = tensor.astype(str)
        return tensor

    return get_ecg_tensor


def make_language_tensor(key: str):
    def language_tensor(tm, hd5, dependents={}):
        words = str(
            decompress_data(
                data_compressed=hd5[key][()], dtype=hd5[key].attrs["dtype"],
            ),
        )
        tensor = np.zeros(tm.shape, dtype=np.float32)
        for i, c in enumerate(words):
            if i >= tm.shape[0]:
                logging.debug(
                    f"Text {words} is longer than {tm.name} can store in"
                    f" shape:{tm.shape}, truncating...",
                )
                break
            tensor[i, tm.channel_map[c]] = 1.0
        tensor[min(tm.shape[0] - 1, i + 1), tm.channel_map[STOP_CHAR]] = 1.0
        return tensor

    return language_tensor


task = "ecg_read_md"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="read_md_clean"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


task = "ecg_read_pc"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="read_pc_clean"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


task = "ecg_patientid"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="patientid"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


def validator_clean_mrn(tm: TensorMap, tensor: np.ndarray, hd5: h5py.File):
    int(tensor)


task = "ecg_patientid_clean"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="patientid_clean"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_clean_mrn,
)


task = "ecg_firstname"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="patientfirstname"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


task = "ecg_lastname"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="patientlastname"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


task = "ecg_sex"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="gender"),
    channel_map={"female": 0, "male": 1},
    time_series_limit=0,
    validator=validator_not_all_zero,
)

task = "ecg_date"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="acquisitiondate"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


task = "ecg_time"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="acquisitiontime"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


task = "ecg_sitename"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="sitename"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


task = "ecg_location"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="location"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


task = "ecg_dob"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="dateofbirth"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


def make_sampling_frequency_from_file(
    lead: str = "I",
    duration: int = 10,
    channel_prefix: str = "_",
    channel_unknown: str = "other",
    fill: int = -1,
):
    def sampling_frequency_from_file(
        tm: TensorMap, hd5: h5py.File, dependents: Dict = {},
    ):
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
                    sampling_frequency = f"{channel_prefix}{sampling_frequency}"
                    for cm in tm.channel_map:
                        if sampling_frequency.lower() == cm.lower():
                            slices = (
                                (i, tm.channel_map[cm])
                                if dynamic
                                else (tm.channel_map[cm],)
                            )
                            tensor[slices] = 1.0
                            matched = True
                            break
                    if not matched:
                        slices = (
                            (i, tm.channel_map[channel_unknown])
                            if dynamic
                            else (tm.channel_map[channel_unknown],)
                        )
                        tensor[slices] = 1.0
                else:
                    tensor[i] = sampling_frequency
            except (KeyError, ValueError):
                logging.debug(
                    f"Could not calculate sampling frequency from ECG on {ecg_date} in"
                    f" {hd5.filename}",
                )
        return tensor

    return sampling_frequency_from_file


# sampling frequency without any suffix calculates the sampling frequency directly from the voltage array
# other metadata that are reported by the muse system are unreliable
task = "ecg_sampling_frequency"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_sampling_frequency_from_file(),
    channel_map={"_250": 0, "_500": 1, "other": 2},
    time_series_limit=0,
    validator=validator_not_all_zero,
)


task = "ecg_sampling_frequency_pc"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="ecgsamplebase_pc", channel_prefix="_"),
    channel_map={"_0": 0, "_250": 1, "_500": 2, "other": 3},
    time_series_limit=0,
    validator=validator_not_all_zero,
)


task = "ecg_sampling_frequency_md"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="ecgsamplebase_md", channel_prefix="_"),
    channel_map={"_0": 0, "_250": 1, "_500": 2, "other": 3},
    time_series_limit=0,
    validator=validator_not_all_zero,
)


task = "ecg_sampling_frequency_lead"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="waveform_samplebase", channel_prefix="_"),
    channel_map={"_0": 0, "_240": 1, "_250": 2, "_500": 3, "other": 4},
    time_series_limit=0,
    validator=validator_not_all_zero,
)


task = "ecg_sampling_frequency_continuous"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_sampling_frequency_from_file(),
    time_series_limit=0,
    shape=(None, 1),
    validator=validator_no_negative,
)


task = "ecg_sampling_frequency_pc_continuous"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="ecgsamplebase_pc", fill=-1),
    time_series_limit=0,
    shape=(None, 1),
    validator=validator_no_negative,
)


task = "ecg_sampling_frequency_md_continuous"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="ecgsamplebase_md", fill=-1),
    time_series_limit=0,
    shape=(None, 1),
    validator=validator_no_negative,
)


task = "ecg_sampling_frequency_lead_continuous"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="waveform_samplebase", fill=-1),
    time_series_limit=0,
    shape=(None, 1),
    validator=validator_no_negative,
)


task = "ecg_time_resolution"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(
        key="intervalmeasurementtimeresolution", channel_prefix="_",
    ),
    channel_map={"_25": 0, "_50": 1, "_100": 2, "other": 3},
    time_series_limit=0,
    validator=validator_not_all_zero,
)


task = "ecg_amplitude_resolution"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(
        key="intervalmeasurementamplituderesolution", channel_prefix="_",
    ),
    channel_map={"_10": 0, "_20": 1, "_40": 2, "other": 3},
    time_series_limit=0,
    validator=validator_not_all_zero,
)


task = "ecg_measurement_filter"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(
        key="intervalmeasurementfilter", channel_prefix="_",
    ),
    time_series_limit=0,
    channel_map={"_None": 0, "_40": 1, "_80": 2, "other": 3},
    validator=validator_not_all_zero,
)


task = "ecg_high_pass_filter"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="waveform_highpassfilter", fill=-1),
    time_series_limit=0,
    shape=(None, 1),
    validator=validator_no_negative,
)


task = "ecg_low_pass_filter"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="waveform_lowpassfilter", fill=-1),
    time_series_limit=0,
    shape=(None, 1),
    validator=validator_no_negative,
)


task = "ecg_ac_filter"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="waveform_acfilter", channel_prefix="_"),
    time_series_limit=0,
    channel_map={"_None": 0, "_50": 1, "_60": 2, "other": 3},
    validator=validator_not_all_zero,
)

# Creates TMaps for interval measurements.
# Creates TMaps with _md and _pc suffix.
# Examples:
#     ecg_rate_md
#     ecg_rate_pc

# fmt: off
# TMap name      ->      (hd5 key,          fill, validator,                       normalization)
interval_key_map = {
    "ecg_rate":          ("ventricularrate", 0,   make_range_validator(10, 200),   None),
    "ecg_rate_std":      ("ventricularrate", 0,   make_range_validator(10, 200),   Standardize(mean=70, std=16)),
    "ecg_pr":            ("printerval",      0,   make_range_validator(50, 500),   None),
    "ecg_pr_std":        ("printerval",      0,   make_range_validator(50, 500),   Standardize(mean=175, std=36)),
    "ecg_qrs":           ("qrsduration",     0,   make_range_validator(20, 400),   None),
    "ecg_qrs_std":       ("qrsduration",     0,   make_range_validator(20, 400),   Standardize(mean=104, std=26)),
    "ecg_qt":            ("qtinterval",      0,   make_range_validator(100, 800),  None),
    "ecg_qt_std":        ("qtinterval",      0,   make_range_validator(100, 800),  Standardize(mean=411, std=45)),
    "ecg_qtc":           ("qtcorrected",     0,   make_range_validator(100, 800),  None),
    "ecg_qtc_std":       ("qtcorrected",     0,   make_range_validator(100, 800),  Standardize(mean=440, std=39)),
    "ecg_paxis":         ("paxis",           999, make_range_validator(-90, 360),  None),
    "ecg_paxis_std":     ("paxis",           999, make_range_validator(-90, 360),  Standardize(mean=47, std=30)),
    "ecg_raxis":         ("raxis",           999, make_range_validator(-90, 360),  None),
    "ecg_raxis_std":     ("raxis",           999, make_range_validator(-90, 360),  Standardize(mean=18, std=53)),
    "ecg_taxis":         ("taxis",           999, make_range_validator(-90, 360),  None),
    "ecg_taxis_std":     ("taxis",           999, make_range_validator(-90, 360),  Standardize(mean=58, std=63)),
    "ecg_qrs_count":     ("qrscount",        -1,  make_range_validator(0, 100),    None),
    "ecg_qrs_count_std": ("qrscount",        -1,  make_range_validator(0, 100),    Standardize(mean=12, std=3)),
    "ecg_qonset":        ("qonset",          -1,  make_range_validator(0, 500),    None),
    "ecg_qonset_std":    ("qonset",          -1,  make_range_validator(0, 500),    Standardize(mean=204, std=36)),
    "ecg_qoffset":       ("qoffset",         -1,  make_range_validator(0, 500),    None),
    "ecg_qoffset_std":   ("qoffset",         -1,  make_range_validator(0, 500),    Standardize(mean=252, std=44)),
    "ecg_ponset":        ("ponset",          -1,  make_range_validator(0, 1000),   None),
    "ecg_ponset_std":    ("ponset",          -1,  make_range_validator(0, 1000),   Standardize(mean=122, std=27)),
    "ecg_poffset":       ("poffset",         -1,  make_range_validator(10, 500),   None),
    "ecg_poffset_std":   ("poffset",         -1,  make_range_validator(10, 500),   Standardize(mean=170, std=42)),
    "ecg_toffset":       ("toffset",         -1,  make_range_validator(0, 1000),   None),
    "ecg_toffset_std":   ("toffset",         -1,  make_range_validator(0, 1000),   Standardize(mean=397, std=73)),
}
# fmt: on

for interval, (key, fill, validator, normalization) in interval_key_map.items():
    for suffix in ["_md", "_pc"]:
        name = f"{interval}{suffix}"
        _key = f"{key}{suffix}"
        TMAPS[name] = TensorMap(
            name,
            interpretation=Interpretation.CONTINUOUS,
            path_prefix=ECG_PREFIX,
            loss="logcosh",
            tensor_from_file=make_ecg_tensor(key=_key, fill=fill),
            shape=(None, 1),
            time_series_limit=0,
            validator=validator,
            normalization=normalization,
        )


task = "ecg_weight_lbs"
TMAPS[task] = TensorMap(
    task,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    loss="logcosh",
    tensor_from_file=make_ecg_tensor(key="weightlbs"),
    shape=(None, 1),
    time_series_limit=0,
    validator=make_range_validator(100, 800),
)


def ecg_age(tm, hd5, dependents={}):
    ecg_dates = _get_ecg_dates(tm, hd5)
    dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
    tensor = np.zeros(shape, dtype=float)
    for i, ecg_date in enumerate(ecg_dates):
        if i >= shape[0]:
            break
        path = lambda key: _make_hd5_path(tm, ecg_date, key)
        try:
            birthday = decompress_data(
                data_compressed=hd5[path("dateofbirth")][()], dtype="str",
            )
            acquisition = decompress_data(
                data_compressed=hd5[path("acquisitiondate")][()], dtype="str",
            )
            delta = _ecg_str2date(acquisition) - _ecg_str2date(birthday)
            years = delta.days / YEAR_DAYS
            tensor[i] = years
        except KeyError:
            try:
                tensor[i] = decompress_data(
                    data_compressed=hd5[path("patientage")][()], dtype="str",
                )
            except KeyError:
                logging.debug(
                    f"Could not get patient date of birth or age from ECG on {ecg_date}"
                    f" in {hd5.filename}",
                )
    return tensor


task = "ecg_age"
TMAPS[task] = TensorMap(
    task,
    path_prefix=ECG_PREFIX,
    loss="logcosh",
    tensor_from_file=ecg_age,
    shape=(None, 1),
    time_series_limit=0,
    validator=make_range_validator(0, 120),
)

task = "ecg_age_std"
TMAPS[task] = TensorMap(
    task,
    path_prefix=ECG_PREFIX,
    loss="logcosh",
    tensor_from_file=ecg_age,
    shape=(None, 1),
    time_series_limit=0,
    validator=make_range_validator(0, 120),
    normalization=Standardize(mean=65, std=16),
)


def ecg_acquisition_year(tm, hd5, dependents={}):
    ecg_dates = _get_ecg_dates(tm, hd5)
    dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
    tensor = np.zeros(shape, dtype=int)
    for i, ecg_date in enumerate(ecg_dates):
        path = _make_hd5_path(tm, ecg_date, "acquisitiondate")
        try:
            acquisition = decompress_data(data_compressed=hd5[path][()], dtype="str")
            tensor[i] = _ecg_str2date(acquisition).year
        except KeyError:
            pass
    return tensor


TMAPS["ecg_acquisition_year"] = TensorMap(
    "ecg_acquisition_year",
    path_prefix=ECG_PREFIX,
    loss="logcosh",
    tensor_from_file=ecg_acquisition_year,
    shape=(None, 1),
    time_series_limit=0,
)


def ecg_bmi(tm, hd5, dependents={}):
    ecg_dates = _get_ecg_dates(tm, hd5)
    dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
    tensor = np.zeros(shape, dtype=float)
    for i, ecg_date in enumerate(ecg_dates):
        path = lambda key: _make_hd5_path(tm, ecg_date, key)
        try:
            weight_lbs = decompress_data(
                data_compressed=hd5[path("weightlbs")][()], dtype="str",
            )
            weight_kg = 0.453592 * float(weight_lbs)
            height_in = decompress_data(
                data_compressed=hd5[path("heightin")][()], dtype="str",
            )
            height_m = 0.0254 * float(height_in)
            bmi = weight_kg / (height_m * height_m)
            logging.info(f" Height was {height_in} weight: {weight_lbs} bmi is {bmi}")
            tensor[i] = bmi
        except KeyError:
            pass
    return tensor


TMAPS["ecg_bmi"] = TensorMap(
    "ecg_bmi",
    path_prefix=ECG_PREFIX,
    channel_map={"bmi": 0},
    tensor_from_file=ecg_bmi,
    time_series_limit=0,
)


def ecg_channel_string(hd5_key, race_synonyms={}, unspecified_key=None):
    def tensor_from_string(tm, hd5, dependents={}):
        ecg_dates = _get_ecg_dates(tm, hd5)
        dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
        tensor = np.zeros(shape, dtype=np.float32)
        for i, ecg_date in enumerate(ecg_dates):
            path = _make_hd5_path(tm, ecg_date, hd5_key)
            found = False
            try:
                hd5_string = decompress_data(data_compressed=hd5[path][()], dtype="str")
                for key in tm.channel_map:
                    slices = (
                        (i, tm.channel_map[key]) if dynamic else (tm.channel_map[key],)
                    )
                    if hd5_string.lower() == key.lower():
                        tensor[slices] = 1.0
                        found = True
                        break
                    if key in race_synonyms:
                        for synonym in race_synonyms[key]:
                            if hd5_string.lower() == synonym.lower():
                                tensor[slices] = 1.0
                                found = True
                            if found:
                                break
                        if found:
                            break
            except KeyError:
                pass
            if not found:
                if unspecified_key is None:
                    # TODO Do we want to try to continue to get tensors for other ECGs in HD5?
                    raise ValueError(
                        f"No channel keys found in {hd5_string} for {tm.name} with"
                        f" channel map {tm.channel_map}.",
                    )
                slices = (
                    (i, tm.channel_map[unspecified_key])
                    if dynamic
                    else (tm.channel_map[unspecified_key],)
                )
                tensor[slices] = 1.0
        return tensor

    return tensor_from_string


race_synonyms = {"asian": ["oriental"], "hispanic": ["latino"], "white": ["caucasian"]}
TMAPS["ecg_race"] = TensorMap(
    "ecg_race",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"asian": 0, "black": 1, "hispanic": 2, "white": 3, "unknown": 4},
    tensor_from_file=ecg_channel_string("race", race_synonyms),
    time_series_limit=0,
)


def _ecg_adult(hd5_key, minimum_age=18):
    def tensor_from_string(tm, hd5, dependents={}):
        ecg_dates = _get_ecg_dates(tm, hd5)
        dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
        tensor = np.zeros(shape, dtype=np.float32)
        for i, ecg_date in enumerate(ecg_dates):
            path = lambda key: _make_hd5_path(tm, ecg_date, key)
            birthday = decompress_data(
                data_compressed=hd5[path("dateofbirth")][()], dtype="str",
            )
            acquisition = decompress_data(
                data_compressed=hd5[path("acquisitiondate")][()], dtype="str",
            )
            delta = _ecg_str2date(acquisition) - _ecg_str2date(birthday)
            years = delta.days / YEAR_DAYS
            if years < minimum_age:
                raise ValueError(f"ECG taken on patient below age cutoff.")
            hd5_string = decompress_data(
                data_compressed=hd5[path(hd5_key)][()],
                dtype=hd5[path(hd5_key)].attrs["dtype"],
            )
            found = False
            for key in tm.channel_map:
                if hd5_string.lower() == key.lower():
                    slices = (
                        (i, tm.channel_map[key]) if dynamic else (tm.channel_map[key],)
                    )
                    tensor[slices] = 1.0
                    found = True
                    break
            if not found:
                # TODO Do we want to try to continue to get tensors for other ECGs in HD5?
                raise ValueError(
                    f"No channel keys found in {hd5_string} for {tm.name} with channel"
                    f" map {tm.channel_map}.",
                )
        return tensor

    return tensor_from_string


TMAPS["ecg_adult_sex"] = TensorMap(
    "ecg_adult_sex",
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    channel_map={"female": 0, "male": 1},
    tensor_from_file=_ecg_adult("gender"),
    time_series_limit=0,
)


def voltage_zeros(tm, hd5, dependents={}):
    ecg_dates = _get_ecg_dates(tm, hd5)
    dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
    tensor = np.zeros(shape, dtype=np.float32)
    for i, ecg_date in enumerate(ecg_dates):
        for cm in tm.channel_map:
            path = _make_hd5_path(tm, ecg_date, cm)
            voltage = decompress_data(
                data_compressed=hd5[path][()], dtype=hd5[path].attrs["dtype"],
            )
            slices = (i, tm.channel_map[cm]) if dynamic else (tm.channel_map[cm],)
            tensor[slices] = np.count_nonzero(voltage == 0)
    return tensor


TMAPS["voltage_zeros"] = TensorMap(
    "voltage_zeros",
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=voltage_zeros,
    shape=(None, 12),
    channel_map=ECG_REST_AMP_LEADS,
    time_series_limit=0,
)


def v6_zeros_validator(tm: TensorMap, tensor: np.ndarray, hd5: h5py.File):
    voltage = decompress_data(
        data_compressed=hd5["V6"][()], dtype=hd5["V6"].attrs["dtype"],
    )
    if np.count_nonzero(voltage == 0) > 10:
        raise ValueError(f"TensorMap {tm.name} has too many zeros in V6.")


def build_ecg_time_series_tensor_maps(
    needed_tensor_maps: List[str], time_series_limit: int = 1,
) -> Dict[str, TensorMap]:
    name2tensormap: Dict[str:TensorMap] = {}

    for needed_name in needed_tensor_maps:
        if needed_name.endswith("_newest"):
            base_split = "_newest"
            time_series_order = TimeSeriesOrder.NEWEST
        elif needed_name.endswith("_oldest"):
            base_split = "_oldest"
            time_series_order = TimeSeriesOrder.OLDEST
        elif needed_name.endswith("_random"):
            base_split = "_random"
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


def _ecg_str2date(d) -> datetime.date:
    return datetime.datetime.strptime(d, ECG_DATE_FORMAT).date()


def _cardiac_surgery_str2date(
    input_date: str, date_format: str = CARDIAC_SURGERY_DATE_FORMAT,
) -> datetime.datetime:
    return datetime.datetime.strptime(input_date, date_format)


def _outcome_channels(outcome: str):
    return {f"no_{outcome}": 0, f"{outcome}": 1}


def build_ecg_tensor_maps(needed_tensor_maps: List[str]) -> Dict[str, TensorMap]:
    name2tensormap: Dict[str, TensorMap] = {}
    diagnosis2column = {
        "atrial_fibrillation": "first_af",
        "blood_pressure_medication": "first_bpmed",
        "coronary_artery_disease": "first_cad",
        "cardiovascular_disease": "first_cvd",
        "death": "death_date",
        "diabetes_mellitus": "first_dm",
        "heart_failure": "first_hf",
        "hypertension": "first_htn",
        "left_ventricular_hypertrophy": "first_lvh",
        "myocardial_infarction": "first_mi",
        "pulmonary_artery_disease": "first_pad",
        "stroke": "first_stroke",
        "valvular_disease": "first_valvular_disease",
    }
    logging.info(f"needed name {needed_tensor_maps}")
    for diagnosis in diagnosis2column:
        # Build diagnosis classification TensorMaps
        name = f"diagnosis_{diagnosis}"
        if name in needed_tensor_maps:
            tensor_from_file_fxn = build_incidence_tensor_from_file(
                INCIDENCE_CSV, diagnosis_column=diagnosis2column[diagnosis],
            )
            name2tensormap[name] = TensorMap(
                f"{name}_newest",
                Interpretation.CATEGORICAL,
                path_prefix=ECG_PREFIX,
                channel_map=_diagnosis_channels(diagnosis),
                tensor_from_file=tensor_from_file_fxn,
            )
        name = f"incident_diagnosis_{diagnosis}"
        if name in needed_tensor_maps:
            tensor_from_file_fxn = build_incidence_tensor_from_file(
                INCIDENCE_CSV,
                diagnosis_column=diagnosis2column[diagnosis],
                incidence_only=True,
            )
            name2tensormap[name] = TensorMap(
                f"{name}_newest",
                Interpretation.CATEGORICAL,
                path_prefix=ECG_PREFIX,
                channel_map=_diagnosis_channels(diagnosis, incidence_only=True),
                tensor_from_file=tensor_from_file_fxn,
            )

        # Build time to event TensorMaps
        name = f"cox_{diagnosis}"
        if name in needed_tensor_maps:
            tff = loyalty_time_to_event(
                INCIDENCE_CSV, diagnosis_column=diagnosis2column[diagnosis],
            )
            name2tensormap[name] = TensorMap(
                f"{name}_newest",
                Interpretation.TIME_TO_EVENT,
                path_prefix=ECG_PREFIX,
                tensor_from_file=tff,
            )
        name = f"incident_cox_{diagnosis}"
        if name in needed_tensor_maps:
            tff = loyalty_time_to_event(
                INCIDENCE_CSV,
                diagnosis_column=diagnosis2column[diagnosis],
                incidence_only=True,
            )
            name2tensormap[name] = TensorMap(
                f"{name}_newest",
                Interpretation.TIME_TO_EVENT,
                path_prefix=ECG_PREFIX,
                tensor_from_file=tff,
            )

        # Build survival curve TensorMaps
        for needed_name in needed_tensor_maps:
            if "survival" not in needed_name:
                continue
            potential_day_string = needed_name.split("_")[-1]
            try:
                days_window = int(potential_day_string)
            except ValueError:
                days_window = 1825  # Default to 5 years of follow up
            name = f"survival_{diagnosis}"
            if name in needed_name:
                tff = _survival_from_file(
                    days_window,
                    INCIDENCE_CSV,
                    diagnosis_column=diagnosis2column[diagnosis],
                )
                name2tensormap[needed_name] = TensorMap(
                    f"{needed_name}_newest",
                    Interpretation.SURVIVAL_CURVE,
                    path_prefix=ECG_PREFIX,
                    shape=(50,),
                    days_window=days_window,
                    tensor_from_file=tff,
                )
            name = f"incident_survival_{diagnosis}"
            if name in needed_name:
                tff = _survival_from_file(
                    days_window,
                    INCIDENCE_CSV,
                    diagnosis_column=diagnosis2column[diagnosis],
                    incidence_only=True,
                )
                name2tensormap[needed_name] = TensorMap(
                    f"{needed_name}_newest",
                    Interpretation.SURVIVAL_CURVE,
                    path_prefix=ECG_PREFIX,
                    shape=(50,),
                    days_window=days_window,
                    tensor_from_file=tff,
                )
    logging.info(f"return names {list(name2tensormap.keys())}")
    return name2tensormap


def build_cardiac_surgery_dict(
    filename: str = CARDIAC_SURGERY_OUTCOMES_CSV,
    patient_column: str = "medrecn",
    date_column: str = "surgdt",
    additional_columns: List[str] = [],
) -> Dict[int, Dict[str, Union[int, str]]]:
    keys = [date_column] + additional_columns
    cardiac_surgery_dict = {}
    df = pd.read_csv(
        filename, low_memory=False, usecols=[patient_column] + keys,
    ).sort_values(by=[patient_column, date_column])
    # sort dataframe such that newest surgery per patient appears later and is used in lookup table
    for row in df.itertuples():
        patient_key = getattr(row, patient_column)
        cardiac_surgery_dict[patient_key] = {key: getattr(row, key) for key in keys}
    return cardiac_surgery_dict


def build_date_interval_lookup(
    cardiac_surgery_dict: Dict[int, Dict[str, Union[int, str]]],
    start_column: str = "surgdt",
    start_offset: int = -30,
    end_column: str = "surgdt",
    end_offset: int = 0,
) -> Dict[int, Tuple[str, str]]:
    date_interval_lookup = {}
    for mrn in cardiac_surgery_dict:
        start_date = (
            _cardiac_surgery_str2date(
                cardiac_surgery_dict[mrn][start_column],
                ECG_DATETIME_FORMAT.replace("T", " "),
            )
            + datetime.timedelta(days=start_offset)
        ).strftime(ECG_DATETIME_FORMAT)
        end_date = (
            _cardiac_surgery_str2date(
                cardiac_surgery_dict[mrn][end_column],
                ECG_DATETIME_FORMAT.replace("T", " "),
            )
            + datetime.timedelta(days=end_offset)
        ).strftime(ECG_DATETIME_FORMAT)
        date_interval_lookup[mrn] = (start_date, end_date)
    return date_interval_lookup


def make_cardiac_surgery_outcome_tensor_from_file(
    cardiac_surgery_dict: Dict[int, Dict[str, Union[int, str]]], outcome_column: str,
) -> Callable:
    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}):
        mrn = _hd5_filename_to_mrn_int(hd5.filename)
        tensor = np.zeros(tm.shape, dtype=np.float32)
        outcome = cardiac_surgery_dict[mrn][outcome_column]

        if type(outcome) is float and not outcome.is_integer():
            raise ValueError(
                f"Cardiac Surgery categorical outcome {tm.name} ({outcome_column}) got"
                f" non-discrete value: {outcome}",
            )

        # ensure binary outcome
        if outcome != 0 and outcome != 1:
            raise ValueError(
                f"Cardiac Surgery categorical outcome {tm.name} ({outcome_column}) got"
                f" non-binary value: {outcome}",
            )

        tensor[outcome] = 1
        return tensor

    return tensor_from_file


def build_cardiac_surgery_tensor_maps(
    needed_tensor_maps: List[str],
) -> Dict[str, TensorMap]:
    """
    Create tmaps for the Society of Thoracic Surgeons (STS) project.
    Tensor Maps returned by this function fall into two categories:
     1. cardiac surgery outcomes defined by the outcome tmaps defined below.
        The outcome tmaps can be loaded to use a weighted loss function
        by appending '_weighted_loss_x' to the end of the outcome tmap name
        where the weight of the negative outcome (e.g. no_death) remains 1.0
        and the weight of the positive outcome (e.g. death) = float(x).
        examples of valid tmaps in category 1:
            'sts_death'
            'sts_death_weighted_loss_2'
            'sts_death_weighted_loss_25.7'
     2. ecg tensors defined by any of the tmaps defined at top level in this file,
        including dynamically generated time series tmaps like '_newest', '_oldest', '_random' tmaps.
        The ecg tmaps are modified to only load tensors within a given time window
        defined relative to the surgery date. The modification is made by appending
        '_sts' to the end of the ecg tensor name.
        examples of valid tmaps in category 2:
            'ecg_2500_std_newest_sts'
            'ecg_datetime_oldest_sts'
            'ecg_age_sts'

    :param needed_tensor_maps: A list of tmap names to try to create tensor map objects for
    :return: A dictionary of tmap names to tensor map objects
    """
    name2tensormap: Dict[str, TensorMap] = {}
    if not any("sts" in needed_name for needed_name in needed_tensor_maps):
        return name2tensormap

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

    needed_outcome_columns = {}
    for needed_name in needed_tensor_maps:
        for outcome, column in outcome2column.items():
            if outcome in needed_name:
                needed_outcome_columns[needed_name] = column
    cardiac_surgery_dict = build_cardiac_surgery_dict(
        additional_columns=list(needed_outcome_columns.values()),
    )
    date_interval_lookup = build_date_interval_lookup(cardiac_surgery_dict)

    for needed_name in needed_tensor_maps:
        if needed_name in needed_outcome_columns:
            clean_name = needed_name.replace(".", "_")
            channel_map = _outcome_channels(clean_name)
            loss_function = None
            if "_weighted_loss_" in needed_name:
                positive_outcome_weight = float(needed_name.split("_weighted_loss_")[1])
                loss_function = weighted_crossentropy(
                    [1.0, positive_outcome_weight], clean_name,
                )
            sts_tmap = TensorMap(
                clean_name,
                Interpretation.CATEGORICAL,
                path_prefix=ECG_PREFIX,
                tensor_from_file=make_cardiac_surgery_outcome_tensor_from_file(
                    cardiac_surgery_dict, needed_outcome_columns[needed_name],
                ),
                channel_map=channel_map,
                validator=validator_not_all_zero,
                loss=loss_function,
            )
        else:
            if not needed_name.endswith("_sts"):
                continue

            base_name = needed_name.split("_sts")[0]
            if base_name not in TMAPS:
                TMAPS.update(build_ecg_time_series_tensor_maps([base_name]))
                if base_name not in TMAPS:
                    continue

            sts_tmap = copy.deepcopy(TMAPS[base_name])
            sts_tmap.name = needed_name
            sts_tmap.time_series_lookup = date_interval_lookup

        name2tensormap[needed_name] = sts_tmap

    return name2tensormap
