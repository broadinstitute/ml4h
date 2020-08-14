# Imports: standard library
import os
import re
import copy
import logging
import datetime
from typing import Dict, List, Tuple, Union, Callable, Optional

# Imports: third party
import h5py
import numpy as np
import pandas as pd

# Imports: first party
from ml4cvd.metrics import weighted_crossentropy
from ml4cvd.TensorMap import (
    TensorMap,
    Interpretation,
    TimeSeriesOrder,
    decompress_data,
    id_from_filename,
)
from ml4cvd.normalizer import Standardize
from ml4cvd.validators import (
    RangeValidator,
    validator_no_empty,
    validator_clean_mrn,
    validator_no_negative,
    validator_not_all_zero,
    validator_voltage_no_zero_padding,
)
from ml4cvd.definitions import (
    STOP_CHAR,
    YEAR_DAYS,
    ECG_PREFIX,
    ECG_DATE_FORMAT,
    ECG_REST_AMP_LEADS,
    ECG_DATETIME_FORMAT,
    ECG_REST_INDEPENDENT_LEADS,
)

tmaps: Dict[str, TensorMap] = {}


def _get_ecg_dates(tm, hd5):
    # preserve state across tensor maps
    # keyed by the mrn, time series order, tmap shape
    if not hasattr(_get_ecg_dates, "mrn_lookup"):
        _get_ecg_dates.mrn_lookup = dict()
    mrn = id_from_filename(hd5.filename)
    if (mrn, tm.time_series_order, tm.shape) in _get_ecg_dates.mrn_lookup:
        return _get_ecg_dates.mrn_lookup[(mrn, tm.time_series_order, tm.shape)]

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
    _get_ecg_dates.mrn_lookup[(mrn, tm.time_series_order, tm.shape)] = dates
    return dates


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
            x_interp = np.linspace(0, len(voltage), desired_samples)
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


name2augmentations = {
    "crop": _crop_ecg,
    "noise": _noise_ecg,
    "warp": _warp_ecg,
}


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


tmaps["ecg_voltage_stats"] = TensorMap(
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


tmaps["voltage_len"] = TensorMap(
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
                        if string.lower() not in read.lower():
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


tmap_name = "ecg_datetime"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
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
    def tensor_from_file(tm, hd5, dependents={}):
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

    return tensor_from_file


for lead in ECG_REST_AMP_LEADS:
    tmap_name = f"lead_{lead}_len"
    tmaps[tmap_name] = TensorMap(
        name=tmap_name,
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
                np.zeros(shape, dtype=float)
                if fill == 0
                else np.full(shape, fill, dtype=float)
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


tmap_name = "ecg_read_md"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="read_md_clean"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


tmap_name = "ecg_read_pc"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="read_pc_clean"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


tmap_name = "ecg_patientid"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="patientid"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


tmap_name = "ecg_patientid_clean"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="patientid_clean"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_clean_mrn,
)


tmap_name = "ecg_firstname"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="patientfirstname"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


tmap_name = "ecg_lastname"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="patientlastname"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


tmap_name = "ecg_sex"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="gender"),
    channel_map={"female": 0, "male": 1},
    time_series_limit=0,
    validator=validator_not_all_zero,
)

tmap_name = "ecg_date"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="acquisitiondate"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


tmap_name = "ecg_time"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="acquisitiontime"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


tmap_name = "ecg_sitename"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="sitename"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


tmap_name = "ecg_location"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.LANGUAGE,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="location"),
    shape=(None, 1),
    time_series_limit=0,
    validator=validator_no_empty,
)


tmap_name = "ecg_dob"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
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
tmap_name = "ecg_sampling_frequency"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_sampling_frequency_from_file(),
    channel_map={"_250": 0, "_500": 1, "other": 2},
    time_series_limit=0,
    validator=validator_not_all_zero,
)


tmap_name = "ecg_sampling_frequency_pc"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="ecgsamplebase_pc", channel_prefix="_"),
    channel_map={"_0": 0, "_250": 1, "_500": 2, "other": 3},
    time_series_limit=0,
    validator=validator_not_all_zero,
)


tmap_name = "ecg_sampling_frequency_md"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="ecgsamplebase_md", channel_prefix="_"),
    channel_map={"_0": 0, "_250": 1, "_500": 2, "other": 3},
    time_series_limit=0,
    validator=validator_not_all_zero,
)


tmap_name = "ecg_sampling_frequency_lead"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="waveform_samplebase", channel_prefix="_"),
    channel_map={"_0": 0, "_240": 1, "_250": 2, "_500": 3, "other": 4},
    time_series_limit=0,
    validator=validator_not_all_zero,
)


tmap_name = "ecg_sampling_frequency_continuous"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_sampling_frequency_from_file(),
    time_series_limit=0,
    shape=(None, 1),
    validator=validator_no_negative,
)


tmap_name = "ecg_sampling_frequency_pc_continuous"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="ecgsamplebase_pc", fill=-1),
    time_series_limit=0,
    shape=(None, 1),
    validator=validator_no_negative,
)


tmap_name = "ecg_sampling_frequency_md_continuous"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="ecgsamplebase_md", fill=-1),
    time_series_limit=0,
    shape=(None, 1),
    validator=validator_no_negative,
)


tmap_name = "ecg_sampling_frequency_lead_continuous"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="waveform_samplebase", fill=-1),
    time_series_limit=0,
    shape=(None, 1),
    validator=validator_no_negative,
)


tmap_name = "ecg_time_resolution"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(
        key="intervalmeasurementtimeresolution", channel_prefix="_",
    ),
    channel_map={"_25": 0, "_50": 1, "_100": 2, "other": 3},
    time_series_limit=0,
    validator=validator_not_all_zero,
)


tmap_name = "ecg_amplitude_resolution"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(
        key="intervalmeasurementamplituderesolution", channel_prefix="_",
    ),
    channel_map={"_10": 0, "_20": 1, "_40": 2, "other": 3},
    time_series_limit=0,
    validator=validator_not_all_zero,
)


tmap_name = "ecg_measurement_filter"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CATEGORICAL,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(
        key="intervalmeasurementfilter", channel_prefix="_",
    ),
    time_series_limit=0,
    channel_map={"_None": 0, "_40": 1, "_80": 2, "other": 3},
    validator=validator_not_all_zero,
)


tmap_name = "ecg_high_pass_filter"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="waveform_highpassfilter", fill=-1),
    time_series_limit=0,
    shape=(None, 1),
    validator=validator_no_negative,
)


tmap_name = "ecg_low_pass_filter"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=make_ecg_tensor(key="waveform_lowpassfilter", fill=-1),
    time_series_limit=0,
    shape=(None, 1),
    validator=validator_no_negative,
)


tmap_name = "ecg_ac_filter"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
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
#     ecg_rate_std_md
#     ecg_rate_pc
#     ecg_rate_std_pc

# fmt: off
# TMap name      ->      (hd5 key,          fill, validator,                       normalization)
interval_key_map = {
    "ecg_rate":          ("ventricularrate", 0,   RangeValidator(10, 200),   None),
    "ecg_rate_std":      ("ventricularrate", 0,   RangeValidator(10, 200),   Standardize(mean=70, std=16)),
    "ecg_pr":            ("printerval",      0,   RangeValidator(50, 500),   None),
    "ecg_pr_std":        ("printerval",      0,   RangeValidator(50, 500),   Standardize(mean=175, std=36)),
    "ecg_qrs":           ("qrsduration",     0,   RangeValidator(20, 400),   None),
    "ecg_qrs_std":       ("qrsduration",     0,   RangeValidator(20, 400),   Standardize(mean=104, std=26)),
    "ecg_qt":            ("qtinterval",      0,   RangeValidator(100, 800),  None),
    "ecg_qt_std":        ("qtinterval",      0,   RangeValidator(100, 800),  Standardize(mean=411, std=45)),
    "ecg_qtc":           ("qtcorrected",     0,   RangeValidator(100, 800),  None),
    "ecg_qtc_std":       ("qtcorrected",     0,   RangeValidator(100, 800),  Standardize(mean=440, std=39)),
    "ecg_paxis":         ("paxis",           999, RangeValidator(-90, 360),  None),
    "ecg_paxis_std":     ("paxis",           999, RangeValidator(-90, 360),  Standardize(mean=47, std=30)),
    "ecg_raxis":         ("raxis",           999, RangeValidator(-90, 360),  None),
    "ecg_raxis_std":     ("raxis",           999, RangeValidator(-90, 360),  Standardize(mean=18, std=53)),
    "ecg_taxis":         ("taxis",           999, RangeValidator(-90, 360),  None),
    "ecg_taxis_std":     ("taxis",           999, RangeValidator(-90, 360),  Standardize(mean=58, std=63)),
    "ecg_qrs_count":     ("qrscount",        -1,  RangeValidator(0, 100),    None),
    "ecg_qrs_count_std": ("qrscount",        -1,  RangeValidator(0, 100),    Standardize(mean=12, std=3)),
    "ecg_qonset":        ("qonset",          -1,  RangeValidator(0, 500),    None),
    "ecg_qonset_std":    ("qonset",          -1,  RangeValidator(0, 500),    Standardize(mean=204, std=36)),
    "ecg_qoffset":       ("qoffset",         -1,  RangeValidator(0, 500),    None),
    "ecg_qoffset_std":   ("qoffset",         -1,  RangeValidator(0, 500),    Standardize(mean=252, std=44)),
    "ecg_ponset":        ("ponset",          -1,  RangeValidator(0, 1000),   None),
    "ecg_ponset_std":    ("ponset",          -1,  RangeValidator(0, 1000),   Standardize(mean=122, std=27)),
    "ecg_poffset":       ("poffset",         -1,  RangeValidator(10, 500),   None),
    "ecg_poffset_std":   ("poffset",         -1,  RangeValidator(10, 500),   Standardize(mean=170, std=42)),
    "ecg_toffset":       ("toffset",         -1,  RangeValidator(0, 1000),   None),
    "ecg_toffset_std":   ("toffset",         -1,  RangeValidator(0, 1000),   Standardize(mean=397, std=73)),
}
# fmt: on

for interval, (key, fill, validator, normalization) in interval_key_map.items():
    for suffix in ["_md", "_pc"]:
        name = f"{interval}{suffix}"
        _key = f"{key}{suffix}"
        tmaps[name] = TensorMap(
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


tmap_name = "ecg_weight_lbs"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    loss="logcosh",
    tensor_from_file=make_ecg_tensor(key="weightlbs"),
    shape=(None, 1),
    time_series_limit=0,
    validator=RangeValidator(100, 800),
)


def get_ecg_age_from_hd5(tm, hd5, dependents={}):
    ecg_dates = _get_ecg_dates(tm, hd5)
    dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
    tensor = np.full(shape, fill_value=-1, dtype=float)
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


tmap_name = "ecg_age"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    path_prefix=ECG_PREFIX,
    loss="logcosh",
    tensor_from_file=get_ecg_age_from_hd5,
    shape=(None, 1),
    time_series_limit=0,
    validator=RangeValidator(0, 120),
)

tmap_name = "ecg_age_std"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    path_prefix=ECG_PREFIX,
    loss="logcosh",
    tensor_from_file=get_ecg_age_from_hd5,
    shape=(None, 1),
    time_series_limit=0,
    validator=RangeValidator(0, 120),
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


tmaps["ecg_acquisition_year"] = TensorMap(
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


tmaps["ecg_bmi"] = TensorMap(
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
tmaps["ecg_race"] = TensorMap(
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


tmaps["ecg_adult_sex"] = TensorMap(
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


tmaps["voltage_zeros"] = TensorMap(
    "voltage_zeros",
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=voltage_zeros,
    shape=(None, 12),
    channel_map=ECG_REST_AMP_LEADS,
    time_series_limit=0,
)


def _ecg_str2date(d) -> datetime.date:
    return datetime.datetime.strptime(d, ECG_DATE_FORMAT).date()
