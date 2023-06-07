import os
import copy
import logging
import datetime
from itertools import product
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Union

import csv
import h5py
import numpy as np
import pandas as pd

from ml4h.defines import CARDIAC_SURGERY_DATE_FORMAT
from ml4h.normalizer import Standardize, ZeroMeanStd1
from ml4h.defines import ECG_REST_AMP_LEADS, ECG_REST_UKB_LEADS
from ml4h.defines import PARTNERS_DATE_FORMAT, PARTNERS_DATETIME_FORMAT
from ml4h.TensorMap import TensorMap, str2date, Interpretation, decompress_data
from ml4h.tensormap.mgb.ecg import _get_ecg_dates, _is_dynamic_shape, _make_hd5_path, make_voltage
from ml4h.tensormap.mgb.ecg import validator_not_all_zero, _hd5_filename_to_mrn_int, _resample_voltage

YEAR_DAYS = 365.26
INCIDENCE_CSV = '/media/erisone_snf13/lc_outcomes.csv'
CARDIAC_SURGERY_OUTCOMES_CSV = '/data/sts-data/mgh-preop-ecg-outcome-labels.csv'
PARTNERS_PREFIX = 'partners_ecg_rest'
WIDE_FILE = '/home/sam/ml/hf-wide-2020-09-15-with-lvh-and-lbbb.tsv'
#WIDE_FILE = '/home/sam/ml/mgh-wide-2020-06-25-with-mrn.tsv'


def make_mgb_dynamic_tensor_maps(desired_map_name: str) -> TensorMap:
    tensor_map_maker_fxns = [
        make_waveform_maps, make_partners_diagnosis_maps, make_waveform_maps_for_ukb, make_waveform_maps_lead_I,
    ]
    for map_maker_function in tensor_map_maker_fxns:
        desired_map = map_maker_function(desired_map_name)
        if desired_map is not None:
            return desired_map


def make_waveform_maps(desired_map_name: str) -> TensorMap:
    """Creates 12 possible Tensor Maps and returns the desired one or None:

        mgb_ecg_2500_std    mgb_ecg_2500_std_exact   mgb_ecg_5000_std    mgb_ecg_5000_std_exact
        mgb_ecg_2500_mv     mgb_ecg_2500_mv_exact    mgb_ecg_5000_mv     mgb_ecg_5000_mv_exact
        mgb_ecg_2500_raw    mgb_ecg_2500_raw_exact   mgb_ecg_5000_raw    mgb_ecg_5000_raw_exact

        _std normalizes with ZeroMeanStd1 and resamples
        _mv normalizes with Standardize mean = 0, std = 1000
        _raw does not normalize
        _exact does not resample
    :param desired_map_name: The name of the TensorMap and
    :return: The desired TensorMap
    """
    length_options = [2500, 5000]
    exact_options = [True, False]
    normalize_options = [ZeroMeanStd1(), Standardize(mean=0, std=1000), None]
    for length, exact_length, normalization in product(length_options, exact_options, normalize_options):
        norm = '_std' if isinstance(normalization, ZeroMeanStd1) else '_mv' if isinstance(normalization, Standardize) else '_raw'
        exact = '_exact' if exact_length else ''
        name = f'ecg_{length}{norm}{exact}'
        #name = f'ecg_{length}_std'
        if name == desired_map_name:
            return TensorMap(
                name,
                shape=(length, 12),
                path_prefix=PARTNERS_PREFIX,
                tensor_from_file=make_voltage(exact_length),
                normalization=normalization,
                channel_map=ECG_REST_AMP_LEADS,
            )


def make_waveform_maps_lead_I(desired_map_name: str) -> TensorMap:
    """Creates 12 possible Tensor Maps and returns the desired one or None:
    :param desired_map_name: The name of the TensorMap and
    :return: The desired TensorMap
    """
    length_options = [2500, 5000]
    exact_options = [True, False]
    normalize_options = [ZeroMeanStd1(), Standardize(mean=0, std=1000), None]
    for length, exact_length, normalization in product(length_options, exact_options, normalize_options):
        norm = '_std' if isinstance(normalization, ZeroMeanStd1) else '_mv' if isinstance(normalization, Standardize) else '_raw'
        exact = '_exact' if exact_length else ''
        name = f'ecg_lead_I_{length}{norm}{exact}'
        if name == desired_map_name:
            return TensorMap(
                name,
                shape=(length, 1),
                path_prefix=PARTNERS_PREFIX,
                tensor_from_file=make_voltage(exact_length),
                normalization=normalization,
                channel_map={'I': 0},
            )


def _dummy_tensor_from_file(tm, hd5, dependents={}):
    return np.zeros(tm.shape, dtype=np.float32)

def make_waveform_maps_for_ukb(desired_map_name: str) -> TensorMap:
    """Creates Tensor Maps and returns the desired one or None
    :param desired_map_name: The name of the TensorMap and
    :return: The desired TensorMap
    """
    if 'strip' == desired_map_name:
        return TensorMap(
            desired_map_name,
            shape=(5000, 12),
            path_prefix=PARTNERS_PREFIX,
            tensor_from_file=make_voltage(False),
            normalization=ZeroMeanStd1(), #normalization,
            channel_map=ECG_REST_AMP_LEADS,
        )
    elif 'ecg_rest_median_raw_10' == desired_map_name:
        return TensorMap(
            desired_map_name,
            shape=(600, 12),
            tensor_from_file=_dummy_tensor_from_file,
            channel_map=ECG_REST_AMP_LEADS,
        )

def make_lead_maps(desired_map_name: str) -> TensorMap:
    for lead in ECG_REST_AMP_LEADS:
        tensormap_name = f'lead_{lead}_len'
        if desired_map_name == tensormap_name:
            return TensorMap(
                tensormap_name, interpretation=Interpretation.CATEGORICAL, path_prefix=PARTNERS_PREFIX,
                channel_map={'_2500': 0, '_5000': 1, 'other': 2}, time_series_limit=0, validator=validator_not_all_zero,
                tensor_from_file=make_voltage_len_categorical_tmap(lead=lead),
            )


def make_voltage_len_categorical_tmap(lead, channel_prefix='_', channel_unknown='other'):
    def _tensor_from_file(tm, hd5, dependents={}):
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


# Date formatting
def _partners_str2date(d) -> datetime.date:
    return datetime.datetime.strptime(d, PARTNERS_DATE_FORMAT).date()


def _loyalty_str2date(date_string: str) -> datetime.date:
    return str2date(date_string.split(' ')[0])


def _cardiac_surgery_str2date(input_date: str, date_format: str = CARDIAC_SURGERY_DATE_FORMAT) -> datetime.datetime:
    return datetime.datetime.strptime(input_date, date_format)


def build_incidence_tensor_from_file(
    file_name: str, patient_column: str = 'Mrn', birth_column: str = 'birth_date',
    diagnosis_column: str = 'first_stroke', start_column: str = 'start_fu',
    delimiter: str = ',', incidence_only: bool = False, check_birthday: bool = True,
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

        ecg_dates = _get_ecg_dates(tm, hd5)
        if len(ecg_dates) > 1:
            raise NotImplementedError('Diagnosis models for multiple ECGs are not implemented.')
        dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
        categorical_data = np.zeros(shape, dtype=np.float32)
        for i, ecg_date in enumerate(ecg_dates):
            path = lambda key: _make_hd5_path(tm, ecg_date, key)
            mrn = _hd5_filename_to_mrn_int(hd5.filename)
            mrn_int = int(mrn)

            if mrn_int not in patient_table:
                raise KeyError(f'{tm.name} mrn not in incidence csv')

            if check_birthday:
                birth_date = _partners_str2date(decompress_data(data_compressed=hd5[path('dateofbirth')][()], dtype=hd5[path('dateofbirth')].attrs['dtype']))
                if birth_date != birth_table[mrn_int]:
                    raise ValueError(f'Birth dates do not match! CSV had {birth_table[patient_key]} but HD5 has {birth_date}')

            assess_date = _partners_str2date(decompress_data(data_compressed=hd5[path('acquisitiondate')][()], dtype=hd5[path('acquisitiondate')].attrs['dtype']))
            if assess_date < patient_table[mrn_int]:
                raise ValueError(f'{tm.name} Assessed earlier than enrollment')
            if mrn_int not in date_table:
                index = 0
            else:
                disease_date = date_table[mrn_int]

                if incidence_only and disease_date < assess_date:
                    raise ValueError(f'{tm.name} is skipping prevalent cases.')
                elif incidence_only and disease_date >= assess_date:
                    index = 1
                else:
                    index = 1 if disease_date < assess_date else 2
                logging.debug(f'mrn: {mrn_int}  Got disease_date: {disease_date} assess  {assess_date} index  {index}.')
            slices = (i, index) if dynamic else (index,)
            categorical_data[slices] = 1.0
        return categorical_data
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

        ecg_dates = _get_ecg_dates(tm, hd5)
        if len(ecg_dates) > 1:
            raise NotImplementedError('Cox hazard models for multiple ECGs are not implemented.')
        dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
        tensor = np.zeros(tm.shape, dtype=np.float32)
        for i, ecg_date in enumerate(ecg_dates):
            patient_key_from_ecg = _hd5_filename_to_mrn_int(hd5.filename)
            if patient_key_from_ecg not in disease_dicts['follow_up_start']:
                raise KeyError(f'{tm.name} mrn not in incidence csv')

            path = _make_hd5_path(tm, ecg_date, 'acquisitiondate')
            assess_date = _partners_str2date(decompress_data(data_compressed=hd5[path][()], dtype=hd5[path].attrs['dtype']))
            if assess_date < disease_dicts['follow_up_start'][patient_key_from_ecg]:
                raise ValueError(f'Assessed earlier than enrollment.')

            if patient_key_from_ecg not in disease_dicts['diagnosis_dates']:
                has_disease = 0
                censor_date = disease_dicts['follow_up_start'][patient_key_from_ecg] + datetime.timedelta(
                    days=YEAR_DAYS * disease_dicts['follow_up_total'][patient_key_from_ecg],
                )
            else:
                has_disease = 1
                censor_date = disease_dicts['diagnosis_dates'][patient_key_from_ecg]

            if incidence_only and censor_date <= assess_date and has_disease:
                raise ValueError(f'{tm.name} only considers incident diagnoses')

            tensor[(i, 0) if dynamic else 0] = has_disease
            tensor[(i, 1) if dynamic else 1] = (censor_date - assess_date).days
        return tensor
    return _cox_tensor_from_file


def _survival_from_file(
    day_window: int, file_name: str, incidence_only: bool = False, patient_column: str = 'Mrn',
    follow_up_start_column: str = 'start_fu', follow_up_total_column: str = 'total_fu',
    diagnosis_column: str = 'first_stroke', delimiter: str = ',',
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

        ecg_dates = _get_ecg_dates(tm, hd5)
        if len(ecg_dates) > 1:
            raise NotImplementedError('Survival curve models for multiple ECGs are not implemented.')
        dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
        survival_then_censor = np.zeros(shape, dtype=np.float32)
        for ed, ecg_date in enumerate(ecg_dates):
            patient_key_from_ecg = _hd5_filename_to_mrn_int(hd5.filename)
            if patient_key_from_ecg not in disease_dicts['follow_up_start']:
                raise KeyError(f'{tm.name} mrn not in incidence csv')

            path = _make_hd5_path(tm, ecg_date, 'acquisitiondate')
            assess_date = _partners_str2date(decompress_data(data_compressed=hd5[path][()], dtype=hd5[path].attrs['dtype']))
            if assess_date < disease_dicts['follow_up_start'][patient_key_from_ecg]:
                raise ValueError(f'Assessed earlier than enrollment.')

            if patient_key_from_ecg not in disease_dicts['diagnosis_dates']:
                has_disease = 0
                censor_date = disease_dicts['follow_up_start'][patient_key_from_ecg] + datetime.timedelta(days=YEAR_DAYS*disease_dicts['follow_up_total'][patient_key_from_ecg])
            else:
                has_disease = 1
                censor_date = disease_dicts['diagnosis_dates'][patient_key_from_ecg]

            intervals = int(shape[1] if dynamic else shape[0] / 2)
            days_per_interval = day_window / intervals

            for i, day_delta in enumerate(np.arange(0, day_window, days_per_interval)):
                cur_date = assess_date + datetime.timedelta(days=day_delta)
                survival_then_censor[(ed, i) if dynamic else i] = float(cur_date < censor_date)
                survival_then_censor[(ed, intervals+i) if dynamic else intervals+i] = has_disease * float(censor_date <= cur_date < censor_date + datetime.timedelta(days=days_per_interval))
                if i == 0 and censor_date <= cur_date:  # Handle prevalent diseases
                    survival_then_censor[(ed, intervals) if dynamic else intervals] = has_disease
                    if has_disease and incidence_only:
                        raise ValueError(f'{tm.name} is skipping prevalent cases.')
            logging.debug(
                f"Got survival disease {has_disease}, censor: {censor_date}, assess {assess_date}, fu start {disease_dicts['follow_up_start'][patient_key_from_ecg]} "
                f"fu total {disease_dicts['follow_up_total'][patient_key_from_ecg]} tensor:{(survival_then_censor[ed] if dynamic else survival_then_censor)[:4]} mid tense: {(survival_then_censor[ed] if dynamic else survival_then_censor)[intervals:intervals+4]} ",
            )
        return survival_then_censor
    return tensor_from_file


def make_partners_diagnosis_maps(desired_map_name: str) -> Union[TensorMap, None]:
    diagnosis_columns = ['AF', 'CAD', 'DM', 'MI']
    if desired_map_name in diagnosis_columns:
        # Build survival curve TensorMaps
        return TensorMap(f'{desired_map_name.lower()}_event', Interpretation.SURVIVAL_CURVE, shape=(50,))


def make_wide_file_maps(desired_map_name: str) -> Union[TensorMap, None]:
    days_window = 1825

    if desired_map_name == 'sex_from_wide_csv':
        csv_tff = tensor_from_wide(WIDE_FILE, target='sex')
        return TensorMap(
            'sex_from_wide', Interpretation.CATEGORICAL, annotation_units=2, tensor_from_file=csv_tff,
            channel_map={'female': 0, 'male': 1},
        )
    elif desired_map_name == 'age_from_wide_csv':
        csv_tff = tensor_from_wide(WIDE_FILE, target='age')
        return TensorMap(
            'age_from_wide', Interpretation.CONTINUOUS, shape=(1,),
            tensor_from_file=csv_tff, channel_map={'age': 0},
            normalization={'mean': 63.35798891483556, 'std': 7.554638350423902},
        )
    elif desired_map_name == 'bmi_from_wide_csv':
        csv_tff = csv_tff = tensor_from_wide(WIDE_FILE, target='bmi')
        return TensorMap(
            'bmi_from_wide', Interpretation.CONTINUOUS, shape=(1,), channel_map={'bmi': 0},
            annotation_units=1, normalization={'mean': 27.3397, 'std': 4.77216}, tensor_from_file=csv_tff,
        )
    elif desired_map_name == 'ecg_2500_from_wide_csv':
        tff = tensor_from_wide(WIDE_FILE, target='ecg')
        return TensorMap(
            'ecg_rest_raw', shape=(2500, 12), path_prefix=PARTNERS_PREFIX, tensor_from_file=tff,
            cacheable=False, channel_map=ECG_REST_UKB_LEADS,
        )
    elif desired_map_name == 'ecg_5000_from_wide_csv':
        tff = tensor_from_wide(WIDE_FILE, target='ecg')
        return TensorMap(
            'ecg_rest_raw', shape=(5000, 12), path_prefix=PARTNERS_PREFIX, tensor_from_file=tff,
            cacheable=False, channel_map=ECG_REST_UKB_LEADS,
        )
    elif desired_map_name == 'time_to_hf_wide_csv':
        tff = tensor_from_wide(WIDE_FILE, target='time_to_event')
        return TensorMap('time_to_hf', Interpretation.TIME_TO_EVENT, tensor_from_file=tff)
    elif desired_map_name == 'survival_curve_hf_wide_csv':
        tff = tensor_from_wide(WIDE_FILE, target='survival_curve')
        return TensorMap('survival_curve_hf', Interpretation.SURVIVAL_CURVE, tensor_from_file=tff, shape=(50,), days_window=days_window)


def _date_from_dates(ecg_dates, target_date=None, earliest_date=None):
    if target_date:
        if target_date and earliest_date:
            incident_dates = [d for d in ecg_dates if earliest_date < datetime.datetime.strptime(d, PARTNERS_DATETIME_FORMAT) < target_date]
        else:
            incident_dates = [d for d in ecg_dates if datetime.datetime.strptime(d, PARTNERS_DATETIME_FORMAT) < target_date]
        if len(incident_dates) == 0:
            raise ValueError('No ECGs prior to target were found.')
        return np.random.choice(incident_dates)
    return np.random.choice(ecg_dates)


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


def _to_float_or_none(s):
    try:
        return float(s)
    except ValueError:
        return None


def _days_to_years_float(s: str):
    try:
        return float(s.split(' ')[0]) / YEAR_DAYS
    except ValueError:
        return None


def _time_to_event_tensor_from_days(tm: TensorMap, has_disease: int, follow_up_days: int):
    tensor = np.zeros(tm.shape, dtype=np.float32)
    if follow_up_days > tm.days_window:
        has_disease = 0
        follow_up_days = tm.days_window + 1
    tensor[0] = has_disease
    tensor[1] = follow_up_days
    return tensor


def _survival_curve_tensor_from_dates(tm: TensorMap, has_disease: int, assessment_date: datetime.datetime, censor_date: datetime.datetime):
    intervals = tm.shape[0] // 2
    days_per_interval = tm.days_window / intervals
    survival_then_censor = np.zeros(tm.shape, dtype=np.float32)
    for i, day_delta in enumerate(np.arange(0, tm.days_window, days_per_interval)):
        cur_date = assessment_date + datetime.timedelta(days=day_delta)
        survival_then_censor[i] = float(cur_date < censor_date)
        survival_then_censor[intervals + i] = has_disease * float(censor_date <= cur_date < censor_date + datetime.timedelta(days=days_per_interval))
        if i == 0 and censor_date <= cur_date:  # Handle prevalent diseases
            survival_then_censor[intervals] = has_disease
    return survival_then_censor


def tensor_from_wide(
    file_name: str, patient_column: str = 'Mrn', age_column: str = 'age', bmi_column: str = 'bmi',
    sex_column: str = 'sex', hf_column: str = 'any_hf_age', start_column: str = 'start_fu',
    end_column: str = 'last_encounter', delimiter: str = '\t', population_normalize: int = 2000,
    target: str = 'ecg', skip_prevalent: bool = True,
) -> Callable:
    """Build a tensor_from_file function for ECGs in the legacy cohort.

    :param file_name: CSV or TSV file with header of patient IDs (MRNs) dates of enrollment and dates of diagnosis
    :param patient_column: The header name of the column of patient ids
    :param delimiter: The delimiter separating columns of the TSV or CSV
    :return: The tensor_from_file function to provide to TensorMap constructors
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader)
        hf_index = header.index(hf_column)
        age_index = header.index(age_column)
        bmi_index = header.index(bmi_column)
        sex_index = header.index(sex_column)
        end_index = header.index(end_column)
        start_index = header.index(start_column)
        patient_index = header.index(patient_column)
        patient_data = defaultdict(dict)
        for row in reader:
            try:
                patient_key = int(float(row[patient_index]))
                patient_data[patient_key] = {
                    'age': _days_to_years_float(row[age_index]),
                    'bmi': _to_float_or_none(row[bmi_index]),
                    'sex': row[sex_index],
                    'hf_age': _days_to_years_float(row[hf_index]),
                    'end_age': _days_to_years_float(row[end_index]),
                    'start_date': datetime.datetime.strptime(row[start_index], CARDIAC_SURGERY_DATE_FORMAT),
                }

            except ValueError as e:
                logging.debug(f'val err {e}')
        logging.info(f'Done processing. Got {len(patient_data)} patient rows.')

    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        mrn_int = _hd5_filename_to_mrn_int(hd5.filename)
        if mrn_int not in patient_data:
            raise KeyError(f'{tm.name} mrn not in csv.')
        if patient_data[mrn_int]['end_age'] is None or patient_data[mrn_int]['age'] is None:
            raise ValueError(f'{tm.name} could not find ages.')
        if patient_data[mrn_int]['end_age'] - patient_data[mrn_int]['age'] < 0:
            raise ValueError(f'{tm.name} has negative follow up time.')

        if patient_data[mrn_int]['hf_age'] is None:
            has_disease = 0
            follow_up_days = (patient_data[mrn_int]['end_age'] - patient_data[mrn_int]['age']) * YEAR_DAYS
        elif patient_data[mrn_int]['hf_age'] > patient_data[mrn_int]['age']:
            has_disease = 1
            follow_up_days = (patient_data[mrn_int]['hf_age'] - patient_data[mrn_int]['age']) * YEAR_DAYS
        elif skip_prevalent and patient_data[mrn_int]['age'] > patient_data[mrn_int]['hf_age']:
            raise ValueError(f'{tm.name} skips prevalent cases.')
        else:
            has_disease = 1
            follow_up_days = (patient_data[mrn_int]['hf_age'] - patient_data[mrn_int]['age']) * YEAR_DAYS

        if target == 'time_to_event':
            tensor = _time_to_event_tensor_from_days(tm, has_disease, follow_up_days)
            logging.debug(f'Returning {tensor} for {patient_data[mrn_int]} key {mrn_int}')
            return tensor
        elif target == 'survival_curve':
            end_date = patient_data[mrn_int]['start_date'] + datetime.timedelta(days=follow_up_days)
            tensor = _survival_curve_tensor_from_dates(tm, has_disease, patient_data[mrn_int]['start_date'], end_date)
            logging.debug(
                f"Got survival disease {has_disease}, censor: {end_date}, assess {patient_data[mrn_int]['start_date']}, age {patient_data[mrn_int]['age']} "
                f"end age: {patient_data[mrn_int]['end_age']} hf age: {patient_data[mrn_int]['hf_age']} "
                f"fu total {follow_up_days/YEAR_DAYS} tensor:{tensor[:4]} mid tense: {tensor[tm.shape[0] // 2:(tm.shape[0] // 2)+4]} ",
            )
            return tensor
        elif target == 'ecg':
            ecg_dates = list(hd5[tm.path_prefix])
            earliest = patient_data[mrn_int]['start_date'] - datetime.timedelta(days=3*YEAR_DAYS)
            ecg_date_key = _date_from_dates(ecg_dates, patient_data[mrn_int]['start_date'], earliest)
            return _ecg_tensor_from_date(tm, hd5, ecg_date_key, population_normalize)
        elif target in ['age', 'bmi']:
            tensor = np.zeros(tm.shape, dtype=np.float32)
            if patient_data[mrn_int][target] is None:
                raise ValueError(f'Missing target value {target}')
            tensor[0] = patient_data[mrn_int][target]
            return tensor
        elif target == 'sex':
            tensor = np.zeros(tm.shape, dtype=np.float32)
            if patient_data[mrn_int][target].lower() == 'female':
                tensor[0] = 1.0
            elif patient_data[mrn_int][target].lower() == 'male':
                tensor[1] = 1.0
            logging.debug(f'Returning {tensor} for {patient_data[mrn_int][target]} key {mrn_int}')
            return tensor
        else:
            raise ValueError(f'{tm.name} has no way to handle target {target}')
    return tensor_from_file


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
    needed_name: str,
) -> TensorMap:
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
    if needed_name in outcome2column:
        if cardiac_surgery_dict is None:
            cardiac_surgery_dict = build_cardiac_surgery_dict(
                additional_columns=[outcome2column[needed_name]],
            )
            channel_map = _outcome_channels(needed_name)
            sts_tmap = TensorMap(
                needed_name,
                Interpretation.CATEGORICAL,
                path_prefix=PARTNERS_PREFIX,
                tensor_from_file=make_cardiac_surgery_outcome_tensor_from_file(
                    cardiac_surgery_dict, outcome2column[needed_name],
                ),
                channel_map=channel_map,
                validator=validator_not_all_zero,
            )
        else:
            if needed_name.endswith('_sts'):
                base_name = needed_name.split('_sts')[0]
                tmap_map = build_partners_time_series_tensor_maps([base_name])

                if cardiac_surgery_dict is None:
                    cardiac_surgery_dict = build_cardiac_surgery_dict(
                        additional_columns=[outcome2column[needed_name]],
                    )
                if date_interval_lookup is None:
                    date_interval_lookup = build_date_interval_lookup(cardiac_surgery_dict)
                sts_tmap = copy.deepcopy(tmap_map[base_name])
                sts_tmap.name = needed_name
                sts_tmap.time_series_lookup = date_interval_lookup

    return sts_tmap


def _get_measurement_matrix_entry(matrix: np.ndarray, key_idx: int, lead_idx: Union[int, List[int]] = None):
    # First 18 words of measurement matrix are for global measurements, then each lead has 53*2 words
    lead_start = 18
    lead_words = 53 * 2
    # In the measurement matrix, NA values are encoded as 2^15 - 1. Introducing here a threshold
    nan_threshold = 32765
    if lead_idx is None:
        idx = key_idx
    elif isinstance(lead_idx, list):
        matrix_values = []
        for lead_index in lead_idx:
            idx = lead_start + lead_index * lead_words + (key_idx-1)*2+1
            if matrix[idx] > nan_threshold:
                matrix_values.append(np.nan)
            else:
                matrix_values.append(matrix[idx])
        return max(matrix_values)
    else:
        idx = lead_start + lead_idx * lead_words + (key_idx-1)*2+1
    value = np.nan if matrix[idx] > nan_threshold else matrix[idx]
    return value


def make_measurement_matrix_from_file(key_idx: int, lead_idx: Union[int, List[int]] = None):
    def measurement_matrix_from_file(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}):
        ecg_dates = _get_ecg_dates(tm, hd5)
        dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
        tensor = np.zeros(shape, dtype=float)
        for i, ecg_date in enumerate(ecg_dates):
            path = _make_hd5_path(tm, ecg_date, 'measurementmatrix')
            matrix = decompress_data(data_compressed=hd5[path][()], dtype=hd5[path].attrs['dtype'])
            tensor[i] = _get_measurement_matrix_entry(matrix, key_idx, lead_idx)
        return tensor
    return measurement_matrix_from_file


def make_mgb_ecg_measurement_matrix_global_tensor_maps(needed_name: str):
    # Measurement matrix TMAPS -- indices from MUSE XML dev manual, page 49 and following
    measurement_matrix_global_measures = {
        'pon': 1,       # P-wave onset in median beat (in samples)
        'poff': 2,      # P-wave offset in median beat
        'qon': 3,       # Q-Onset in median beat
        'qoff': 4,      # Q-Offset in median beat
        'ton': 5,       # T-Onset in median beat
        'toff': 6,      # T-Offset in median beat
        'nqrs': 7,      # Number of QRS Complexes
        'qrsdur': 8,    # QRS Duration
        'qt': 9,        # QT Interval
        'qtc': 10,      # QT Corrected
        'print': 11,    # PR Interval
        'vrate': 12,    # Ventricular Rate
        'avgrr': 13,    # Average R-R Interval
    }
    for measure, measure_idx in measurement_matrix_global_measures.items():
        if f'partners_ecg_measurement_matrix_{measure}' == needed_name:
            return TensorMap(
                f'partners_ecg_measurement_matrix_{measure}',
                interpretation=Interpretation.CONTINUOUS,
                shape=(None, 1),
                path_prefix=PARTNERS_PREFIX,
                loss='logcosh',
                time_series_limit=0,
                tensor_from_file=make_measurement_matrix_from_file(measure_idx),
            )


measurement_matrix_leads = {
    'I': 0, 'II': 1, 'V1': 2, 'V2': 3, 'V3': 4, 'V4':5, 'V5': 6, 'V6': 7, 'III': 8, 'aVR': 9, 'aVL': 10, 'aVF': 11,
}
measurement_matrix_lead_measures = {
    'pona': 1,      # P Wave amplitude at P-onset
    'pamp': 2,      # P wave amplitude
    'pdur': 3,      # P wave duration
    'bmpar': 4,     # P wave area
    'bmpi': 5,      # P wave intrinsicoid (time from P onset to peak of P)
    'ppamp': 6,     # P Prime amplitude
    'ppdur': 7,     # P Prime duration
    'bmppar': 8,    # P Prime area
    'bmppi': 9,     # P Prime intrinsicoid (time from P onset to peak of P')
    'qamp': 10,     # Q wave amplitude
    'qdur': 11,     # Q wave duration
    'bmqar': 12,    # Q wave area
    'bmqi': 13,     # Q intrinsicoid (time from Q onset to peak of Q)
    'ramp': 14,     # R amplitude
    'rdur': 15,     # R duration
    'bmrar': 16,    # R wave area
    'bmri': 17,     # R intrinsicoid (time from R onset to peak of R)
    'samp': 18,     # S amplitude
    'sdur': 19,     # S duration
    'bmsar': 20,    # S wave area
    'bmsi': 21,     # S intrinsicoid (time from Q onset to peak of S)
    'rpamp': 22,    # R Prime amplitude
    'rpdur': 23,    # R Prime duration
    'bmrpar': 24,   # R Prime wave area
    'bmrpi': 25,    # R Prime intrinsicoid (time from Q onset to peak of R Prime)
    'spamp': 26,    # S Prime Amplitude
    'spdur': 27,    # S Prime Duration
    'bmspar': 28,   # S Prime wave area
    'bmspi': 29,    # S intriniscoid (time from Q onset to peak of S prime)
    'stj': 30,      # STJ point, End of QRS Point Amplitude
    'stm': 31,      # STM point, Middle of the ST Segment Amplitude
    'ste': 32,      # STE point, End of ST Segment Amplitude
    'mxsta': 33,    # Maximum of STJ, STM, STE Amplitudes
    'mnsta': 34,    # Minimum of STJ and STM Amplitudes
    'spta': 35,     # Special T-Wave amplitude
    'qrsa': 36,     # Total QRS area
    'qrsdef': 37,   # QRS Deflection
    'maxra': 38,    # Maximum R Amplitude (R or R Prime)
    'maxsa': 39,    # Maximum S Amplitude (S or S Prime)
    'tamp': 40,     # T amplitude
    'tdur': 41,     # T duration
    'bmtar': 42,    # T wave area
    'bmti': 43,     # T intriniscoid (time from STE to peak of T)
    'tpamp': 44,    # T Prime amplitude
    'tpdur': 45,    # T Prime duration
    'bmtpar': 46,   # T Prime area
    'bmtpi': 47,    # T Prime intriniscoid (time from STE to peak of T)
    'tend': 48,     # T Amplitude at T offset
    'parea': 49,    # P wave area, includes P and P Prime
    'qrsar': 50,    # QRS area
    'tarea': 51,    # T wave area, include T and T Prime
    'qrsint': 52,    # QRS intriniscoid (see following)
}


def make_mgb_ecg_measurement_matrix_lead_tensor_maps(needed_name: str):
    for lead, lead_idx in measurement_matrix_leads.items():
        for measure, measure_idx in measurement_matrix_lead_measures.items():
            if f'partners_ecg_measurement_matrix_{lead}_{measure}' == needed_name:
                return TensorMap(
                    f'partners_ecg_measurement_matrix_{lead}_{measure}',
                    interpretation=Interpretation.CONTINUOUS,
                    shape=(None, 1),
                    path_prefix=PARTNERS_PREFIX,
                    loss='logcosh',
                    time_series_limit=0,
                    tensor_from_file=make_measurement_matrix_from_file(measure_idx, lead_idx=lead_idx),
                )

    # Max values across leads
    for measure, measure_idx in measurement_matrix_lead_measures.items():
        if f'partners_ecg_measurement_matrix_max_{measure}' == needed_name:
            return TensorMap(
                f'partners_ecg_measurement_matrix_max_{measure}',
                interpretation=Interpretation.CONTINUOUS,
                shape=(None, 1),
                path_prefix=PARTNERS_PREFIX,
                loss='logcosh',
                time_series_limit=0,
                tensor_from_file=make_measurement_matrix_from_file(measure_idx, lead_idx=list(measurement_matrix_leads.values())),
            )


def make_mgb_ecg_lvh_tensormaps(needed_name: str):
    def ecg_lvh_from_file(tm: TensorMap, hd5: h5py.File, dependents={}):
        # Lead order seems constant and standard throughout, but we could eventually tensorize it from XML
        avl_min = 1100.0
        sl_min = 3500.0
        cornell_female_min = 2000.0
        cornell_male_min = 2800.0
        sleads = ['V1', 'V3']
        rleads = ['aVL', 'V5', 'V6']
        ecg_dates = _get_ecg_dates(tm, hd5)
        dynamic, shape = _is_dynamic_shape(tm, len(ecg_dates))
        tensor = np.zeros(shape, dtype=float)

        for i, ecg_date in enumerate(ecg_dates):
            path = _make_hd5_path(tm, ecg_date, 'measurementmatrix')
            matrix = decompress_data(data_compressed=hd5[path][()], dtype=hd5[path].attrs['dtype'])
            criteria_sleads = {lead: _get_measurement_matrix_entry(matrix, measurement_matrix_lead_measures['samp'], measurement_matrix_leads[lead]) for lead in sleads}
            criteria_rleads = {lead: _get_measurement_matrix_entry(matrix, measurement_matrix_lead_measures['ramp'], measurement_matrix_leads[lead]) for lead in rleads}
            if 'avl_lvh' in tm.name:
                is_lvh = np.nan if np.isnan(criteria_rleads['aVL']) else criteria_rleads['aVL'] > avl_min
            elif 'sokolow_lyon_lvh' in tm.name:
                criteria_sum = criteria_sleads['V1'] + np.maximum(criteria_rleads['V5'], criteria_rleads['V6'])
                is_lvh = np.nan if np.isnan(criteria_sum) else criteria_sum > sl_min
            elif 'cornell_lvh' in tm.name:
                is_lvh = criteria_rleads['aVL'] + criteria_sleads['V3']
                sex_path = _make_hd5_path(tm, ecg_date, 'gender')
                is_female = 'female' in decompress_data(data_compressed=hd5[sex_path][()], dtype='str').lower()
                if is_female:
                    is_lvh = np.nan if np.isnan(is_lvh) else is_lvh > cornell_female_min
                else:
                    is_lvh = np.nan if np.isnan(is_lvh) else is_lvh > cornell_male_min
            else:
                raise ValueError(f'{tm.name} criterion for LVH is not accounted for')
            # Following convention from categorical TMAPS, positive has cmap index 1
            if np.isnan(is_lvh):
                if dynamic:
                    tensor[i, :] = np.nan
                else:
                    tensor[:] = np.nan
            else:
                index = 1 if is_lvh else 0
                slices = (i, index) if dynamic else (index,)
                tensor[slices] = 1.0
        return tensor

    for criterion in ['avl_lvh', 'sokolow_lyon_lvh', 'cornell_lvh']:
        if f'partners_ecg_{criterion}' == needed_name:
            return TensorMap(
                f'partners_ecg_{criterion}',
                interpretation=Interpretation.CATEGORICAL,
                path_prefix=PARTNERS_PREFIX,
                tensor_from_file=ecg_lvh_from_file,
                channel_map={f'no_{criterion}': 0, criterion: 1},
                shape=(None, 2),
                time_series_limit=0,
            )
