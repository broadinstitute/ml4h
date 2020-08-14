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

#from ml4cvd.arguments import tensormap_lookup
from ml4cvd.metrics import weighted_crossentropy
from ml4cvd.normalizer import Standardize, ZeroMeanStd1
from ml4cvd.tensormap.partners.ecg import _get_ecg_dates, _is_dynamic_shape, _make_hd5_path, validator_not_all_zero, make_voltage, _hd5_filename_to_mrn_int
from ml4cvd.TensorMap import TensorMap, str2date, Interpretation, make_range_validator, decompress_data, TimeSeriesOrder
from ml4cvd.defines import ECG_REST_AMP_LEADS, PARTNERS_DATE_FORMAT, STOP_CHAR, PARTNERS_DATETIME_FORMAT, CARDIAC_SURGERY_DATE_FORMAT


YEAR_DAYS = 365.26
INCIDENCE_CSV = '/media/erisone_snf13/lc_outcomes.csv'
CARDIAC_SURGERY_OUTCOMES_CSV = '/data/sts-data/mgh-preop-ecg-outcome-labels.csv'
PARTNERS_PREFIX = 'partners_ecg_rest'


def make_partners_dynamic_tensor_maps(desired_map_name: str) -> TensorMap:
    dynamic_tensor_map_makers = [make_lead_maps, make_waveform_maps, make_partners_diagnosis_maps]
    for map_maker_function in dynamic_tensor_map_makers:
        desired_map = map_maker_function(desired_map_name)
        if desired_map is not None:
            return desired_map


# Creates 12 TMaps:
# partners_ecg_2500      partners_ecg_2500_exact      partners_ecg_5000      partners_ecg_5000_exact
# partners_ecg_2500_std  partners_ecg_2500_std_exact  partners_ecg_5000_std  partners_ecg_5000_std_exact
# partners_ecg_2500_raw  partners_ecg_2500_raw_exact  partners_ecg_5000_raw  partners_ecg_5000_raw_exact
#
# default normalizes with ZeroMeanStd1 and resamples
# _std normalizes with Standardize mean = 0, std = 2000
# _raw does not normalize
# _exact does not resample

def make_waveform_maps(desired_map_name: str):
    length_options = [2500, 5000]
    exact_options = [True, False]
    normalize_options = [ZeroMeanStd1(), Standardize(mean=0, std=2000), None]
    for length, exact_length, normalization in product(length_options, exact_options, normalize_options):
        norm = '' if isinstance(normalization, ZeroMeanStd1) else '_std' if isinstance(normalization, Standardize) else '_raw'
        exact = '_exact' if exact_length else ''
        name = f'partners_ecg_{length}{norm}{exact}'
        if name == desired_map_name:
            return TensorMap(
                name,
                shape=(None, length, 12),
                path_prefix=PARTNERS_PREFIX,
                tensor_from_file=make_voltage(exact_length),
                normalization=normalization,
                channel_map=ECG_REST_AMP_LEADS,
                time_series_limit=0,
                validator=validator_not_all_zero,
            )


def make_lead_maps(desired_map_name: str):
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
    diagnosis2column = {
        'atrial_fibrillation': 'first_af', 'blood_pressure_medication': 'first_bpmed',
        'coronary_artery_disease': 'first_cad', 'cardiovascular_disease': 'first_cvd',
        'death': 'death_date', 'diabetes_mellitus': 'first_dm', 'heart_failure': 'first_hf',
        'hypertension': 'first_htn', 'left_ventricular_hypertrophy': 'first_lvh',
        'myocardial_infarction': 'first_mi', 'pulmonary_artery_disease': 'first_pad',
        'stroke': 'first_stroke', 'valvular_disease': 'first_valvular_disease',
    }
    for diagnosis in diagnosis2column:
        # Build diagnosis classification TensorMaps
        name = f'diagnosis_{diagnosis}'
        if name == desired_map_name:
            tensor_from_file_fxn = build_incidence_tensor_from_file(INCIDENCE_CSV, diagnosis_column=diagnosis2column[diagnosis])
            return TensorMap(f'{name}_newest', Interpretation.CATEGORICAL, path_prefix=PARTNERS_PREFIX, channel_map=_diagnosis_channels(diagnosis), tensor_from_file=tensor_from_file_fxn)
        name = f'incident_diagnosis_{diagnosis}'
        if name == desired_map_name:
            tensor_from_file_fxn = build_incidence_tensor_from_file(INCIDENCE_CSV, diagnosis_column=diagnosis2column[diagnosis], incidence_only=True)
            return TensorMap(f'{name}_newest', Interpretation.CATEGORICAL, path_prefix=PARTNERS_PREFIX, channel_map=_diagnosis_channels(diagnosis, incidence_only=True), tensor_from_file=tensor_from_file_fxn)

        # Build time to event TensorMaps
        name = f'cox_{diagnosis}'
        if name == desired_map_name:
            tff = loyalty_time_to_event(INCIDENCE_CSV, diagnosis_column=diagnosis2column[diagnosis])
            return TensorMap(f'{name}_newest', Interpretation.TIME_TO_EVENT, path_prefix=PARTNERS_PREFIX, tensor_from_file=tff)
        name = f'incident_cox_{diagnosis}'
        if name == desired_map_name:
            tff = loyalty_time_to_event(INCIDENCE_CSV, diagnosis_column=diagnosis2column[diagnosis], incidence_only=True)
            return TensorMap(f'{name}_newest', Interpretation.TIME_TO_EVENT, path_prefix=PARTNERS_PREFIX, tensor_from_file=tff)

        # Build survival curve TensorMaps
        for days_window in [1825]:
            name = f'survival_{diagnosis}_{days_window}'
            if name == desired_map_name:
                tff = _survival_from_file(days_window, INCIDENCE_CSV, diagnosis_column=diagnosis2column[diagnosis])
                return TensorMap(f'{name}', Interpretation.SURVIVAL_CURVE, path_prefix=PARTNERS_PREFIX, shape=(50,), days_window=days_window, tensor_from_file=tff)
            name = f'incident_survival_{diagnosis}'
            if name == desired_map_name:
                tff = _survival_from_file(days_window, INCIDENCE_CSV, diagnosis_column=diagnosis2column[diagnosis], incidence_only=True)
                return TensorMap(f'{name}', Interpretation.SURVIVAL_CURVE, path_prefix=PARTNERS_PREFIX, shape=(50,), days_window=days_window, tensor_from_file=tff)


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
        # if base_name not in TMAPS:
        #     continue

        # time_tmap = copy.deepcopy(TMAPS[base_name])
        #time_tmap = copy.deepcopy(tensormap_lookup(base_name, preix="ml4cvd.tensormap.partners"))
        time_tmap.name = needed_name
        time_tmap.shape = time_tmap.shape[1:]
        time_tmap.time_series_limit = time_series_limit
        time_tmap.time_series_order = time_series_order
        time_tmap.metrics = None
        time_tmap.infer_metrics()

        name2tensormap[needed_name] = time_tmap
    return name2tensormap



# MDRK: Disabled during update of TensorMaps
# def build_cardiac_surgery_tensor_maps(
#     needed_tensor_maps: List[str],
# ) -> Dict[str, TensorMap]:
#     name2tensormap: Dict[str, TensorMap] = {}
#     outcome2column = {
#         "sts_death": "mtopd",
#         "sts_stroke": "cnstrokp",
#         "sts_renal_failure": "crenfail",
#         "sts_prolonged_ventilation": "cpvntlng",
#         "sts_dsw_infection": "deepsterninf",
#         "sts_reoperation": "reop",
#         "sts_any_morbidity": "anymorbidity",
#         "sts_long_stay": "llos",
#     }

#     cardiac_surgery_dict = None
#     date_interval_lookup = None
#     for needed_name in needed_tensor_maps:
#         if needed_name in outcome2column:
#             if cardiac_surgery_dict is None:
#                 cardiac_surgery_dict = build_cardiac_surgery_dict(additional_columns=[column for outcome, column in outcome2column.items() if outcome in needed_tensor_maps])
#             channel_map = _outcome_channels(needed_name)
#             sts_tmap = TensorMap(
#                 needed_name,
#                 Interpretation.CATEGORICAL,
#                 path_prefix=PARTNERS_PREFIX,
#                 tensor_from_file=make_cardiac_surgery_outcome_tensor_from_file(cardiac_surgery_dict, outcome2column[needed_name]),
#                 channel_map=channel_map,
#                 validator=validator_not_all_zero,
#             )
#         else:
#             if not needed_name.endswith('_sts'):
#                 continue

#             base_name = needed_name.split('_sts')[0]
#             if base_name not in TMAPS:
#                 TMAPS.update(build_partners_time_series_tensor_maps([base_name]))
#                 if base_name not in TMAPS:
#                     continue

#             if cardiac_surgery_dict is None:
#                 cardiac_surgery_dict = build_cardiac_surgery_dict(additional_columns=[column for outcome, column in outcome2column.items() if outcome in needed_tensor_maps])
#             if date_interval_lookup is None:
#                 date_interval_lookup = build_date_interval_lookup(cardiac_surgery_dict)
#             sts_tmap = copy.deepcopy(TMAPS[base_name])
#             sts_tmap.name = needed_name
#             sts_tmap.time_series_lookup = date_interval_lookup

#         name2tensormap[needed_name] = sts_tmap

#     return name2tensormap
