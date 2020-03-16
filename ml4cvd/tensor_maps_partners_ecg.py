import os
import csv
import logging
import datetime

import h5py
import numcodecs
import numpy as np
from typing import Dict, List

from ml4cvd.metrics import weighted_crossentropy
from ml4cvd.TensorMap import (TensorMap, no_nans, str2date, make_range_validator, Interpretation)
from ml4cvd.defines import ECG_REST_AMP_LEADS
from ml4cvd.tensor_maps_by_hand import TMAPS


def _compress_data(hf, name, data, dtype, method='zstd', compression_opts=19):
    # Define codec
    codec = numcodecs.zstd.Zstd(level=compression_opts)

    # If data is string, encode to bytes
    if dtype == 'str':
        data_compressed = codec.encode(data.encode())
        dsize = len(data.encode())
    else:
        data_compressed = codec.encode(data)
        dsize = len(data) * data.itemsize

    # Save data to hdf5
    dat = hf.create_dataset(name=name, data=np.void(data_compressed))

    # Set attributes
    dat.attrs['method']              = method
    dat.attrs['compression_level']   = compression_opts
    dat.attrs['len']                 = len(data)
    dat.attrs['uncompressed_length'] = dsize
    dat.attrs['compressed_length']   = len(data_compressed)
    dat.attrs['dtype'] = dtype
   

def _decompress_data(data_compressed, dtype):
    codec = numcodecs.zstd.Zstd() 
    data_decompressed = codec.decode(data_compressed)
    if dtype == 'str':
        data = data_decompressed.decode()
    else:
        data = np.frombuffer(data_decompressed, dtype)
    return data


def _resample_voltage(voltage, desired_samples):
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


def make_voltage(population_normalize: float = None):
    def get_voltage_from_file(tm, hd5, dependents={}):
        tensor = np.zeros(tm.shape, dtype=np.float32)
        for cm in tm.channel_map:
            voltage = _decompress_data(data_compressed=hd5[cm][()], dtype=hd5[cm].attrs['dtype'])
            voltage = _resample_voltage(voltage, tm.shape[0])
            tensor[:, tm.channel_map[cm]] = voltage 
        if population_normalize is None:
            tm.normalization = {'zero_mean_std1': True}
        else:
            tensor /= population_normalize 
        return tensor
    return get_voltage_from_file


TMAPS['partners_ecg_voltage'] = TensorMap('partners_ecg_voltage',
                                        shape=(2500, 12),
                                        interpretation=Interpretation.CONTINUOUS,
                                        tensor_from_file=make_voltage(population_normalize=2000.0),
                                        channel_map=ECG_REST_AMP_LEADS)

TMAPS['partners_ecg_2500'] = TensorMap('ecg_rest_2500', shape=(2500, 12), tensor_from_file=make_voltage(), channel_map=ECG_REST_AMP_LEADS)
TMAPS['partners_ecg_5000'] = TensorMap('ecg_rest_5000', shape=(5000, 12), tensor_from_file=make_voltage(), channel_map=ECG_REST_AMP_LEADS)
TMAPS['partners_ecg_2500_raw'] = TensorMap('ecg_rest_2500_raw', shape=(2500, 12), tensor_from_file=make_voltage(population_normalize=2000.0), channel_map=ECG_REST_AMP_LEADS)
TMAPS['partners_ecg_5000_raw'] = TensorMap('ecg_rest_5000_raw', shape=(5000, 12), tensor_from_file=make_voltage(population_normalize=2000.0), channel_map=ECG_REST_AMP_LEADS)


def make_voltage_attr(volt_attr: str = ""):
    def get_voltage_attr_from_file(tm, hd5, dependents={}):
        tensor = np.zeros(tm.shape, dtype=np.float32)
        for cm in tm.channel_map:
            tensor[tm.channel_map[cm]] = hd5[cm].attrs[volt_attr]
        return tensor
    return get_voltage_attr_from_file


TMAPS["voltage_len"] = TensorMap("voltage_len",
                                 interpretation=Interpretation.CONTINUOUS,
                                 tensor_from_file=make_voltage_attr(volt_attr="len"),
                                 shape=(12,),
                                 channel_map=ECG_REST_AMP_LEADS)

TMAPS["len_i"] = TensorMap("len_i", shape=(1,), tensor_from_file=make_voltage_attr(volt_attr="len"), channel_map={'I': 0})
TMAPS["len_v6"] = TensorMap("len_v6", shape=(1,), tensor_from_file=make_voltage_attr(volt_attr="len"), channel_map={'V6': 0})


def make_partners_ecg_label(key: str = "read_md_clean", dict_of_list: Dict = dict(), not_found_key: str = "unspecified"):
    def get_partners_ecg_label(tm, hd5, dependents={}):
        read = _decompress_data(data_compressed=hd5[key][()], dtype=hd5[key].attrs['dtype'])
        label_array = np.zeros(tm.shape, dtype=np.float32)
        for channel, idx in sorted(tm.channel_map.items(), key=lambda cm: cm[1]):
            if channel in dict_of_list:
                for string in dict_of_list[channel]:
                    if string in read:
                        label_array[idx] = 1
                        return label_array
        label_array[tm.channel_map[not_found_key]] = 1
        return label_array
    return get_partners_ecg_label


def partners_ecg_label_from_list(keys: List[str] = ["read_md_clean"], dict_of_list: Dict = dict(), not_found_key: str = "unspecified"):
    def get_partners_ecg_label(tm, hd5, dependents={}):
        label_array = np.zeros(tm.shape, dtype=np.float32)
        for key in keys:
            if key not in hd5:
                continue
            read = _decompress_data(data_compressed=hd5[key][()], dtype=hd5[key].attrs['dtype'])
            for channel, idx in sorted(tm.channel_map.items(), key=lambda cm: cm[1]):
                if channel in dict_of_list:
                    for string in dict_of_list[channel]:
                        if string in read:
                            label_array[idx] = 1
                            return label_array
        label_array[tm.channel_map[not_found_key]] = 1
        return label_array
    return get_partners_ecg_label


def make_partners_ecg_tensor(key: str):
    def get_partners_ecg_tensor(tm, hd5, dependents={}):
        tensor = _decompress_data(data_compressed=hd5[key][()], dtype=hd5[key].attrs['dtype'])
        if tm.interpretation == Interpretation.LANGUAGE:
            return np.array([str(tensor)])
        elif tm.interpretation == Interpretation.CONTINUOUS:
            return np.array([tensor], dtype=np.float32)
        elif tm.interpretation == Interpretation.CATEGORICAL:
            return np.array([float(tensor)])
    return get_partners_ecg_tensor


task = "partners_ecg_read_md_raw"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.LANGUAGE,
                        tensor_from_file=make_partners_ecg_tensor(key="read_md_clean"),
                        shape=(1,))


task = "partners_ecg_read_pc_raw"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.LANGUAGE,
                        tensor_from_file=make_partners_ecg_tensor(key="read_pc_clean"),
                        shape=(1,))


def validator_cross_reference(tm: TensorMap, tensor: np.ndarray):
    if int(tensor) not in tm.cross_reference:
        raise ValueError(f"Skipping TensorMap {tm.name} not found in Apollo.")


def create_cross_reference_dict(fpath="/data/apollo/demographics.csv"):
    try:
        with open(fpath, mode="r") as f:
            reader = csv.reader(f)
            next(reader)
            cross_reference_dict = {int(rows[0]):None for rows in reader}
        return cross_reference_dict
    except FileNotFoundError:
        return {}


task = "partners_ecg_patientid_cross_reference_apollo"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.LANGUAGE,
                        tensor_from_file=make_partners_ecg_tensor(key="patientid"),
                        shape=(1,),
                        validator=validator_cross_reference)

TMAPS[task].cross_reference = create_cross_reference_dict()

task = "partners_ecg_patientid"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.LANGUAGE,
                        tensor_from_file=make_partners_ecg_tensor(key="patientid"),
                        shape=(1,))

task = "partners_ecg_firstname"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.LANGUAGE,
                        tensor_from_file=make_partners_ecg_tensor(key="patientfirstname"),
                        shape=(1,))

task = "partners_ecg_lastname"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.LANGUAGE,
                        tensor_from_file=make_partners_ecg_tensor(key="patientlastname"),
                        shape=(1,))

task = "partners_ecg_date"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.LANGUAGE,
                        tensor_from_file=make_partners_ecg_tensor(key="acquisitiondate"),
                        shape=(1,))

task = "partners_ecg_dob"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.LANGUAGE,
                        tensor_from_file=make_partners_ecg_tensor(key="dateofbirth"),
                        shape=(1,))

task = "partners_ecg_sampling_frequency"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.CONTINUOUS,
                        tensor_from_file=make_partners_ecg_tensor(key="ecgsamplebase"),
                        shape=(1,))

task = "partners_ecg_rate"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.CONTINUOUS,
                        loss='logcosh',
                        tensor_from_file=make_partners_ecg_tensor(key="ventricularrate"),
                        shape=(1,),
                        validator=make_range_validator(10, 200))

TMAPS['partners_ventricular_rate'] = TensorMap('VentricularRate', loss='logcosh', tensor_from_file=make_partners_ecg_tensor(key="ventricularrate"), shape=(1,),
                                               validator=make_range_validator(10, 200), normalization={'mean': 59.3, 'std': 10.6})

task = "partners_ecg_qrs"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.CONTINUOUS,
                        loss='logcosh',
                        metrics=['mse'],
                        tensor_from_file=make_partners_ecg_tensor(key="qrsduration"),
                        shape=(1,),
                        validator=make_range_validator(20, 400))

task = "partners_ecg_pr"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.CONTINUOUS,
                        loss='logcosh',
                        metrics=['mse'],
                        tensor_from_file=make_partners_ecg_tensor(key="printerval"),
                        shape=(1,),
                        validator=make_range_validator(50, 500))

task = "partners_ecg_qt"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.CONTINUOUS,
                        loss='logcosh',
                        tensor_from_file=make_partners_ecg_tensor(key="qtinterval"),
                        shape=(1,),
                        validator=make_range_validator(100, 800))

task = "partners_ecg_qtc"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.CONTINUOUS,
                        loss='logcosh',
                        tensor_from_file=make_partners_ecg_tensor(key="qtcorrected"),
                        shape=(1,),
                        validator=make_range_validator(100, 800))


task = "partners_weight_lbs"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.CONTINUOUS,
                        loss='logcosh',
                        tensor_from_file=make_partners_ecg_tensor(key="weightlbs"),
                        shape=(1,),
                        validator=make_range_validator(100, 800))


def _partners_str2date(d):
    parts = d.split('-')
    if len(parts) < 2:
        raise ValueError(f'Can not parse date: {d}')
    return datetime.date(int(parts[2]), int(parts[0]), int(parts[1]))


def partners_ecg_age(tm, hd5, dependents={}):
    birthday = _decompress_data(data_compressed=hd5['dateofbirth'][()], dtype=hd5['dateofbirth'].attrs['dtype'])
    acquisition = _decompress_data(data_compressed=hd5['acquisitiondate'][()], dtype=hd5['acquisitiondate'].attrs['dtype'])
    delta = _partners_str2date(acquisition) - _partners_str2date(birthday)
    years = delta.days / 365.0
    return np.array([years])


TMAPS['partners_ecg_age'] = TensorMap('partners_ecg_age', loss='logcosh', tensor_from_file=partners_ecg_age, shape=(1,))


def partners_ecg_acquisition_year(tm, hd5, dependents={}):
    acquisition = _decompress_data(data_compressed=hd5['acquisitiondate'][()], dtype=hd5['acquisitiondate'].attrs['dtype'])
    return np.array([_partners_str2date(acquisition).year])


TMAPS['partners_ecg_acquisition_year'] = TensorMap('partners_ecg_acquisition_year', loss='logcosh',  tensor_from_file=partners_ecg_acquisition_year, shape=(1,))


def partners_bmi(tm, hd5, dependents={}):
    weight_lbs = _decompress_data(data_compressed=hd5['weightlbs'][()], dtype=hd5['weightlbs'].attrs['dtype'])
    weight_kg = 0.453592 * float(weight_lbs)
    height_in = _decompress_data(data_compressed=hd5['heightin'][()], dtype=hd5['heightin'].attrs['dtype'])
    height_m = 0.0254 * float(height_in)
    logging.info(f' Height was {height_in} weight: {weight_lbs} bmi is {weight_kg / (height_m*height_m)}')
    return np.array([weight_kg / (height_m*height_m)])


TMAPS['partners_bmi'] = TensorMap('bmi', channel_map={'bmi': 0}, tensor_from_file=partners_bmi)


def partners_channel_string(hd5_key, race_synonyms={}, unspecified_key=None):
    def tensor_from_string(tm, hd5, dependents={}):
        hd5_string = _decompress_data(data_compressed=hd5[hd5_key][()], dtype=hd5[hd5_key].attrs['dtype'])
        tensor = np.zeros(tm.shape, dtype=np.float32)
        for key in tm.channel_map:
            if hd5_string.lower() == key.lower():
                tensor[tm.channel_map[key]] = 1.0
                return tensor
            if key in race_synonyms:
                for synonym in race_synonyms[key]:
                    if hd5_string.lower() == synonym.lower():
                        tensor[tm.channel_map[key]] = 1.0
                        return tensor
        if unspecified_key is None:
            raise ValueError(f'No channel keys found in {hd5_string} for {tm.name} with channel map {tm.channel_map}.')
        tensor[tm.channel_map[unspecified_key]] = 1.0
        return tensor
    return tensor_from_string


race_synonyms = {'asian': ['oriental'], 'hispanic': ['latino'], 'white': ['caucasian']}
TMAPS['partners_race'] = TensorMap('race', interpretation=Interpretation.CATEGORICAL, channel_map={'asian': 0, 'black': 1, 'hispanic': 2, 'white': 3, 'unknown': 4},
                                   tensor_from_file=partners_channel_string('race', race_synonyms))
TMAPS['partners_gender'] = TensorMap('gender', interpretation=Interpretation.CATEGORICAL, channel_map={'female': 0, 'male': 1},
                                     tensor_from_file=partners_channel_string('gender'))


def _partners_adult(hd5_key, minimum_age=18):
    def tensor_from_string(tm, hd5, dependents={}):
        birthday = _decompress_data(data_compressed=hd5['dateofbirth'][()], dtype=hd5['dateofbirth'].attrs['dtype'])
        acquisition = _decompress_data(data_compressed=hd5['acquisitiondate'][()], dtype=hd5['acquisitiondate'].attrs['dtype'])
        delta = _partners_str2date(acquisition) - _partners_str2date(birthday)
        years = delta.days / 365.0
        if years < minimum_age:
            raise ValueError(f'ECG taken on patient below age cutoff.')
        hd5_string = _decompress_data(data_compressed=hd5[hd5_key][()], dtype=hd5[hd5_key].attrs['dtype'])
        tensor = np.zeros(tm.shape, dtype=np.float32)
        for key in tm.channel_map:
            if hd5_string.lower() == key.lower():
                tensor[tm.channel_map[key]] = 1.0
                return tensor
        raise ValueError(f'No channel keys found in {hd5_string} for {tm.name} with channel map {tm.channel_map}.')
    return tensor_from_string


TMAPS['partners_adult_gender'] = TensorMap('adult_gender', interpretation=Interpretation.CATEGORICAL, channel_map={'female': 0, 'male': 1},
                                           tensor_from_file=_partners_adult('gender'))


def voltage_zeros(tm, hd5, dependents={}):
    tensor = np.zeros(tm.shape, dtype=np.float32)
    for cm in tm.channel_map:
        voltage = _decompress_data(data_compressed=hd5[cm][()], dtype=hd5[cm].attrs['dtype'])
        tensor[tm.channel_map[cm]] = np.count_nonzero(voltage == 0)
    return tensor


TMAPS["lead_i_zeros"] = TensorMap("lead_i_zeros", shape=(1,), tensor_from_file=voltage_zeros, channel_map={'I': 0})
TMAPS["lead_v6_zeros"] = TensorMap("lead_v6_zeros", shape=(1,), tensor_from_file=voltage_zeros, channel_map={'V6': 0})


def v6_zeros_validator(tm: TensorMap, tensor: np.ndarray, hd5: h5py.File):
    voltage = _decompress_data(data_compressed=hd5['V6'][()], dtype=hd5['V6'].attrs['dtype'])
    if np.count_nonzero(voltage == 0) > 10:
        raise ValueError(f'TensorMap {tm.name} has too many zeros in V6.')


def build_incidence_tensor_from_file(file_name: str, patient_column: str='Mrn', date_column: str='first_stroke', censor_date_str: str = '2020-03-16', delimiter: str = ','):
    """
    Build a tensor_from_file function from a column and date in a file.
    Only works for continuous values.
    """
    error = None
    censor_date = str2date(censor_date_str)
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader)
            patient_index = header.index(patient_column)
            date_index = header.index(date_column)
            date_table = {}
            patient_table = {}
            for row in reader:
                try:
                    patient_key = int(row[patient_index])
                    patient_table[patient_key] = True
                    if row[date_index] == '' or row[date_index] == 'NULL':
                        continue
                    disease_date = str2date(row[date_index].split(' ')[0])
                    date_table[patient_key] = disease_date
                    if len(patient_table) % 2000 == 0:
                        logging.debug(f'Processed: {len(patient_table)} patient rows.')
                except ValueError as e:
                    logging.warning(f'val err {e}')
            logging.info(f'Done processing {date_column} Got {len(patient_table)} patient rows and {len(date_table)} events.')
    except FileNotFoundError as e:
        error = e

    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        if error:
            raise error

        categorical_data = np.zeros(tm.shape, dtype=np.float32)
        file_split = os.path.basename(hd5.filename).split('-')
        mrn = file_split[0]
        mrn_int = int(mrn)
        if int(file_split[1]) < 2000:
            raise ValueError(f'Asssessed earlier than enrollment.')
        if mrn_int not in patient_table:
            raise KeyError(f'{tm.name} mrn not in incidence csv')
        if mrn_int not in date_table:
            index = 0
        else:
            disease_date = date_table[mrn_int]
            assess_date = _partners_str2date(_decompress_data(data_compressed=hd5['acquisitiondate'][()], dtype=hd5['acquisitiondate'].attrs['dtype']))
            index = 1 if disease_date < assess_date else 2
            logging.debug(f'mrn: {mrn_int}  Got disease_date: {disease_date} assess  {assess_date} index  {index}.')
        categorical_data[index] = 1.0
        return categorical_data
    return tensor_from_file


TMAPS["loyalty_stroke_wrt_ecg"] = TensorMap('stroke_wrt_ecg', Interpretation.CATEGORICAL,
                                            tensor_from_file=build_incidence_tensor_from_file('/media/erisone_snf13/lc_outcomes.csv'),
                                            channel_map={'no_stroke': 0, 'prevalent_stroke': 1, 'incident_stroke': 2})
TMAPS["loyalty_stroke_wrt_ecg_weighted"] = TensorMap('stroke_wrt_ecg', Interpretation.CATEGORICAL,
                                                     tensor_from_file=build_incidence_tensor_from_file('/media/erisone_snf13/lc_outcomes.csv', ),
                                                     channel_map={'no_stroke': 0, 'prevalent_stroke': 1, 'incident_stroke': 2},
                                                     loss=weighted_crossentropy([1.0, 10.0, 10.0], 'loyal_stroke'))

TMAPS["loyalty_afib_wrt_ecg"] = TensorMap('afib_wrt_ecg', Interpretation.CATEGORICAL,
                                            tensor_from_file=build_incidence_tensor_from_file('/media/erisone_snf13/lc_outcomes.csv', date_column='first_af'),
                                            channel_map={'no_afib': 0, 'prevalent_afib': 1, 'incident_afib': 2})

TMAPS["loyalty_lvh_wrt_ecg"] = TensorMap('lvh_wrt_ecg', Interpretation.CATEGORICAL,
                                            tensor_from_file=build_incidence_tensor_from_file('/media/erisone_snf13/lc_outcomes.csv', date_column='first_lvh'),
                                            channel_map={'no_lvh': 0, 'prevalent_lvh': 1, 'incident_lvh': 2})
TMAPS["loyalty_cad_wrt_ecg"] = TensorMap('cad_wrt_ecg', Interpretation.CATEGORICAL,
                                            tensor_from_file=build_incidence_tensor_from_file('/media/erisone_snf13/lc_outcomes.csv', date_column='first_cad'),
                                            channel_map={'no_cad': 0, 'prevalent_cad': 1, 'incident_cad': 2})
TMAPS["loyalty_cvd_wrt_ecg"] = TensorMap('cvd_wrt_ecg', Interpretation.CATEGORICAL,
                                            tensor_from_file=build_incidence_tensor_from_file('/media/erisone_snf13/lc_outcomes.csv', date_column='first_cvd'),
                                            channel_map={'no_cvd': 0, 'prevalent_cvd': 1, 'incident_cvd': 2})
TMAPS["loyalty_bpmed_wrt_ecg"] = TensorMap('bpmed_wrt_ecg', Interpretation.CATEGORICAL,
                                            tensor_from_file=build_incidence_tensor_from_file('/media/erisone_snf13/lc_outcomes.csv', date_column='first_bpmed'),
                                            channel_map={'no_bpmed': 0, 'prevalent_bpmed': 1, 'incident_bpmed': 2})
TMAPS["loyalty_htn_wrt_ecg"] = TensorMap('htn_wrt_ecg', Interpretation.CATEGORICAL,
                                            tensor_from_file=build_incidence_tensor_from_file('/media/erisone_snf13/lc_outcomes.csv', date_column='first_htn'),
                                            channel_map={'no_htn': 0, 'prevalent_htn': 1, 'incident_htn': 2})
TMAPS["loyalty_hf_wrt_ecg"] = TensorMap('hf_wrt_ecg', Interpretation.CATEGORICAL,
                                            tensor_from_file=build_incidence_tensor_from_file('/media/erisone_snf13/lc_outcomes.csv', date_column='first_hf'),
                                            channel_map={'no_hf': 0, 'prevalent_hf': 1, 'incident_hf': 2})

# DEFINE ALL DISEASES!!

# ALSO Make COx Proportional Hazard,

'''
task = "partners_ecg_rate_norm"
TMAPS[task] = TensorMap(task,
                        interpretation=interpretation,
                        dtype=DataSetType.CONTINUOUS,
                        loss='logcosh',
                        metrics=['mse'],
                        normalization={'mean': 81.620467, 'std': 20.352292},
                        tensor_from_file=make_partners_ecg_tensor(key="ventricularrate"),
                        shape=(1,),
                        validator=make_range_validator(10, 200))

task = "partners_ecg_qrs_norm"
TMAPS[task] = TensorMap(task,
                        interpretation=interpretation,
                        dtype=DataSetType.CONTINUOUS,
                        loss='logcosh',
                        metrics=['mse'],
                        normalization={'mean': 94.709106, 'std': 22.610711},
                        tensor_from_file=make_partners_ecg_tensor(key="qrsduration"),
                        shape=(1,),
                        validator=make_range_validator(20, 400))

task = "partners_ecg_pr_norm"
TMAPS[task] = TensorMap(task,
                        interpretation=interpretation,
                        dtype=DataSetType.CONTINUOUS,
                        loss='logcosh',
                        metrics=['mse'],
                        normalization={'std': 35.003017, 'mean': 161.040738},
                        tensor_from_file=make_partners_ecg_tensor(key="printerval"),
                        shape=(1,),
                        validator=make_range_validator(50, 500))

task = "partners_ecg_qt_norm"
TMAPS[task] = TensorMap(task,
                        interpretation=interpretation,
                        loss='logcosh',
                        metrics=['mse'],
                        normalization={'mean': 390.995792, 'std': 50.923113},
                        dtype=DataSetType.CONTINUOUS,
                        tensor_from_file=make_partners_ecg_tensor(key="qtinterval"),
                        shape=(1,),
                        validator=make_range_validator(100, 800))

task = "partners_ecg_qtc_norm"
TMAPS[task] = TensorMap(task,
                        interpretation=interpretation,
                        loss='logcosh',
                        metrics=['mse'],
                        normalization={'std': 39.762255, 'mean': 446.505327},
                        dtype=DataSetType.CONTINUOUS,
                        tensor_from_file=make_partners_ecg_tensor(key="qtcorrected"),
                        shape=(1,),
                        validator=make_range_validator(100, 800))
'''

