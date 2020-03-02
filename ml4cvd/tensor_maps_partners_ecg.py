import csv
import numpy as np
from typing import Dict
from ml4cvd.TensorMap import (TensorMap, no_nans, str2date,
        make_range_validator, Interpretation)
from ml4cvd.defines import ECG_REST_AMP_LEADS
from ml4cvd.tensor_maps_by_hand import TMAPS


def _resample_voltage(voltage):
    if len(voltage) == 5000:
        return voltage[::2]
    else:
        return voltage

def make_voltage(population_normalize: float = None):
    def get_voltage_from_file(tm, hd5, dependents={}):
        tensor = np.zeros(tm.shape, dtype=np.float32)
        for cm in tm.channel_map:
            voltage = _decompress_data(data_compressed=hd5[cm][()],
                                       dtype=hd5[cm].attrs['dtype'])
            voltage = _resample_voltage(voltage)
            tensor[:, tm.channel_map[cm]] = voltage 
        if population_normalize is None:
            tensor = tm.zero_mean_std1(tensor)
        else:
            tensor /= population_normalize 
        return tensor
    return get_voltage_from_file


TMAPS['partners_ecg_voltage'] = TensorMap('partners_ecg_voltage',
                                        shape=(2500, 12),
                                        interpretation=Interpretation.CONTINUOUS,
                                        tensor_from_file=make_voltage(population_normalize=2000.0),
                                        channel_map=ECG_REST_AMP_LEADS)


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


def make_partners_ecg_label(key: str = "read_md_clean",
                            dict_of_list: Dict = dict(),
                            not_found_key: str = "unspecified"):
    def get_partners_ecg_label(tm, hd5, dependents={}):
        read = _decompress_data(data_compressed=hd5[key][()],
                                dtype=hd5[key].attrs['dtype'])       
        label_array = np.zeros(tm.shape, dtype=np.float32)
        for cm in tm.channel_map:
            for string in dict_of_list[cm]:
                if string in read:
                    label_array[tm.channel_map[cm]] = 1
                    return label_array
        label_array[tm.channel_map[not_found_key]] = 1
        return label_array
    return get_partners_ecg_label


def make_partners_ecg_tensor(key: str, tensor_type = float):
    def get_partners_ecg_tensor(tm, hd5, dependents={}):
        tensor = _decompress_data(data_compressed=hd5[key][()],
                                  dtype=hd5[key].attrs['dtype'])
        return np.array(tensor_type(tensor))
    return get_partners_ecg_tensor


task = "partners_ecg_read_md_raw"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.LANGUAGE,
                        tensor_from_file=make_partners_ecg_tensor(
                            key="read_md_clean", tensor_type=str),
                        shape=(1,))


task = "partners_ecg_read_pc_raw"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.LANGUAGE,
                        tensor_from_file=make_partners_ecg_tensor(
                            key="read_pc_clean", tensor_type=str),
                        shape=(1,))

def validator_cross_reference(tm: TensorMap, tensor: np.ndarray):
    if int(tensor) not in tm.cross_reference:
        raise ValueError(f"Skipping TensorMap {tm.name} not found in Apollo.")

def create_cross_reference_dict(fpath="/data/apollo/demographics.csv"):
    with open(fpath, mode="r") as f:
        reader = csv.reader(f)
        next(reader)
        cross_reference_dict = {int(rows[0]):None for rows in reader}
    return cross_reference_dict

task = "partners_ecg_patientid_cross_reference_apollo"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.LANGUAGE,
                        tensor_from_file=make_partners_ecg_tensor(
                            key="patientid", tensor_type=float),
                        shape=(1,),
                        validator=validator_cross_reference)

TMAPS[task].cross_reference = create_cross_reference_dict()

task = "partners_ecg_patientid"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.LANGUAGE,
                        tensor_from_file=make_partners_ecg_tensor(
                            key="patientid", tensor_type=float),
                        shape=(1,))

task = "partners_ecg_date"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.LANGUAGE,
                        tensor_from_file=make_partners_ecg_tensor(
                            key="acquisitiondate", tensor_type=str),
                        shape=(1,))

task = "partners_ecg_dob"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.LANGUAGE,
                        tensor_from_file=make_partners_ecg_tensor(
                            key="dateofbirth", tensor_type=str),
                        shape=(1,))

task = "partners_ecg_sampling_frequency"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.CONTINUOUS,
                        tensor_from_file=make_partners_ecg_tensor(
                            key="ecgsamplebase", tensor_type=float),
                        shape=(1,))

task = "partners_ecg_rate"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.CONTINUOUS,
                        loss='logcosh',
                        metrics=['mse'],
                        tensor_from_file=make_partners_ecg_tensor(
                            key="ventricularrate", tensor_type=float),
                        shape=(1,),
                        validator=make_range_validator(10, 200))

task = "partners_ecg_qrs"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.CONTINUOUS,
                        loss='logcosh',
                        metrics=['mse'],
                        tensor_from_file=make_partners_ecg_tensor(
                            key="qrsduration", tensor_type=float),
                        shape=(1,),
                        validator=make_range_validator(20, 400))

task = "partners_ecg_pr"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.CONTINUOUS,
                        loss='logcosh',
                        metrics=['mse'],
                        tensor_from_file=make_partners_ecg_tensor(
                            key="printerval", tensor_type=float),
                        shape=(1,),
                        validator=make_range_validator(50, 500))

task = "partners_ecg_qt"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.CONTINUOUS,
                        loss='logcosh',
                        metrics=['mse'],
                        tensor_from_file=make_partners_ecg_tensor(
                            key="qtinterval", tensor_type=float),
                        shape=(1,),
                        validator=make_range_validator(100, 800))

task = "partners_ecg_qtc"
TMAPS[task] = TensorMap(task,
                        interpretation=Interpretation.CONTINUOUS,
                        loss='logcosh',
                        metrics=['mse'],
                        tensor_from_file=make_partners_ecg_tensor(
                            key="qtcorrected", tensor_type=float),
                        shape=(1,),
                        validator=make_range_validator(100, 800))

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

