# Imports: standard library
import os
import socket
from enum import Enum, auto
from typing import Dict, List, Union

# Imports: third party
import numpy as np


class StorageType(Enum):
    CONTINUOUS = auto()
    CATEGORICAL_INDEX = auto()
    CATEGORICAL_FLAG = auto()
    ONE_HOT = auto()
    STRING = auto()
    BYTE_STRING = auto()

    def __str__(self):
        """StorageType.FLOAT_ARRAY becomes float_array"""
        return str.lower(super().__str__().split(".")[1])


ArgumentList = List[Union[int, float]]
Arguments = Dict[str, Union[int, float, ArgumentList]]
Inputs = Dict[str, np.ndarray]
Outputs = Inputs
Path = str
Paths = List[Path]
Predictions = List[np.ndarray]

YEAR_DAYS = 365.26
ECG_ZERO_PADDING_THRESHOLD = 0.25


def _get_sts_data_path() -> str:
    """Get path to STS data depending on the machine hostname"""
    if "anduril" == socket.gethostname():
        path = "~/dropbox/sts-data"
    elif "mithril" == socket.gethostname():
        path = "~/dropbox/sts-data"
    elif "stultzlab" in socket.gethostname():
        path = "/storage/shared/sts-data-deid"
    else:
        path = "~/dropbox/sts-data"
    return os.path.expanduser(path)


STS_PREOP_ECG_CSV = os.path.join(
    _get_sts_data_path(), "mgh-preop-ecg-outcome-labels.csv",
)
STS_DATA_CSV = os.path.join(_get_sts_data_path(), "mgh-all-features-labels.csv")

ECG_PREFIX = "partners_ecg_rest"
DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"
ECG_DATE_FORMAT = "%m-%d-%Y"
ECG_TIME_FORMAT = "%H:%M:%S"
ECG_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
STS_DATE_FORMAT = "%Y-%m-%d"
STS_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
MRN_COLUMNS = {"mgh_mrn", "sampleid", "medrecn", "mrn", "patient_id", "patientid"}

EPS = 1e-7

DICOM_EXT = ".dcm"
IMAGE_EXT = ".png"
PDF_EXT = ".pdf"
TENSOR_EXT = ".hd5"
MODEL_EXT = ".h5"
XML_EXT = ".xml"
CSV_EXT = ".csv"

STOP_CHAR = "!"
JOIN_CHAR = "_"
CONCAT_CHAR = "-"
HD5_GROUP_CHAR = "/"

ECG_READ_TEXT = "_read"

ECG_REST_LEADS = {
    "strip_I": 0,
    "strip_II": 1,
    "strip_III": 2,
    "strip_V1": 3,
    "strip_V2": 4,
    "strip_V3": 5,
    "strip_V4": 6,
    "strip_V5": 7,
    "strip_V6": 8,
    "strip_aVF": 9,
    "strip_aVL": 10,
    "strip_aVR": 11,
}
ECG_REST_MEDIAN_LEADS = {
    "median_I": 0,
    "median_II": 1,
    "median_III": 2,
    "median_V1": 3,
    "median_V2": 4,
    "median_V3": 5,
    "median_V4": 6,
    "median_V5": 7,
    "median_V6": 8,
    "median_aVF": 9,
    "median_aVL": 10,
    "median_aVR": 11,
}
ECG_REST_AMP_LEADS = {
    "I": 0,
    "II": 1,
    "III": 2,
    "aVR": 3,
    "aVL": 4,
    "aVF": 5,
    "V1": 6,
    "V2": 7,
    "V3": 8,
    "V4": 9,
    "V5": 10,
    "V6": 11,
}
ECG_REST_INDEPENDENT_LEADS = {
    "I": 0,
    "II": 1,
    "V1": 2,
    "V2": 3,
    "V3": 4,
    "V4": 5,
    "V5": 6,
    "V6": 7,
}
ECG_SEGMENTED_CHANNEL_MAP = {
    "unknown": 0,
    "TP_segment": 1,
    "P_wave": 2,
    "PQ_segment": 3,
    "QRS_complex": 4,
    "ST_segment": 5,
    "T_wave": 6,
    "U_wave": 7,
}
