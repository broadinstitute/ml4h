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


def _get_path_to_sts_data() -> str:
    """Get path to STS data depending on the machine hostname"""
    if "anduril" == socket.gethostname():
        path = "~/dropbox/sts_data"
    elif "mithril" == socket.gethostname():
        path = "~/dropbox/sts_data"
    elif "stultzlab" in socket.gethostname():
        path = "/storage/shared/sts_data_deid"
    else:
        path = "~/dropbox/sts_data"
    return os.path.expanduser(path)


CARDIAC_SURGERY_OUTCOMES_CSV = os.path.join(
    _get_path_to_sts_data(), "mgh-preop-ecg-outcome-labels.csv",
)
CARDIAC_SURGERY_FEATURES_CSV = os.path.join(
    _get_path_to_sts_data(), "mgh-all-features-labels.csv",
)

ECG_PREFIX = "partners_ecg_rest"
DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"
ECG_DATE_FORMAT = "%m-%d-%Y"
ECG_TIME_FORMAT = "%H:%M:%S"
ECG_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
CARDIAC_SURGERY_DATE_FORMAT = "%Y-%m-%d"
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

CAD_ICDS = [
    "K401",
    "K402",
    "K403",
    "K404",
    "K411",
    "K412",
    "K413",
    "K414",
    "K451",
    "K452",
    "K453",
    "K454",
    "K455",
    "K491",
    "K492",
    "K498",
    "K499",
    "K502",
    "K751",
    "K752",
    "K753",
    "K754",
    "K758",
    "K759",
]

# TODO: These values should ultimately come from the coding table
CODING_VALUES_LESS_THAN_ONE = [-10, -1001]
CODING_VALUES_MISSING = [-3, -1, -2, -11, -818, -121, -313, -906]

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

ECG_BIKE_LEADS = {"I": 0, "2": 1, "3": 2}
ECG_BIKE_MEDIAN_SIZE = (5500, len(ECG_BIKE_LEADS))
ECG_BIKE_STRIP_SIZE = (5000, len(ECG_BIKE_LEADS))
ECG_BIKE_FULL_SIZE = (216500, len(ECG_BIKE_LEADS))
ECG_BIKE_RECOVERY_SIZE = (500 * 60, len(ECG_BIKE_LEADS))

ECG_CHAR_2_IDX = {
    "!": 0,
    " ": 1,
    "'": 2,
    ")": 4,
    "(": 3,
    "-": 5,
    "/": 6,
    "1": 7,
    "3": 9,
    "2": 8,
    "4": 10,
    ":": 11,
    "?": 12,
    "A": 13,
    "C": 15,
    "B": 14,
    "E": 17,
    "D": 16,
    "G": 18,
    "I": 20,
    "H": 19,
    "J": 21,
    "M": 23,
    "L": 22,
    "N": 24,
    "Q": 26,
    "P": 25,
    "S": 28,
    "R": 27,
    "U": 30,
    "T": 29,
    "W": 32,
    "V": 31,
    "a": 33,
    "c": 35,
    "b": 34,
    "e": 37,
    "d": 36,
    "g": 39,
    "f": 38,
    "i": 41,
    "h": 40,
    "k": 43,
    "j": 42,
    "m": 45,
    "l": 44,
    "o": 47,
    "n": 46,
    "q": 49,
    "p": 48,
    "s": 51,
    "r": 50,
    "u": 53,
    "t": 52,
    "w": 55,
    "v": 54,
    "y": 57,
    "x": 56,
    "z": 58,
    "O": 59,
    "5": 60,
}
ECG_IDX_2_CHAR = {
    0: "!",
    1: " ",
    2: "'",
    3: "(",
    4: ")",
    5: "-",
    6: "/",
    7: "1",
    8: "2",
    9: "3",
    10: "4",
    11: ":",
    12: "?",
    13: "A",
    14: "B",
    15: "C",
    16: "D",
    17: "E",
    18: "G",
    19: "H",
    20: "I",
    21: "J",
    22: "L",
    23: "M",
    24: "N",
    25: "P",
    26: "Q",
    27: "R",
    28: "S",
    29: "T",
    30: "U",
    31: "V",
    32: "W",
    33: "a",
    34: "b",
    35: "c",
    36: "d",
    37: "e",
    38: "f",
    39: "g",
    40: "h",
    41: "i",
    42: "j",
    43: "k",
    44: "l",
    45: "m",
    46: "n",
    47: "o",
    48: "p",
    49: "q",
    50: "r",
    51: "s",
    52: "t",
    53: "u",
    54: "v",
    55: "w",
    56: "x",
    57: "y",
    58: "z",
    59: "O",
    60: "5",
}

PARTNERS_READ_TEXT = "_read"
PARTNERS_CHAR_2_IDX = {
    " ": 0,
    "0": 1,
    "1": 2,
    "2": 3,
    "3": 4,
    "4": 5,
    "5": 6,
    "6": 7,
    "7": 8,
    "8": 9,
    "9": 10,
    "a": 11,
    "b": 12,
    "c": 13,
    "d": 14,
    "e": 15,
    "f": 16,
    "g": 17,
    "h": 18,
    "i": 19,
    "j": 20,
    "k": 21,
    "l": 22,
    "m": 23,
    "n": 24,
    "o": 25,
    "p": 26,
    "q": 27,
    "r": 28,
    "s": 29,
    "t": 30,
    "u": 31,
    "v": 32,
    "w": 33,
    "x": 34,
    "y": 35,
    "z": 36,
}
PARTNERS_IDX_2_CHAR = {
    0: " ",
    1: "0",
    2: "1",
    3: "2",
    4: "3",
    5: "4",
    6: "5",
    7: "6",
    8: "7",
    9: "8",
    10: "9",
    11: "a",
    12: "b",
    13: "c",
    14: "d",
    15: "e",
    16: "f",
    17: "g",
    18: "h",
    19: "i",
    20: "j",
    21: "k",
    22: "l",
    23: "m",
    24: "n",
    25: "o",
    26: "p",
    27: "q",
    28: "r",
    29: "s",
    30: "t",
    31: "u",
    32: "v",
    33: "w",
    34: "x",
    35: "y",
    36: "z",
}

CARDIAC_SURGERY_PREOPERATIVE_FEATURES = {
    "age": 0,
    "carshock": 1,
    "chf": 2,
    "chrlungd": 3,
    "classnyh": 4,
    "creatlst": 5,
    "cva": 6,
    "cvd": 7,
    "cvdpcarsurg": 8,
    "cvdtia": 9,
    "diabetes": 10,
    "dialysis": 11,
    "ethnicity": 12,
    "gender": 13,
    "hct": 14,
    "hdef": 15,
    "heightcm": 16,
    "hypertn": 17,
    "immsupp": 18,
    "incidenc": 19,
    "infendty": 20,
    "medadp5days": 21,
    "medgp": 22,
    "medinotr": 23,
    "medster": 24,
    "numdisv": 25,
    "platelets": 26,
    "pocpci": 27,
    "pocpciin": 28,
    "prcab": 29,
    "prcvint": 30,
    "prvalve": 31,
    "pvd": 32,
    "raceasian": 33,
    "raceblack": 34,
    "racecaucasian": 35,
    "racenativeam": 36,
    "raceothernativepacific": 37,
    "resusc": 38,
    "status": 39,
    "vdinsufa": 40,
    "vdinsufm": 41,
    "vdinsuft": 42,
    "vdstena": 43,
    "vdstenm": 44,
    "wbc": 45,
    "weightkg": 46,
}
