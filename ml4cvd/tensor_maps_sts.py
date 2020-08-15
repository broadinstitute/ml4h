# Imports: standard library
import logging
import datetime
from typing import Set, Dict, List, Tuple, Union, Callable, Optional

# Imports: third party
import h5py
import numpy as np
import pandas as pd

# Imports: first party
from ml4cvd.TensorMap import (
    TensorMap,
    Interpretation,
    TimeSeriesOrder,
    decompress_data,
    id_from_filename,
    outcome_channels,
)
from ml4cvd.normalizer import RobustScaler
from ml4cvd.validators import (
    validator_no_nans,
    validator_no_empty,
    validator_clean_mrn,
    validator_no_negative,
    validator_not_all_zero,
    validator_voltage_no_zero_padding,
)
from ml4cvd.definitions import (
    ECG_PREFIX,
    STS_DATA_CSV,
    STS_DATE_FORMAT,
    STS_PREOP_ECG_CSV,
    ECG_DATETIME_FORMAT,
    STS_DATETIME_FORMAT,
)

tmaps: Dict[str, TensorMap] = {}

sts_features_preoperative = {
    "age",
    "carshock",
    "chf",
    "chrlungd",
    "classnyh",
    "creatlst",
    "cva",
    "cvd",
    "cvdpcarsurg",
    "cvdtia",
    "diabetes",
    "dialysis",
    "ethnicity",
    "gender",
    "hct",
    "hdef",
    "heightcm",
    "hypertn",
    "immsupp",
    "incidenc",
    "infendty",
    "medadp5days",
    "medgp",
    "medinotr",
    "medster",
    "numdisv",
    "platelets",
    "pocpci",
    "pocpciin",
    "prcab",
    "prcvint",
    "prvalve",
    "pvd",
    "raceasian",
    "raceblack",
    "racecaucasian",
    "racenativeam",
    "raceothernativepacific",
    "resusc",
    "status",
    "vdinsufa",
    "vdinsufm",
    "vdinsuft",
    "vdstena",
    "vdstenm",
    "wbc",
    "weightkg",
}

sts_features_categorical = {
    "classnyh": [1, 2, 3, 4],
    "incidenc": [1, 2, 3],
    "numdisv": [0, 1, 2, 3],
    "infendty": [1, 2, 3],
    "status": [1, 2, 3, 4],
}

# Define the name, median, and IQR of continuous features to enable standardization
# These values are calculated from the entire STS MGH cohort using a Jupyter Notebook
# fmt: off
sts_features_continuous = {
    "age":       {"median": 67, "iqr": 18},
    "creatlst":  {"median": 1, "iqr": 0.36},
    "hct":       {"median": 39, "iqr": 8},
    "hdef":      {"median": 60, "iqr": 16},
    "heightcm":  {"median": 173, "iqr": 15},
    "platelets": {"median": 20700, "iqr": 90000},
    "wbc":       {"median": 7.3, "iqr": 3},
    "weightkg":  {"median": 82, "iqr": 24},
    "perfustm":  {"median": 123, "iqr": 72},
    "xclamptm":  {"median": 90, "iqr": 65},
}
# fmt: on

# Binary features are all pre-op features minus categorical and continuous features,
# plus cabg and valve procedures (binary)
sts_features_binary = (
    set(sts_features_preoperative)
    - set(sts_features_categorical)
    - set(sts_features_continuous)
)
sts_features_binary.add("opcab")
sts_features_binary.add("opvalve")

# fmt: off
sts_outcomes = {
    "sts_death":                 "mtopd",
    "sts_stroke":                "cnstrokp",
    "sts_renal_failure":         "crenfail",
    "sts_prolonged_ventilation": "cpvntlng",
    "sts_dsw_infection":         "deepsterninf",
    "sts_reoperation":           "reop",
    "sts_any_morbidity":         "anymorbidity",
    "sts_long_stay":             "llos",
}
# fmt: on


def _get_sts_features_dict(
    patient_column: str = "medrecn", date_column: str = "surgdt",
) -> Dict[int, Dict[str, float]]:
    """Get dict of STS features indexed by MRN;
    only keep last row (surgery) for each MRN"""
    sts_features = (
        pd.read_csv(STS_DATA_CSV)
        .sort_values([patient_column, date_column])
        .drop_duplicates(patient_column, keep="last")
        .set_index(patient_column)
        .to_dict("index")
    )
    return sts_features


def _make_sts_tff_continuous(
    sts_features: Dict[int, Dict[str, float]], key: str = "",
) -> Callable:
    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}):
        mrn = id_from_filename(hd5.filename)
        tensor = np.zeros(tm.shape, dtype=np.float32)
        for feature, idx in tm.channel_map.items():
            try:
                if key == "":
                    feature_value = sts_features[mrn][tm.name]
                else:
                    feature_value = sts_features[mrn][key]
                tensor[idx] = feature_value
            except:
                logging.debug(
                    f"Could not get continuous tensor using TMap {tm.name} from {hd5.filename}",
                )
        return tensor

    return tensor_from_file


def _make_sts_tff_binary(
    sts_features: Dict[int, Dict[str, Union[int, str]]],
    key: str = "",
    negative_value: int = 1,
    positive_value: int = 2,
) -> Callable:
    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}):
        """Parses MRN from the HD5 file name, look up the feature in a dict, and
        return the feature. Note the default encoding of +/- features in STS is 1/2, but
        for outcomes it is 0/1.
        """
        mrn = id_from_filename(hd5.filename)
        tensor = np.zeros(tm.shape, dtype=np.float32)
        if key == "":
            feature_value = sts_features[mrn][tm.name]
        else:
            feature_value = sts_features[mrn][key]
        if feature_value == positive_value:
            idx = 1
        elif feature_value == negative_value:
            idx = 0
        else:
            raise ValueError(
                f"TMap {tm.name} has value {feature_value} that is not a positive value ({positive_value}), or negative value ({negative_value})",
            )
        tensor[idx] = 1

        return tensor

    return tensor_from_file


def _make_sts_tff_categorical(
    sts_features: Dict[int, Dict[str, float]], key: str = "",
) -> Callable:
    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}):
        mrn = id_from_filename(hd5.filename)
        tensor = np.zeros(tm.shape, dtype=np.float32)
        if key == "":
            feature_value = sts_features[mrn][tm.name]
        else:
            feature_value = sts_features[mrn][key]
        for cm in tm.channel_map:
            original_value = cm.split("_")[1]
            if str(feature_value) == str(original_value):
                tensor[tm.channel_map[cm]] = 1
                break
        return tensor

    return tensor_from_file


def _make_sts_categorical_channel_map(feature: str) -> Dict[str, int]:
    """Create channel map for categorical STS feature;
    e.g. turns {"classnyh": [1, 2, 3, 4]} into
               {classnyh_1: 0, classnyh_2: 1, classnyh_3: 2, classnyh_4: 3}"""
    values = sts_features_categorical[feature]
    channel_map = dict()
    for idx, value in enumerate(values):
        channel_map[f"{feature}_{value}"] = idx
    return channel_map


def get_sts_surgery_dates(
    filename: str = STS_PREOP_ECG_CSV,
    patient_column: str = "medrecn",
    date_column: str = "surgdt",
    additional_columns: List[str] = [],
) -> Dict[int, Dict[str, Union[int, str]]]:
    keys = [date_column] + additional_columns
    sts_surgery_dates = {}
    df = (
        pd.read_csv(filename, low_memory=False, usecols=[patient_column] + keys)
        .sort_values(by=[patient_column, date_column])
        .drop_duplicates(patient_column, keep="last")
    )

    list_of_dicts = df.to_dict("records")

    # Iterate over each entry in list of dicts;
    # each entry is a unique MRN returned by entry[patient_column]
    sts_surgery_dates = {}
    for entry in list_of_dicts:
        sts_surgery_dates[entry[patient_column]] = {
            k: v for k, v in entry.items() if k != patient_column
        }

    return sts_surgery_dates


def build_date_interval_lookup(
    sts_features: Dict[int, Dict[str, Union[int, str]]],
    start_column: str = "surgdt",
    start_offset: int = -30,
    end_column: str = "surgdt",
    end_offset: int = 0,
) -> Dict[int, Tuple[str, str]]:
    """
    Get dict of tuples defining the date interval (start, end), indexed by MRN.
    This dict can be used for a tmap's time_series_lookup
    """
    date_interval_lookup = {}

    for mrn in sts_features:
        start_date = (
            str2datetime(
                input_date=sts_features[mrn][start_column],
                date_format=STS_DATETIME_FORMAT,
            )
            + datetime.timedelta(days=start_offset)
        ).strftime(ECG_DATETIME_FORMAT)

        end_date = (
            str2datetime(
                input_date=sts_features[mrn][end_column],
                date_format=STS_DATETIME_FORMAT,
            )
            + datetime.timedelta(days=end_offset)
        ).strftime(ECG_DATETIME_FORMAT)
        date_interval_lookup[mrn] = (start_date, end_date)
    return date_interval_lookup


def str2datetime(
    input_date: str, date_format: str = STS_DATE_FORMAT,
) -> datetime.datetime:
    return datetime.datetime.strptime(input_date, date_format)


# Get keys of outcomes in STS features CSV
outcome_keys = [key for outcome, key in sts_outcomes.items()]

# Get surgery dates and other outcomes, indexed in dict by MRN
sts_surgery_dates = get_sts_surgery_dates(additional_columns=outcome_keys)

# Build lookup table of date intervals to find pre-op data
date_interval_lookup = build_date_interval_lookup(sts_features=sts_surgery_dates)

# Get STS features from CSV as dict
sts_features = _get_sts_features_dict()

# Operative features
tmap_name = "bypass_time"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=_make_sts_tff_continuous(
        sts_features=sts_features, key="perfustm",
    ),
    channel_map={"perfustm": 0},
    validator=validator_no_nans,
)

tmap_name = "crossclamp_time"
tmaps[tmap_name] = TensorMap(
    name=tmap_name,
    interpretation=Interpretation.CONTINUOUS,
    path_prefix=ECG_PREFIX,
    tensor_from_file=_make_sts_tff_continuous(
        sts_features=sts_features, key="xclamptm",
    ),
    channel_map={"xclamptm": 0},
    validator=validator_no_nans,
)

# Categorical (non-binary)
for tmap_name in sts_features_categorical:
    interpretation = Interpretation.CATEGORICAL
    tff = _make_sts_tff_categorical(sts_features=sts_features)
    channel_map = _make_sts_categorical_channel_map(feature=tmap_name)
    validator = validator_no_nans

    tmaps[tmap_name] = TensorMap(
        name=tmap_name,
        interpretation=interpretation,
        path_prefix=ECG_PREFIX,
        tensor_from_file=tff,
        channel_map=channel_map,
        validator=validator,
    )

# Binary
for tmap_name in sts_features_binary:
    interpretation = Interpretation.CATEGORICAL
    tff = _make_sts_tff_binary(
        sts_features=sts_features, key=tmap_name, negative_value=1, positive_value=2,
    )
    channel_map = outcome_channels(tmap_name)
    validator = validator_no_nans

    tmaps[tmap_name] = TensorMap(
        name=tmap_name,
        interpretation=interpretation,
        path_prefix=ECG_PREFIX,
        tensor_from_file=tff,
        channel_map=channel_map,
        validator=validator,
    )

# Continuous
for tmap_name in sts_features_continuous:
    interpretation = Interpretation.CONTINUOUS

    # Note the need to set the key; otherwise, tff will use the tmap name
    # "foo_scaled" for the key, instead of "foo"
    tff = _make_sts_tff_continuous(sts_features=sts_features, key=tmap_name)
    validator = validator_no_nans

    # Make tmaps for both raw and scaled data
    for standardize in ["", "_scaled"]:
        channel_map = {tmap_name + standardize: 0}
        normalizer = (
            RobustScaler(
                median=sts_features_continuous[tmap_name]["median"],
                iqr=sts_features_continuous[tmap_name]["iqr"],
            )
            if standardize == "_scaled"
            else None
        )
        tmaps[tmap_name + standardize] = TensorMap(
            name=tmap_name + standardize,
            interpretation=interpretation,
            path_prefix=ECG_PREFIX,
            tensor_from_file=tff,
            channel_map=channel_map,
            validator=validator,
            normalization=normalizer,
        )

# Outcomes
for tmap_name in sts_outcomes:
    interpretation = Interpretation.CATEGORICAL
    tff = _make_sts_tff_binary(
        sts_features=sts_features,
        key=sts_outcomes[tmap_name],
        negative_value=0,
        positive_value=1,
    )
    channel_map = outcome_channels(tmap_name)
    validator = validator_not_all_zero

    tmaps[tmap_name] = TensorMap(
        name=tmap_name,
        interpretation=interpretation,
        path_prefix=ECG_PREFIX,
        tensor_from_file=tff,
        channel_map=channel_map,
        validator=validator,
    )
