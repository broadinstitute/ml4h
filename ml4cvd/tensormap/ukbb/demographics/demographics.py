import h5py
import numpy as np
import logging
from typing import List, Tuple
from ml4cvd.tensor_writer_ukbb import tensor_path
from ml4cvd.TensorMap import TensorMap, Interpretation, str2date
from ml4cvd.defines import StorageType


def age_in_years_tensor(date_key,
                         birth_key='continuous/34_Year-of-birth_0_0',
                         population_normalize=False):
    def age_at_tensor_from_file(tm: TensorMap,
                                hd5: h5py.File,
                                dependents=None):
        try:
            age = np.array([hd5['ecg/latest/patient_info/Age'][()]])
        except:
            logging.info('could not get age')
            raise KeyError('cold not')
        # age = age.astype("float")

        return age
        # return tm.normalize_and_validate(np.array([assess_date.year-birth_year]))

    return age_at_tensor_from_file


def prevalent_incident_tensor(start_date_key, event_date_key):
    def _prevalent_incident_tensor_from_file(tm: TensorMap,
                                             hd5: h5py.File,
                                             dependents=None):
        index = 0
        categorical_data = np.zeros(tm.shape, dtype=np.float32)
        if tm.hd5_key_guess() in hd5:
            data = tm.hd5_first_dataset_in_group(hd5, tm.hd5_key_guess())
            if tm.storage_type == StorageType.CATEGORICAL_INDEX or tm.storage_type == StorageType.CATEGORICAL_FLAG:
                index = int(data[0])
                categorical_data[index] = 1.0
            else:
                categorical_data = np.array(data)
        elif tm.storage_type == StorageType.CATEGORICAL_FLAG:
            categorical_data[index] = 1.0
        else:
            raise ValueError(
                f"No HD5 Key at prefix {tm.path_prefix} found for tensor map: {tm.name}."
            )

        if index != 0:
            if event_date_key in hd5 and start_date_key in hd5:
                disease_date = str2date(str(hd5[event_date_key][0]))
                assess_date = str2date(str(hd5[start_date_key][0]))
            else:
                raise ValueError(f"No date found for tensor map: {tm.name}.")
            index = 1 if disease_date < assess_date else 2
        categorical_data[index] = 1.0
        return categorical_data

    return _prevalent_incident_tensor_from_file


def preprocess_with_function(fxn, hd5_key=None):
    def preprocess_tensor_from_file(tm, hd5, dependents={}):
        missing = True
        continuous_data = np.zeros(tm.shape, dtype=np.float32)
        my_key = tm.hd5_key_guess() if hd5_key is None else hd5_key
        if my_key in hd5:
            missing = False
            continuous_data[0] = tm.hd5_first_dataset_in_group(hd5, my_key)[0]
        if missing and tm.sentinel is None:
            raise ValueError(
                f'No value found for {tm.name}, a continuous TensorMap with no sentinel value, and channel keys:{list(tm.channel_map.keys())}.'
            )
        elif missing:
            continuous_data[:] = tm.sentinel
        return fxn(continuous_data)

    return preprocess_tensor_from_file


def _weekly_alcohol(instance):
    alcohol_keys = [
        f'1568_Average-weekly-red-wine-intake_{instance}_0',
        f'1578_Average-weekly-champagne-plus-white-wine-intake_{instance}_0',
        f'1588_Average-weekly-beer-plus-cider-intake_{instance}_0',
        f'1598_Average-weekly-spirits-intake_{instance}_0',
        f'1608_Average-weekly-fortified-wine-intake_{instance}_0',
    ]

    def alcohol_from_file(tm, hd5, dependents={}):
        drinks = 0
        for k in alcohol_keys:
            data = tm.hd5_first_dataset_in_group(
                hd5, key_prefix=f'{tm.path_prefix}/{k}')
            drinks += float(data[0])
        return np.array([drinks], dtype=np.float32)

    return alcohol_from_file


log_25781_2 = TensorMap(
    '25781_Total-volume-of-white-matter-hyperintensities-from-T1-and-T2FLAIR-images_2_0',
    loss='logcosh',
    path_prefix='continuous',
    normalization={
        'mean': 7,
        'std': 8
    },
    tensor_from_file=preprocess_with_function(np.log),
    channel_map={'white-matter-hyper-intensities': 0},
)

weight_lbs_2 = TensorMap(
    'weight_lbs',
    Interpretation.CONTINUOUS,
    normalization={
        'mean': 168.74,
        'std': 34.1
    },
    loss='logcosh',
    channel_map={'weight_lbs': 0},
    tensor_from_file=preprocess_with_function(lambda x: x * 2.20462,
                                              'continuous/21002_Weight_2_0'),
)

weekly_alcohol_0 = TensorMap('weekly_alcohol_0',
                             loss='logcosh',
                             path_prefix='continuous',
                             channel_map={'weekly_alcohol_0': 0},
                             tensor_from_file=_weekly_alcohol(0))
weekly_alcohol_1 = TensorMap('weekly_alcohol_1',
                             loss='logcosh',
                             path_prefix='continuous',
                             channel_map={'weekly_alcohol_1': 0},
                             tensor_from_file=_weekly_alcohol(1))
weekly_alcohol_2 = TensorMap('weekly_alcohol_2',
                             loss='logcosh',
                             path_prefix='continuous',
                             channel_map={'weekly_alcohol_2': 0},
                             tensor_from_file=_weekly_alcohol(2))
