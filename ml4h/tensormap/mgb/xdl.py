from typing import Dict

import h5py
import numpy as np
from ml4h.TensorMap import TensorMap, Interpretation

ecg_5000_std = TensorMap('ecg_5000_std', Interpretation.CONTINUOUS, shape=(5000, 12))
ecg_single_lead_I = TensorMap(f'ecg_strip_I', Interpretation.CONTINUOUS, shape=(5000, 1))

hypertension_icd_only = TensorMap(
    name='hypertension_icd_only', interpretation=Interpretation.CATEGORICAL,
    channel_map={'no_hypertension_icd_only': 0, 'hypertension_icd_only': 1},
)
hypertension_icd_bp = TensorMap(
    name='hypertension_icd_bp', interpretation=Interpretation.CATEGORICAL,
    channel_map={'no_hypertension_icd_bp': 0, 'hypertension_icd_bp': 1},
)
hypertension_icd_bp_med = TensorMap(
    name='hypertension_icd_bp_med', interpretation=Interpretation.CATEGORICAL,
    channel_map={'no_hypertension_icd_bp_med': 0, 'hypertension_icd_bp_med': 1},
)
hypertension_med = TensorMap(
    name='start_fu_hypertension_med', interpretation=Interpretation.CATEGORICAL,
    channel_map={'no_hypertension_medication': 0, 'hypertension_medication': 1},
)

lvef = TensorMap(name='LVEF', interpretation=Interpretation.CONTINUOUS, channel_map={'LVEF': 0})

age = TensorMap(name='age_in_days', interpretation=Interpretation.CONTINUOUS, channel_map={'age_in_days': 0})
sex = TensorMap(name='sex', interpretation=Interpretation.CATEGORICAL, channel_map={'Female': 0, 'Male': 1})

cad = TensorMap(name='cad', interpretation=Interpretation.CATEGORICAL, channel_map={'no_cad': 0, 'cad': 1})
dm = TensorMap(name='dm', interpretation=Interpretation.CATEGORICAL, channel_map={'no_dm': 0, 'dm': 1})
hypercholesterolemia = TensorMap(
    name='hypercholesterolemia', interpretation=Interpretation.CATEGORICAL,
    channel_map={'no_hypercholesterolemia': 0, 'hypercholesterolemia': 1},
)

n_intervals = 25
af_tmap = TensorMap('survival_curve_af', Interpretation.SURVIVAL_CURVE, shape=(n_intervals*2,))
death_tmap = TensorMap('death_event', Interpretation.SURVIVAL_CURVE, shape=(n_intervals*2,))


def ecg_median_biosppy(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}) -> np.ndarray:
    tensor = np.zeros(tm.shape, dtype=np.float32)
    for lead in tm.channel_map:
        tensor[:, tm.channel_map[lead]] = hd5[f'{tm.path_prefix}{lead}']
    tensor = np.nan_to_num(tensor)
    return tensor

ecg_channel_map = {
    'I': 0, 'II': 1, 'III': 2, 'aVR': 3, 'aVL': 4, 'aVF': 5,
    'V1': 6, 'V2': 7, 'V3': 8, 'V4': 9, 'V5': 10, 'V6': 11,
}

ecg_biosppy_median_60bpm = TensorMap(
    'median', Interpretation.CONTINUOUS, path_prefix='median_60bpm_', shape=(600, 12),
    tensor_from_file=ecg_median_biosppy,
    channel_map=ecg_channel_map,
)
