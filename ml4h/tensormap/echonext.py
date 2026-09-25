import numpy as np

from ml4h.TensorMap import TensorMap, Interpretation


def _read_dataset(hd5, path):
    if path not in hd5:
        raise KeyError(f"Missing dataset '{path}' in {hd5.filename}")
    return hd5[path]


def echonext_ecg_2500_12_from_hd5(tm, hd5, dependents=None):
    """Return ECG waveform as 2500 x 12 x 1."""
    x = np.array(_read_dataset(hd5, "ecg/waveform"), dtype=np.float32)
    if x.shape != (2500, 12):
        raise ValueError(f"Expected ecg/waveform shape (2500, 12), got {x.shape} in {hd5.filename}")
    return x[:, :, np.newaxis]


def echonext_ecg_1x2500x12_from_hd5(tm, hd5, dependents=None):
    """Return ECG waveform as 1 x 2500 x 12, matching original EchoNext npy format."""
    x = np.array(_read_dataset(hd5, "ecg/waveform_1x2500x12"), dtype=np.float32)
    if x.shape != (1, 2500, 12):
        raise ValueError(f"Expected ecg/waveform_1x2500x12 shape (1, 2500, 12), got {x.shape} in {hd5.filename}")
    return x


def echonext_tabular_features_from_hd5(tm, hd5, dependents=None):
    x = np.array(_read_dataset(hd5, "ecg/tabular_features"), dtype=np.float32)
    if x.shape != (7,):
        raise ValueError(f"Expected ecg/tabular_features shape (7,), got {x.shape} in {hd5.filename}")
    return x


def _binary_label_from_hd5(label_name):
    def tensor_from_file(tm, hd5, dependents=None):
        path = f"labels/{label_name}"
        y = np.array(_read_dataset(hd5, path), dtype=np.float32).reshape(-1)

        if y.size != 1:
            raise ValueError(f"Expected one value at {path}, got shape {y.shape} in {hd5.filename}")

        label = int(round(float(y[0])))
        if label not in (0, 1):
            raise ValueError(f"Expected binary 0/1 label at {path}, got {y[0]} in {hd5.filename}")

        out = np.zeros((2,), dtype=np.float32)
        out[label] = 1.0
        return out

    return tensor_from_file


echonext_ecg_2500_12 = TensorMap(
    "echonext_ecg_2500_12",
    shape=(2500, 12, 1),
    tensor_from_file=echonext_ecg_2500_12_from_hd5,
)

echonext_ecg_1x2500x12 = TensorMap(
    "echonext_ecg_1x2500x12",
    shape=(1, 2500, 12),
    tensor_from_file=echonext_ecg_1x2500x12_from_hd5,
)

echonext_tabular_features = TensorMap(
    "echonext_tabular_features",
    shape=(7,),
    tensor_from_file=echonext_tabular_features_from_hd5,
)

echonext_shd_moderate_or_greater = TensorMap(
    "echonext_shd_moderate_or_greater",
    Interpretation.CATEGORICAL,
    tensor_from_file=_binary_label_from_hd5("shd_moderate_or_greater_flag"),
    channel_map={
        "no_shd_moderate_or_greater": 0,
        "shd_moderate_or_greater": 1,
    },
)

echonext_lvef_lte_45 = TensorMap(
    "echonext_lvef_lte_45",
    Interpretation.CATEGORICAL,
    tensor_from_file=_binary_label_from_hd5("lvef_lte_45_flag"),
    channel_map={"lvef_gt_45": 0, "lvef_lte_45": 1},
)

echonext_lvwt_gte_13 = TensorMap(
    "echonext_lvwt_gte_13",
    Interpretation.CATEGORICAL,
    tensor_from_file=_binary_label_from_hd5("lvwt_gte_13_flag"),
    channel_map={"lvwt_lt_13": 0, "lvwt_gte_13": 1},
)

echonext_aortic_stenosis_moderate_or_greater = TensorMap(
    "echonext_aortic_stenosis_moderate_or_greater",
    Interpretation.CATEGORICAL,
    tensor_from_file=_binary_label_from_hd5("aortic_stenosis_moderate_or_greater_flag"),
    channel_map={
        "no_aortic_stenosis_moderate_or_greater": 0,
        "aortic_stenosis_moderate_or_greater": 1,
    },
)

echonext_aortic_regurgitation_moderate_or_greater = TensorMap(
    "echonext_aortic_regurgitation_moderate_or_greater",
    Interpretation.CATEGORICAL,
    tensor_from_file=_binary_label_from_hd5("aortic_regurgitation_moderate_or_greater_flag"),
    channel_map={
        "no_aortic_regurgitation_moderate_or_greater": 0,
        "aortic_regurgitation_moderate_or_greater": 1,
    },
)

echonext_mitral_regurgitation_moderate_or_greater = TensorMap(
    "echonext_mitral_regurgitation_moderate_or_greater",
    Interpretation.CATEGORICAL,
    tensor_from_file=_binary_label_from_hd5("mitral_regurgitation_moderate_or_greater_flag"),
    channel_map={
        "no_mitral_regurgitation_moderate_or_greater": 0,
        "mitral_regurgitation_moderate_or_greater": 1,
    },
)

echonext_tricuspid_regurgitation_moderate_or_greater = TensorMap(
    "echonext_tricuspid_regurgitation_moderate_or_greater",
    Interpretation.CATEGORICAL,
    tensor_from_file=_binary_label_from_hd5("tricuspid_regurgitation_moderate_or_greater_flag"),
    channel_map={
        "no_tricuspid_regurgitation_moderate_or_greater": 0,
        "tricuspid_regurgitation_moderate_or_greater": 1,
    },
)

echonext_pulmonary_regurgitation_moderate_or_greater = TensorMap(
    "echonext_pulmonary_regurgitation_moderate_or_greater",
    Interpretation.CATEGORICAL,
    tensor_from_file=_binary_label_from_hd5("pulmonary_regurgitation_moderate_or_greater_flag"),
    channel_map={
        "no_pulmonary_regurgitation_moderate_or_greater": 0,
        "pulmonary_regurgitation_moderate_or_greater": 1,
    },
)

echonext_rv_systolic_dysfunction_moderate_or_greater = TensorMap(
    "echonext_rv_systolic_dysfunction_moderate_or_greater",
    Interpretation.CATEGORICAL,
    tensor_from_file=_binary_label_from_hd5("rv_systolic_dysfunction_moderate_or_greater_flag"),
    channel_map={
        "no_rv_systolic_dysfunction_moderate_or_greater": 0,
        "rv_systolic_dysfunction_moderate_or_greater": 1,
    },
)

echonext_pericardial_effusion_moderate_large = TensorMap(
    "echonext_pericardial_effusion_moderate_large",
    Interpretation.CATEGORICAL,
    tensor_from_file=_binary_label_from_hd5("pericardial_effusion_moderate_large_flag"),
    channel_map={
        "no_pericardial_effusion_moderate_large": 0,
        "pericardial_effusion_moderate_large": 1,
    },
)

echonext_pasp_gte_45 = TensorMap(
    "echonext_pasp_gte_45",
    Interpretation.CATEGORICAL,
    tensor_from_file=_binary_label_from_hd5("pasp_gte_45_flag"),
    channel_map={"pasp_lt_45": 0, "pasp_gte_45": 1},
)

echonext_tr_max_gte_32 = TensorMap(
    "echonext_tr_max_gte_32",
    Interpretation.CATEGORICAL,
    tensor_from_file=_binary_label_from_hd5("tr_max_gte_32_flag"),
    channel_map={"tr_max_lt_32": 0, "tr_max_gte_32": 1},
)

def _shd_condition_plane(tm, hd5, dependents=None):
    value = float(hd5["labels/shd_moderate_or_greater_flag"][0])
    return np.full(tm.shape, value, dtype=np.float32)


echonext_shd_condition_plane = TensorMap(
    "echonext_shd_condition_plane",
    Interpretation.CONTINUOUS,
    shape=(2500, 12, 1),
    tensor_from_file=_shd_condition_plane,
)
