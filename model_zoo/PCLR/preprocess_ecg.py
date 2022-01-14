from typing import Dict
import numpy as np


LEADS = [
    'I', 'II', 'III', 'aVR', 'aVL', 'aVF',
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
]


def process_ecg(ecg: Dict[str, np.ndarray], ecg_samples: int = 4096) -> np.ndarray:
    """
    Prepares an ECG for use in a tensorflow model
    :param ecg: A dictionary mapping lead name to lead values.
                The lead values should be measured in milli-volts.
                Each lead should represent 10s of samples.
    :param ecg_samples: Length of each lead for input into the model.
    :return: a numpy array of the ECG shaped (ecg_samples, 12)
    """
    assert set(ecg.keys()) == set(LEADS)

    out = np.zeros((ecg_samples, 12))
    for i, lead_name in enumerate(LEADS):
        lead = ecg[lead_name]
        interpolated_lead = np.interp(
            np.linspace(0, 1, ecg_samples),
            np.linspace(0, 1, lead.shape[0]),
            lead,
        )
        out[:, i] = interpolated_lead
    return out
