from typing import List, Dict

import numpy as np
from tensorflow.keras.models import load_model, Model

from preprocess_ecg import process_ecg, LEADS


def get_model() -> Model:
    """Get PCLR embedding model"""
    return load_model("./PCLR.h5")


def get_representations(ecgs: List[Dict[str, np.ndarray]]) -> np.ndarray:
    """
    Uses PCLR trained model to build representations of ECGs
    :param ecgs: A list of dictionaries mapping lead name to lead values.
                 The lead values should be measured in milli-volts.
                 Each lead should represent 10s of samples.
    :return:
    """
    model = get_model()
    ecgs = np.stack(list(map(process_ecg, ecgs)))
    return model.predict(ecgs)


def test_get_representations():
    """Test to make sure get_representations works as expected"""
    fake_ecg = {
        lead: np.zeros(2500)
        for lead in LEADS
    }
    fake_ecgs = [fake_ecg for _ in range(10)]
    representations = get_representations(fake_ecgs)
    assert representations.shape == (len(fake_ecgs), 320)
