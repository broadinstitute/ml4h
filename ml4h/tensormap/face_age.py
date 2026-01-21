import gzip
import h5py
import numpy as np

from typing import Dict
from ml4h.TensorMap import TensorMap
from ml4h.normalizer import Standardize, ZeroMeanStd1

def image_from_hd5(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}) -> np.ndarray:
    compressed = hd5["image"][:].tobytes()
    shape = tuple(hd5.attrs["shape"])

    raw = gzip.decompress(compressed)
    return np.array(np.frombuffer(raw, dtype=np.uint8).reshape(shape), dtype=np.float32)


def age_from_hd5(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}) -> np.ndarray:
    return np.array([hd5["age"][()]], dtype=np.float32)


face_image = TensorMap('face_image', shape=(200, 200, 3), tensor_from_file=image_from_hd5)
face_age = TensorMap('face_age', shape=(1,), tensor_from_file=age_from_hd5)

face_image_norm = TensorMap('face_image', shape=(200, 200, 3), tensor_from_file=image_from_hd5, normalization=ZeroMeanStd1())
face_age_norm = TensorMap('face_age', shape=(1,), tensor_from_file=age_from_hd5, normalization=Standardize(mean=45, std=20))