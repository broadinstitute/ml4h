import gzip
import h5py
import numpy as np
from typing import Dict

from scipy.signal import convolve2d
from scipy.ndimage import median_filter
from scipy.ndimage.interpolation import rotate

from ml4h.TensorMap import TensorMap
from ml4h.normalizer import Standardize, ZeroMeanStd1
from ml4h.tensormap.general import pad_or_crop_array_to_shape


def image_from_hd5(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}) -> np.ndarray:
    compressed = hd5["image"][:].tobytes()
    shape = tuple(hd5.attrs["shape"])

    raw = gzip.decompress(compressed)
    arr = np.array(np.frombuffer(raw, dtype=np.uint8).reshape(shape), dtype=np.float32)
    return pad_or_crop_array_to_shape(tm.shape, arr)


def age_from_hd5(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}) -> np.ndarray:
    return np.array([hd5["age"][()]], dtype=np.float32)


face_image = TensorMap('face_image', shape=(200, 200, 3), tensor_from_file=image_from_hd5)
face_age = TensorMap('face_age', shape=(1,), tensor_from_file=age_from_hd5)

face_image_norm = TensorMap('face_image', shape=(200, 200, 3), tensor_from_file=image_from_hd5, normalization=ZeroMeanStd1())
face_age_norm = TensorMap('face_age', shape=(1,), tensor_from_file=age_from_hd5, normalization=Standardize(mean=45, std=20))

face_image_norm_192 = TensorMap('face_image_192', shape=(192, 192, 3), tensor_from_file=image_from_hd5, normalization=ZeroMeanStd1())
face_image_norm_224 = TensorMap('face_image_224', shape=(224, 224, 3), tensor_from_file=image_from_hd5, normalization=ZeroMeanStd1())

sharp_kernel = np.c_[
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
]


def _sharpen(img):
    if np.random.rand() > 0.5:
        img = img.copy()
        img[..., 0] = convolve2d(img[..., 0], sharp_kernel, mode="same", boundary="symm")
        img[..., 1] = convolve2d(img[..., 1], sharp_kernel, mode="same", boundary="symm")
        img[..., 2] = convolve2d(img[..., 2], sharp_kernel, mode="same", boundary="symm")
        return img
    return img


def _median_filter(img):
    window_size = np.random.randint(1, 15)
    return np.expand_dims(median_filter(img[..., :], size=(window_size, window_size)), axis=-1)

def _make_rotate(min: float, max: float):
    def _rotate(img):
        angle = np.random.randint(min, max)
        return rotate(img, angle=angle, reshape=False)
    return _rotate

def _gaussian_noise(img, mean=0, sigma=0.03):
    img = img.copy()
    noise = np.random.normal(mean, sigma, img.shape)
    img += noise
    return img

face_image_norm_192_gaussian = TensorMap('face_image_192', shape=(192, 192, 3), tensor_from_file=image_from_hd5,
                                        normalization=ZeroMeanStd1(),
                                        augmentations=[_gaussian_noise, ])
face_image_norm_192_sharpen = TensorMap('face_image_192', shape=(192, 192, 3), tensor_from_file=image_from_hd5,
                                        normalization=ZeroMeanStd1(),
                                        augmentations=[_sharpen, ])
face_image_norm_192_rotate = TensorMap('face_image_192', shape=(192, 192, 3), tensor_from_file=image_from_hd5,
                                        normalization=ZeroMeanStd1(),
                                        augmentations=[_make_rotate(-15, 15), ])
