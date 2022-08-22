import h5py
import numpy as np

from ml4h.normalizer import ZeroMeanStd1
from ml4h.TensorMap import TensorMap, Interpretation
from ml4h.tensormap.general import get_tensor_at_first_date, normalized_first_date, pad_or_crop_array_to_shape

dxa_8 = TensorMap(
    'dxa_1_8',
    shape=(640, 780, 1),
    path_prefix='ukb_dxa',
    tensor_from_file=normalized_first_date,
    normalization=ZeroMeanStd1(),
)


dxa_12 = TensorMap(
    'dxa_1_12',
    shape=(896, 320, 1),
    path_prefix='ukb_dxa',
    tensor_from_file=normalized_first_date,
    normalization=ZeroMeanStd1(),
)

dxa_11 = TensorMap(
    'dxa_1_11',
    shape=(896, 352, 1),
    path_prefix='ukb_dxa',
    tensor_from_file=normalized_first_date,
    normalization=ZeroMeanStd1(),
)
