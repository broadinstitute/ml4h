import cv2
import h5py
import logging
import numpy as np

from ml4h.normalizer import ZeroMeanStd1
from ml4h.TensorMap import TensorMap, Interpretation
from ml4h.tensormap.general import get_tensor_at_first_date, normalized_first_date, pad_or_crop_array_to_shape

dxa_2 = TensorMap(
    'dxa_1_2',
    shape=(768, 768, 1),
    path_prefix='ukb_dxa',
    tensor_from_file=normalized_first_date,
    normalization=ZeroMeanStd1(),
)




dxa_8 = TensorMap(
    'dxa_1_8',
    shape=(640, 768, 1),
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


def register_to_sample(
        register_hd5='/mnt/disks/dxa-tensors-50k/2022-08-21/1000107.hd5',
        register_path='ukb_dxa',
        register_name='dxa_1_6',
        register_shape=(864, 736, 1),
        ):
    register_tensor = None
    number_of_iterations = 2000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-4

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    warp_mode = cv2.MOTION_TRANSLATION
    with h5py.File(register_hd5, 'r') as r_hd5:
        r_tensor = get_tensor_at_first_date(r_hd5, register_path, register_name)
        register_tensor = pad_or_crop_array_to_shape(register_shape, r_tensor)

    def _registered_tensor(tm, hd5, dependents={}):
        tensor = get_tensor_at_first_date(hd5, tm.path_prefix, tm.name)
        tensor = pad_or_crop_array_to_shape(tm.shape, tensor)
        background = np.bincount(tensor).argmax()
        tensor[tensor == background] = 0
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        try:
            (cc, warp_matrix) = cv2.findTransformECC(register_tensor.astype(np.float32), tensor.astype(np.float32),
                                                     warp_matrix, warp_mode, criteria)
            tensor[..., 0] = cv2.warpAffine(tensor, warp_matrix, (register_shape[1], register_shape[0]),
                                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);


        except cv2.error as e:
            logging.debug(f'Got cv2 error {e}')
        return tensor
    return _registered_tensor


dxa_6_register = TensorMap(
    'dxa_1_6',
    shape=(864, 736, 1),
    path_prefix='ukb_dxa',
    tensor_from_file=register_to_sample(),
    normalization=ZeroMeanStd1(),
)
