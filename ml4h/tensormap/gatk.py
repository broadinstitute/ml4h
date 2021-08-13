import h5py
import logging
import numpy as np
from typing import Dict

from ml4h.TensorMap import TensorMap, Interpretation
from ml4h.normalizer import Standardize

DNA_SYMBOLS = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
VARIANT_LABELS = {'NOT_SNP': 0, 'NOT_INDEL': 1, 'SNP': 2, 'INDEL': 3}


def tensor_from_hd5(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}) -> np.ndarray:
    return np.array(hd5[tm.name])


reference = TensorMap('reference', shape=(128, len(DNA_SYMBOLS)), tensor_from_file=tensor_from_hd5)
read_tensor = TensorMap('read_tensor', shape=(128, 128, 15), tensor_from_file=tensor_from_hd5)
dp = TensorMap('dp', shape=(1,), normalization=Standardize(mean=34, std=8.6), tensor_from_file=tensor_from_hd5)
fs = TensorMap('fs', shape=(1,), normalization=Standardize(mean=4.03, std=7.2), tensor_from_file=tensor_from_hd5)
qd = TensorMap('qd', shape=(1,), normalization=Standardize(mean=12.8, std=6.1), tensor_from_file=tensor_from_hd5)
mq = TensorMap('mq', shape=(1,), normalization=Standardize(mean=59.1, std=8.6), tensor_from_file=tensor_from_hd5)
sor = TensorMap('sor', shape=(1,), normalization=Standardize(mean=1.03, std=0.8), tensor_from_file=tensor_from_hd5)
mqranksum = TensorMap('mqranksum', shape=(1,), normalization=Standardize(mean=-0.23, std=1.1), tensor_from_file=tensor_from_hd5)
readposranksum = TensorMap('readposranksum', shape=(1,), normalization=Standardize(mean=-0.04, std=1.2), tensor_from_file=tensor_from_hd5)


def variant_label_from_hd5(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}) -> np.ndarray:
    one_hot = np.zeros(tm.shape, dtype=np.float32)
    variant_str = str(hd5['variant_label'][()], 'utf-8')
    for channel in tm.channel_map:
        if channel.lower() == variant_str.lower():
            one_hot[tm.channel_map[channel]] = 1.0
    if one_hot.sum() != 1:
        raise ValueError(f'TensorMap {tm.name} missing or invalid label: {variant_str} one_hot: {one_hot}')
    return one_hot


variant_label = TensorMap(
    'variant_label', Interpretation.CATEGORICAL,
    shape=(len(VARIANT_LABELS),),
    tensor_from_file=variant_label_from_hd5,
    channel_map=VARIANT_LABELS
)