# Imports: third party
import h5py
import numpy as np

# Imports: first party
from ml4cvd.TensorMap import TensorMap
from ml4cvd.definitions import ECG_ZERO_PADDING_THRESHOLD


def validator_clean_mrn(tm: TensorMap, tensor: np.ndarray, hd5: h5py.File):
    int(tensor)


def validator_not_all_zero(tm: TensorMap, tensor: np.ndarray, hd5: h5py.File):
    if np.count_nonzero(tensor) == 0:
        raise ValueError(
            f"TensorMap {tm.name} failed all-zero check on hd5 {hd5.filename}",
        )


def validator_no_empty(tm: TensorMap, tensor: np.ndarray, hd5: h5py.File):
    if any(tensor == ""):
        raise ValueError(
            f"TensorMap {tm.name} failed empty string check on hd5 {hd5.filename}",
        )


def validator_no_nans(tm: TensorMap, tensor: np.ndarray, hd5: h5py.File):
    if np.isnan(tensor).any():
        raise ValueError(
            f"TensorMap {tm.name} failed no nans check on hd5 {hd5.filename}.",
        )


def validator_no_negative(tm: TensorMap, tensor: np.ndarray, hd5: h5py.File):
    if any(tensor < 0):
        raise ValueError(
            f"TensorMap {tm.name} failed non-negative check on hd5 {hd5.filename}",
        )


def validator_voltage_no_zero_padding(
    tm: TensorMap, tensor: np.ndarray, hd5: h5py.File,
):
    for cm, idx in tm.channel_map.items():
        lead_length = tm.static_shape[-1]
        lead = tensor[..., tm.channel_map[cm]]
        num_zero = lead_length - np.count_nonzero(lead)
        if num_zero > ECG_ZERO_PADDING_THRESHOLD * lead_length:
            raise ValueError(f"Lead {cm} is zero-padded for ECG in {hd5.filename}")


def v6_zeros_validator(tm: TensorMap, tensor: np.ndarray, hd5: h5py.File):
    voltage = decompress_data(
        data_compressed=hd5["V6"][()], dtype=hd5["V6"].attrs["dtype"],
    )
    if np.count_nonzero(voltage == 0) > 10:
        raise ValueError(f"TensorMap {tm.name} has too many zeros in V6.")


class RangeValidator:
    def __init__(self, minimum: float, maximum: float):
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self, tm: TensorMap, tensor: np.ndarray, hd5: h5py.File):
        if not ((tensor > self.minimum).all() and (tensor < self.maximum).all()):
            raise ValueError(f"TensorMap {tm.name} failed range check.")

    def __str__(self):
        return f"Range Validator (min, max) = ({self.minimum}, {self.maximum})"

    def __repr__(self):
        return self.__str__()
