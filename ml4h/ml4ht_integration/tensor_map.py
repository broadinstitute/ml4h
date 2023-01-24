from typing import Callable, List, Tuple

import h5py
import numpy as np

from ml4ht.data.data_description import DataDescription
from ml4ht.data.defines import SampleID, LoadingOption, Tensor, Batch
from ml4ht.data.util.data_frame_data_description import DataFrameDataDescription

from ml4h.TensorMap import TensorMap, Interpretation


class TensorMapSampleGetter:
    def __init__(
            self,
            tensor_maps_in: List[TensorMap],
            tensor_maps_out: List[TensorMap],
            augment: bool = False,
            return_path: bool = False,
    ):
        """
        A SampleGetter that is built from TensorMaps.
        This allows backwards compatibility with old TensorMaps.

        :param tensor_maps_in: tensor maps to use to get input data for a model
        :param tensor_maps_out: tensor maps to use to get output data for a model
        :param augment: whether to apply the TensorMaps' data augmentations.
                        This should be true for training and false for testing.

        Example preparing TensorMaps for use in ml4h model:
        ```
        import torch

        from ml4h.tensormap.ecg import ecg_rest_raw
        from ml4h.tensormap.ecg import ecg_rest_age
        from ml4h.ml4ht_integration.tensor_map import TensorMapSampleGetter
        from ml4h.models.model_factory import block_make_multimodal_multitask_model

        from ml4ht.data.data_loader import numpy_collate_fn, SampleGetterDataset

        model = block_make_multimodal_multitask_model(
            [ecg_rest_raw],
            [ecg_rest_age],
            ...
        )

        sg = TensorMapSampleGetter(
            [ecg_rest_raw],
            [ecg_rest_age],
            True,
        )
        dset = SampleGetterDataset(
            [1_path_to_hd5.hd5, 2_path_to_hd5.hd5, ...],
            sg,
        )
        data_loader = torch.utils.data.DataLoader(
            dset, batch_size=128, collate_fn=numpy_collate_fn,
        )

        model.fit(iter(data_loader))  # train for one epoch
        ```
        """
        self.tensor_maps_in = tensor_maps_in
        self.tensor_maps_out = tensor_maps_out
        self.augment = augment
        self.return_path = return_path

    def __call__(self, path: str) -> Batch:
        dependents = {}
        with h5py.File(path, 'r') as hd5:
            in_batch = {}
            for tm in self.tensor_maps_in:
                in_batch[tm.input_name()] = tm.postprocess_tensor(
                    tm.tensor_from_file(tm, hd5, dependents),
                    augment=self.augment, hd5=hd5,
                )
            out_batch = {}
            for tm in self.tensor_maps_out:
                out_batch[tm.output_name()] = tm.postprocess_tensor(
                    tm.tensor_from_file(tm, hd5, dependents),
                    augment=self.augment, hd5=hd5,
                )
        if self.return_path:
            return in_batch, out_batch, path
        return in_batch, out_batch


def _not_implemented_tensor_from_file(_, __, ___=None):
    """Used to make sure TensorMap is never used to load data"""
    raise NotImplementedError('This TensorMap cannot load data.')


def tensor_map_from_data_description(
        data_description: DataDescription,
        interpretation: Interpretation,
        shape,
        name=None,
        **tensor_map_kwargs,
) -> TensorMap:
    """
    Allows a DataDescription to be used in the model factory
    by converting a DataDescription into a TensorMap

    :param data_description: Will be converted into a TensorMap
    :param interpretation: How the model factory will interpret
                           the data from the data_description
    :param shape: The shape of the data from the data_description
    :param name: The name of the TensorMap
    :param tensor_map_kwargs: keyword arguments passed to TensorMap.__init__
    :return: A TensorMap created from a DataDescription

    Example using an ECG and a sex data description to build an ml4h model:
    ```
    ecg_data_description = ...
    sex_data_description = ...
    ecg_tmap = tensor_map_from_data_description(
        ecg_data_description,
        Interpretation.CONTINUOUS,
        shape=(5000, 12),
    )
    sex_tmap = tensor_map_from_data_description(
        sex_data_description,
        Interpretation.CATEGORICAL,
        shape=(2,),
    )
    model = block_make_multimodal_multitask_model(
        tensor_maps_in=[ecg_tmap],
        tensor_maps_out=[sex_tmap],
        ...
    )
    ```
    """
    tmap = TensorMap(
        name=name if name else data_description.name,
        interpretation=interpretation,
        shape=shape,
        tensor_from_file=_not_implemented_tensor_from_file,
        **tensor_map_kwargs,
    )
    return tmap


def _one_hot(x):
    return np.array([1, 0], dtype=np.float32) if x == 0 else np.array([0, 1], dtype=np.float32)


def dataframe_data_description_from_tensor_map(
        tensor_map: TensorMap,
        dataframe: pd.DataFrame,
        is_input: bool = False,
) -> DataDescription:
    return DataFrameDataDescription(
        dataframe,
        col=tensor_map.name,
        name=tensor_map.input_name() if is_input else tensor_map.output_name(),
        value_to_tensor = _one_hot if tensor_map.is_categorical() else None,

    )
