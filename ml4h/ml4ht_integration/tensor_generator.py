import logging
from typing import List
from collections import defaultdict

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from ml4ht.data.data_loader import SampleGetterIterableDataset, numpy_collate_fn, shuffle_get_epoch

from ml4h.TensorMap import TensorMap
from ml4h.defines import TensorGeneratorABC
from ml4h.ml4ht_integration.tensor_map import TensorMapSampleGetter


class TensorMapDataLoader(TensorGeneratorABC):
    def __init__(
        self, batch_size: int, input_maps: List[TensorMap], output_maps: List[TensorMap],
        paths: List[str], num_workers: int,
        keep_paths: bool = False,
        drop_last: bool = True,
        augment: bool = False,
        **kwargs,
    ):
        self.paths = paths
        self.input_maps = input_maps
        self.output_maps = output_maps
        self.keep_paths = keep_paths
        self.sample_getter = TensorMapSampleGetter(
            input_maps, output_maps, augment,
            return_path=keep_paths,
        )
        self.dset = SampleGetterIterableDataset(
            paths, self.sample_getter,
            get_epoch=shuffle_get_epoch,
        )
        self.data_loader = DataLoader(
            self.dset, batch_size=batch_size, num_workers=num_workers,
            collate_fn=self._collate_fn, drop_last=drop_last,
        )
        self.iter_loader = iter(self.data_loader)
        self.true_epochs = 0

    def _collate_fn(self, batches):
        if self.keep_paths:
            return numpy_collate_fn([batch[:2] for batch in batches]) + ([batch[2] for batch in batches],)
        return numpy_collate_fn(batches)

    @staticmethod
    def can_apply(paths, weights, mixup, siamese, **kwargs):
        """Can you substitute this TensorGenerator for the ml4h legacy TensorGenerator"""
        if isinstance(paths[0], list):
            raise NotImplementedError(
                "TensorMapDataLoader cannot sample from multiple lists of paths. Pass a list of paths for the paths argument",
            )
        if weights is not None:
            raise NotImplementedError(
                "TensorMapDataLoader cannot sample from multiple lists of paths. Do not pass 'weights' argument.",
            )
        if mixup:
            raise NotImplementedError("Mixup not implemented for TensorMapDataLoader")
        if siamese:
            raise NotImplementedError("Siamese not implemented for TensorMapDataLoader")

    def __iter__(self):
        return self

    def __next__(self):
        """Infinite iterator over data loader"""
        try:
            return next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self.data_loader)
            self.true_epochs += 1
            logging.info(f"Completed {self.true_epochs} true epochs.")
            return next(self.iter_loader)

    def __call__(self):
        try:
            next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self.data_loader)
        return self

    def kill_workers(self):
        """necessary for legacy compatibility"""
        pass


class TensorMapDataLoaderFromDataset(TensorGeneratorABC):
    def __init__(
            self, batch_size: int, input_maps, output_maps,
            dataset, num_workers: int,
            drop_last: bool = True,
            **kwargs,
    ):
        self.input_maps = input_maps
        self.output_maps = output_maps
        self.data_loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers,
            collate_fn=self._collate_fn, drop_last=drop_last,
        )
        self.iter_loader = iter(self.data_loader)
        self.true_epochs = 0

    def _collate_fn(self, batches):
        return numpy_collate_fn(batches)

    def __iter__(self):
        return self

    def __next__(self):
        """Infinite iterator over data loader"""
        try:
            return next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self.data_loader)
            self.true_epochs += 1
            logging.info(f"Completed {self.true_epochs} true epochs.")
            return next(self.iter_loader)

    def __call__(self):
        try:
            next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self.data_loader)
        return self


class TensorMapDataLoader2():
    def __init__(
        self, batch_size: int, input_maps, output_maps,
        dataset, num_workers: int,
        drop_last: bool = True,
        **kwargs,
    ):
        self.input_maps = input_maps
        self.output_maps = output_maps
        self.data_loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers,
            collate_fn=self._collate_fn, drop_last=drop_last,
        )
        self.iter_loader = iter(self.data_loader)
        self.true_iterations = 0

    def _collate_fn(self, batches):
        return numpy_collate_fn(batches)

    def __iter__(self):
        return self

    def __next__(self):
        """Infinite iterator over data loader"""
        try:
            return next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self.data_loader)
            self.true_iterations += 1
            print(f"Completed {self.true_iterations} true epochs.")
            return next(self.iter_loader)

    def __call__(self):
        try:
            next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self.data_loader)
        return self


def _datetime_to_float(d):
    return pd.to_datetime(d, utc=True).timestamp()


def _float_to_datetime(fl):
    return pd.to_datetime(fl, unit='s', utc=True)


def infer_from_dataloader(dataloader, model, tensor_maps_out, max_batches=125000):
    dataloader_iterator = iter(dataloader)
    space_dict = defaultdict(list)
    for i in range(max_batches):
        try:
            data, target = next(dataloader_iterator)
            for k in data:
                data[k] = np.array(data[k])
            prediction = model.predict(data, verbose=0)
            if len(model.output_names) == 1:
                prediction = [prediction]
            predictions_dict = {name: pred for name, pred in zip(model.output_names, prediction)}
            for b in range(prediction[0].shape[0]):
                for otm in tensor_maps_out:
                    y = predictions_dict[otm.output_name()]
                    if otm.is_categorical():
                        space_dict[f'{otm.name}_prediction'].append(y[b, 1])
                    elif otm.is_continuous():
                        space_dict[f'{otm.name}_prediction'].append(y[b, 0])
                    elif otm.is_survival_curve():
                        intervals = otm.shape[-1] // 2
                        days_per_bin = 1 + (2*otm.days_window) // intervals
                        predicted_survivals = np.cumprod(y[:, :intervals], axis=1)
                        space_dict[f'{otm.name}_prediction'].append(str(1 - predicted_survivals[0, -1]))
                        sick = np.sum(target[otm.output_name()][:, intervals:], axis=-1)
                        follow_up = np.cumsum(target[otm.output_name()][:, :intervals], axis=-1)[:, -1] * days_per_bin
                        space_dict[f'{otm.name}_event'].append(str(sick[0]))
                        space_dict[f'{otm.name}_follow_up'].append(str(follow_up[0]))
                for k in target:
                    if k in ['MRN', 'linker_id', 'is_c3po', 'output_age_in_days_continuous']:
                        space_dict[f'{k}'].append(target[k][b].numpy())
                    elif k in ['datetime']:
                        space_dict[f'{k}'].append(_float_to_datetime(int(target[k][b].numpy())))
                    else:
                        space_dict[f'{k}'].append(target[k][b, -1].numpy())
            if i % 100 == 0:
                print(f'Inferred on {i} batches, {len(space_dict[k])} rows')
        except StopIteration:
            print('loaded all batches')
            break
    return pd.DataFrame.from_dict(space_dict)
