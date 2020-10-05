import os
import sys
import time
import h5py
import numpy as np
from petastorm.codecs import NdarrayCodec, ScalarCodec
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import Unischema, UnischemaField, dict_to_spark_row
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType
from typing import Tuple, Optional, List

from ml4h.defines import TENSOR_EXT, StorageType
from ml4h.TensorMap import TensorMap, Interpretation


Shape = Tuple[Optional[int], ...]
DataDescription = Tuple[str, Shape, StorageType]


SYNTHETIC_DATA_PATH = os.path.join(os.path.dirname(__file__), 'synthetic_data')


def random_concrete_shape(shape: Shape) -> Tuple[int, ...]:
    return tuple(x if x is not None else 1 + np.random.randint(10) for x in shape)


def build_example(shape: Shape, storage_type: StorageType) -> np.ndarray:
    shape = random_concrete_shape(shape)
    if storage_type == StorageType.CONTINUOUS:
        return np.random.randn(*shape)
    if storage_type == StorageType.STRING:
        letters = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        return np.random.choice(letters, shape)
    else:
        raise NotImplementedError(f'Random generation not implemented for {storage_type}')


def _hd5_path_to_int(path: str) -> int:
    return int(os.path.basename(path).replace(TENSOR_EXT, ''))


def _int_to_hd5_path(i: int) -> str:
    return os.path.join(SYNTHETIC_DATA_PATH, f'{i}{TENSOR_EXT}')


def get_hd5_paths(overwrite: bool, num_hd5s: int) -> List[str]:
    if overwrite:
        to_write = list(range(num_hd5s))
    else:
        to_write = [i for i in range(num_hd5s) if not os.path.exists(_int_to_hd5_path(i))]
    return [_int_to_hd5_path(i) for i in to_write]


def write_in_hd5_ukbb(
    name: str, storage_type: StorageType, value: np.ndarray, hd5: h5py.File,
    compression,
):
    """Replicates storage behavior in tensor_writer_ukbb"""
    if storage_type == StorageType.STRING:
        hd5.create_dataset(name, data=value, dtype=h5py.special_dtype(vlen=str))
    elif storage_type == StorageType.CONTINUOUS:
        if value.size < 16:
            hd5.create_dataset(name, data=value)
        elif type(compression) == str:
            hd5.create_dataset(name, data=value, compression=compression)
        else:
            hd5.create_dataset(name, data=value, **compression)  # The hdf5plugin case
    else:
        raise NotImplementedError(f'{storage_type} cannot be automatically written yet')


def build_hd5s_ukbb(
        data_descriptions: List[DataDescription], num_hd5s: int,
        overwrite: bool = True, compression='gzip',
):
    paths = get_hd5_paths(overwrite, num_hd5s)
    start_time = time.time()
    print(f'Beginning to write {len(paths)} hd5s')
    for i, path in enumerate(paths):
        with h5py.File(path, 'w') as hd5:
            for name, shape, storage_type in data_descriptions:
                data = build_example(shape, storage_type)
                write_in_hd5_ukbb(name, storage_type, data, hd5, compression)
        print(f'Writing hd5s {(i + 1) / len(paths):.1%} done', end='\r')
        sys.stdout.flush()
    print()
    delta = time.time() - start_time
    print(f'Wrote {len(paths)} hd5s in {delta:.1f} seconds at {len(paths) / delta:.1f} paths/s')


STORAGE_TYPE_TO_INTERPRETATION = {
    StorageType.CONTINUOUS: Interpretation.CONTINUOUS,
    StorageType.STRING: Interpretation.LANGUAGE,
}


def build_tensor_maps(
    data_descriptions: List[DataDescription],
) -> List[TensorMap]:
    tmaps = []
    for name, shape, storage_type in data_descriptions:
        tmaps.append(
            TensorMap(
                name,
                interpretation=STORAGE_TYPE_TO_INTERPRETATION[storage_type],
                shape=shape,
            ),
        )
    return tmaps


def _pick_codec(shape: Shape, storage_type: StorageType):
    if storage_type == StorageType.CONTINUOUS:
        return NdarrayCodec()
    raise NotImplementedError(f'{storage_type} with shape {shape} not implemented for petastorm')


def _pick_dtype(shape: Shape, storage_type: StorageType):
    if storage_type == StorageType.CONTINUOUS:
        return np.float32
    raise NotImplementedError(f'{storage_type} with shape {shape} not implemented for petastorm')


def _build_petastorm_schema(data_descriptions: List[DataDescription]) -> Unischema:
    return Unischema(
        'my_schema',
        [UnischemaField('id', np.int32, (), ScalarCodec(IntegerType()))] + [
            UnischemaField(
                name, _pick_dtype(shape, storage_type), shape, _pick_codec(shape, storage_type))
            for name, shape, storage_type in data_descriptions
        ]
    )


def _petastorm_spark_row(data_descriptions: List[DataDescription], x: int):
    out = {
        name: build_example(shape, storage_type).astype(np.float32)
        for name, shape, storage_type in data_descriptions
    }
    out['id'] = x
    return out


def build_petastorm(
        data_descriptions: List[DataDescription], num_samples: int, output_url: str,
):
    spark = SparkSession.builder.config('spark.driver.memory', '2g').master('local[2]').getOrCreate()
    sc = spark.sparkContext
    schema = _build_petastorm_schema(data_descriptions)

    # rowgroup_size_mb = 256
    rowgroup_size_mb = 128
    with materialize_dataset(spark, output_url, schema, rowgroup_size_mb):
        start_time = time.time()
        print(f'Beginning to write {num_samples} samples using petastorm')
        rows_rdd = sc.parallelize(range(num_samples)) \
            .map(lambda x: _petastorm_spark_row(data_descriptions, x)) \
            .map(lambda x: dict_to_spark_row(schema, x))

        spark.createDataFrame(rows_rdd, schema.as_spark_schema()) \
            .coalesce(num_samples) \
            .write \
            .mode('overwrite') \
            .parquet(output_url)
        delta = time.time() - start_time
        print(f'Wrote {num_samples} samples in {delta:.1f} seconds at {num_samples / delta:.1f} samples/s')


def build_xarray(
    data_descriptions: List[DataDescription], num_samples: int,
):
    pass
