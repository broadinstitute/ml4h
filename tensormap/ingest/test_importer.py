import tensormap.ingest.importer
import tensormap.utils.ingest
import pytest
import numpy
import h5py
import pathlib
from contextlib import contextmanager


@contextmanager
def not_raises(exception):
  try:
    yield
  except exception:
    raise Exception


def test_check_testdata_1():
    cur_path = pathlib.Path(__file__).parent.absolute()
    target_path = str(cur_path) + '/../data/ingest/ecg/muse_ecg_deidentified_1.xml'

    with not_raises(IOError):
        tensormap.utils.ingest.open_read_load(target_path)


def test_check_bad_testdata_1():
    cur_path = pathlib.Path(__file__).parent.absolute()
    target_path = str(cur_path) + '/../nonsense.xml'

    with pytest.raises(IOError):
        tensormap.utils.ingest.open_read_load(target_path)


def test_check_testdata_2():
    cur_path = pathlib.Path(__file__).parent.absolute()
    target_path = str(cur_path) + '/../data/ingest/ecg/muse_ecg_deidentified_1.xml.zst'

    with not_raises(IOError):
        tensormap.utils.ingest.open_read_load(target_path)


def test_check_testdata_3():
    cur_path = pathlib.Path(__file__).parent.absolute()
    target_path = str(cur_path) + '/../data/ingest/ecg/ukbb_scrubbed.xml'

    with not_raises(IOError):
        tensormap.utils.ingest.open_read_load(target_path)


def test_check_testdata_4():
    cur_path = pathlib.Path(__file__).parent.absolute()
    target_path = str(cur_path) + '/../data/ingest/ecg/ukbb_scrubbed.xml.zst'

    with not_raises(IOError):
        tensormap.utils.ingest.open_read_load(target_path)


def test_check_testdata_2_decompress():
    cur_path = pathlib.Path(__file__).parent.absolute()
    target_path = str(cur_path) + '/../data/ingest/ecg/muse_ecg_deidentified_1.xml.zst'

    with not_raises(IOError):
        dat = tensormap.utils.ingest.open_read_load(target_path)

    with not_raises(ValueError):
        tensormap.utils.ingest.zstd_decompress(dat)


def test_check_testdata_2_decompress_bad():
    cur_path = pathlib.Path(__file__).parent.absolute()
    target_path = str(cur_path) + '/../data/ingest/ecg/muse_ecg_deidentified_1.xml'

    with not_raises(IOError):
        dat = tensormap.utils.ingest.open_read_load(target_path)

    with pytest.raises(ValueError):
        tensormap.utils.ingest.zstd_decompress(dat)


def test_check_testdata_4_decompress():
    cur_path = pathlib.Path(__file__).parent.absolute()
    target_path = str(cur_path) + '/../data/ingest/ecg/ukbb_scrubbed.xml.zst'

    with not_raises(IOError):
        dat = tensormap.utils.ingest.open_read_load(target_path)

    with not_raises(ValueError):
        tensormap.utils.ingest.zstd_decompress(dat)


def test_check_testdata_4_decompress_bad():
    cur_path = pathlib.Path(__file__).parent.absolute()
    target_path = str(cur_path) + '/../data/ingest/ecg/ukbb_scrubbed.xml'

    with not_raises(IOError):
        dat = tensormap.utils.ingest.open_read_load(target_path)

    with pytest.raises(ValueError):
        tensormap.utils.ingest.zstd_decompress(dat)