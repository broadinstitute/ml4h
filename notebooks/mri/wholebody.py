# %%
import pandas as pd
import numpy as np
import h5py
import blosc

ff = h5py.File('/home/pdiachil/bananaman_1337.hdf5', 'r')
metadata = pd.read_parquet('/home/pdiachil/bananaman_1337.pq')
# %%
dat = ff['series/1'][()]
dat = np.frombuffer(blosc.decompress(dat), dtype=np.uint16).reshape(ff['series/1'].attrs['shape'])
# %%
pd.io.parquet.get_engine('auto')
# %%
