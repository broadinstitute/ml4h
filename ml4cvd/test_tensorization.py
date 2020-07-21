# %%
import h5py
ff = h5py.File('/home/pdiachil/common_tensors/1048581.hd5', 'r')

# %%
from ml4cvd.tensor_from_file import TMAPS

# %%
tm = TMAPS['c_lipidlowering']
tm.tensor_from_file(ff, tm)

# %%
