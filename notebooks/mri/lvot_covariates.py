# %%
import pandas as pd

covariates = pd.read_csv('/home/pdiachil/projects/lvot/covariates000000000000.csv')
covariates
# %%
covariates[covariates['px_height_mm']==covariates['px_width_mm']]
# %%
