# %%
import pandas as pd
import numpy as np

petersen = pd.read_csv('/home/pdiachil/ml/notebooks/mri/returned_lv_mass.tsv', sep='\t')
vanilla = pd.read_csv('/home/pdiachil/ml/RV_processed_vanilla_v20201124_v20201122_0_1.csv')
lax = pd.read_csv('/home/pdiachil/ml/RV_processed_lax_v20201124_v20201122_0_1.csv')
separation = pd.read_csv('/home/pdiachil/ml/RV_processed_separation_v20201124_v20201122_0_1.csv')
petersen = petersen[petersen['sample_id']==4566955]
# %%
import matplotlib.pyplot as plt

cols = [f'RV_poisson_{t}' for t in range(50)]
f, ax = plt.subplots()
f.set_size_inches(3.5, 3)
t = range(0, 50)
ax.plot([t[0], t[-1]], [120.,  120.], '--', color='black', linewidth=3, label='Petersen')
ax.plot([t[0], t[-1]], [55., 55.], '--', color='black', linewidth=3)
ax.plot(t, vanilla.iloc[0][cols], linewidth=3, label='vanilla')
ax.set_xlabel('Frames')
ax.set_xlim([0, 49])
ax.set_ylabel('RV volume (ml)')
plt.legend()
plt.tight_layout()
plt.savefig('vanilla.png', dpi=500)
ax.plot(t, lax.iloc[0][cols], linewidth=3, label='LAX correction')
plt.legend()
plt.tight_layout()
plt.savefig('lax.png', dpi=500)
ax.plot(t, separation.iloc[0][cols], linewidth=3, label='LAX plane')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('separation.png', dpi=500)

# %%
