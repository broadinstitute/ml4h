# %%
import pandas as pd
import glob
df = pd.concat([pd.read_csv(csv) for csv in glob.glob('/home/pdiachil/projects/chambers/rv/*.csv')])
df = df[df['sample_id']!=-1]
cols = [f'RV_poisson_{d}' for d in range(50)]
# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[cols])
pca = PCA(n_components=5)
pca.fit(df[cols])
f, ax = plt.subplots()
ax.bar(range(5), pca.explained_variance_ratio_)
print(np.sum(pca.explained_variance_ratio_))

# %%
f, ax = plt.subplots(5, 1)
f.set_size_inches(6, 4)
for i, component in enumerate(pca.components_):
    ax[i].plot(component, linewidth=3, color='black')
    ax[i].set_ylabel(f'PC {i+1}')
    ax[i].set_xticklabels([])
    ax[i].set_xticks(range(10, 51, 10))
    ax[i].set_xlim([0, 50])
ax[i].set_xticklabels(map(str, range(10, 51, 10)))
ax[i].set_xlabel('Frames')
plt.tight_layout()
f.savefig('RV_PCs.png', dpi=500)


# %%

# %%
import matplotlib.pyplot as plt
sample_id = 1052364
f, ax = plt.subplots()
f.set_size_inches(4, 3)
pcs = pca.transform(df[df['sample_id']==sample_id][cols].values[0].reshape(1, -1))
smoothed = pca.inverse_transform(pcs)
ax.plot(df[df['sample_id']==sample_id][cols].values[0], linewidth=3, color='black', label='raw')
ax.plot(smoothed[0], '--', linewidth=3, color='black', label='smoothed')
ax.set_xlim([0, 50])
ax.set_xlabel('Frames')
ax.set_ylabel('RV volume (ml)')
ax.legend()
plt.tight_layout()
plt.savefig(f'/home/pdiachil/projects/chambers/rv/volume_trace_{sample_id}.png', dpi=500)
# %%
sample_id = 1176025
f, ax = plt.subplots()
f.set_size_inches(4, 3)
pcs = pca.transform(df[df['sample_id']==sample_id][cols].values[0].reshape(1, -1))
smoothed = pca.inverse_transform(pcs)
ax.plot(df[df['sample_id']==sample_id][cols].values[0], linewidth=3, color='black', label='raw')
ax.plot(smoothed[0], '--', linewidth=3, color='black', label='smoothed')
ax.set_xlim([0, 50])
ax.set_xlabel('Frames')
ax.set_ylabel('RV volume (ml)')
ax.legend()
plt.tight_layout()
plt.savefig(f'/home/pdiachil/projects/chambers/rv/volume_trace_{sample_id}.png', dpi=500)
# %%
# %%
sample_id = 3461576
f, ax = plt.subplots()
f.set_size_inches(4, 3)
pcs = pca.transform(df[df['sample_id']==sample_id][cols].values[0].reshape(1, -1))
smoothed = pca.inverse_transform(pcs)
ax.plot(df[df['sample_id']==sample_id][cols].values[0], linewidth=3, color='black', label='raw')
ax.plot(smoothed[0], '--', linewidth=3, color='black', label='smoothed')
ax.set_xlim([0, 50])
ax.set_xlabel('Frames')
ax.set_ylabel('RV volume (ml)')
ax.legend()
plt.tight_layout()
plt.savefig(f'/home/pdiachil/projects/chambers/rv/volume_trace_{sample_id}.png', dpi=500)

# %%
sample_id = 4136803
f, ax = plt.subplots()
f.set_size_inches(4, 3)
pcs = pca.transform(df[df['sample_id']==sample_id][cols].values[0].reshape(1, -1))
smoothed = pca.inverse_transform(pcs)
ax.plot(df[df['sample_id']==sample_id][cols].values[0], linewidth=3, color='black', label='raw')
ax.plot(smoothed[0], '--', linewidth=3, color='black', label='smoothed')
ax.set_xlim([0, 50])
ax.set_xlabel('Frames')
ax.set_ylabel('RV volume (ml)')
ax.legend()
plt.tight_layout()
plt.savefig(f'/home/pdiachil/projects/chambers/rv/volume_trace_{sample_id}.png', dpi=500)
# %%
