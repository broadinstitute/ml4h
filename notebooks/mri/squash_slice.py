# %%
import pandas as pd
import glob

df = pd.concat([pd.read_csv(csv) for csv in glob.glob('/home/pdiachil/projects/chambers/rv/*.csv')])



# %%
cols = [f'RV_poisson_{d}' for d in range(50)]
# %%
import matplotlib.pyplot as plt
sample_id = 1052364
f, ax = plt.subplots()
f.set_size_inches(4, 3)
ax.plot(df[df['sample_id']==sample_id][cols].values[0], linewidth=3, color='black')
ax.set_xlim([0, 50])
ax.set_xlabel('Frames')
ax.set_ylabel('RV volume (ml)')
plt.tight_layout()
plt.savefig(f'/home/pdiachil/projects/chambers/rv/volume_trace_{sample_id}.png', dpi=500)
# %%
sample_id = 1176025
f, ax = plt.subplots()
f.set_size_inches(4, 3)
ax.plot(df[df['sample_id']==sample_id][cols].values[0], linewidth=3, color='black')
ax.set_xlim([0, 50])
ax.set_xlabel('Frames')
ax.set_ylabel('RV volume (ml)')
plt.tight_layout()
plt.savefig(f'/home/pdiachil/projects/chambers/rv/volume_trace_{sample_id}.png', dpi=500)
# %%
# %%
sample_id = 3461576
f, ax = plt.subplots()
f.set_size_inches(4, 3)
ax.plot(df[df['sample_id']==sample_id][cols].values[0], linewidth=3, color='black')
ax.set_xlim([0, 50])
ax.set_xlabel('Frames')
ax.set_ylabel('RV volume (ml)')
plt.tight_layout()
plt.savefig(f'/home/pdiachil/projects/chambers/rv/volume_trace_{sample_id}.png', dpi=500)

# %%
sample_id = 4136803
f, ax = plt.subplots()
f.set_size_inches(4, 3)
ax.plot(df[df['sample_id']==sample_id][cols].values[0], linewidth=3, color='black')
ax.set_xlim([0, 50])
ax.set_xlabel('Frames')
ax.set_ylabel('RV volume (ml)')
plt.tight_layout()
plt.savefig(f'/home/pdiachil/projects/chambers/rv/volume_trace_{sample_id}.png', dpi=500)
# %%
