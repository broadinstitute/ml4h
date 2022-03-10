# %%
import pandas as pd
from google.cloud import storage

#%%

projections_dir = '/home/pdiachil/projects/strain/'

df = pd.DataFrame()
for i in range(0, 45000, 1000):
    df = pd.concat([df, pd.read_csv(f'{projections_dir}/df_strain_{i}_{i+999}.csv')])

# %%
import seaborn as sns
import numpy as np

sns.distplot(df['rep_time'], kde=False, bins=np.linspace(38, 42, 100))
# %%
from collections import defaultdict

circ_cols = defaultdict(dict)
rad_cols = defaultdict(dict)
for sl in ['bas', 'mid', 'api']:
    for sec in ['S', 'A', 'L', 'P']:
        circ_cols[sl][sec] = [f'circ_{sl}_{sec}_spline_{t}' for t in range(40)]
        rad_cols[sl][sec] = [f'rad_{sl}_{sec}_spline_{t}' for t in range(40)]
# %%
circ_bas_s = np.ma.masked_array(df[circ_cols['bas']['S']].values, mask=df[circ_cols['bas']['S']].values<0.0)
circ_bas_a = np.ma.masked_array(df[circ_cols['bas']['A']].values, mask=df[circ_cols['bas']['A']].values<0.0)
circ_bas_l = np.ma.masked_array(df[circ_cols['bas']['L']].values, mask=df[circ_cols['bas']['L']].values<0.0)
circ_bas_p = np.ma.masked_array(df[circ_cols['bas']['P']].values, mask=df[circ_cols['bas']['P']].values<0.0)
time_bas_to_peak_s = np.argmin(circ_bas_s, axis=1)*df['rep_time'].values
time_bas_to_peak_a = np.argmin(circ_bas_a, axis=1)*df['rep_time'].values
time_bas_to_peak_l = np.argmin(circ_bas_l, axis=1)*df['rep_time'].values
time_bas_to_peak_p = np.argmin(circ_bas_p, axis=1)*df['rep_time'].values

# %%
circ_mid_s = np.ma.masked_array(df[circ_cols['mid']['S']].values, mask=df[circ_cols['mid']['S']].values<0.0)
circ_mid_a = np.ma.masked_array(df[circ_cols['mid']['A']].values, mask=df[circ_cols['mid']['A']].values<0.0)
circ_mid_l = np.ma.masked_array(df[circ_cols['mid']['L']].values, mask=df[circ_cols['mid']['L']].values<0.0)
circ_mid_p = np.ma.masked_array(df[circ_cols['mid']['P']].values, mask=df[circ_cols['mid']['P']].values<0.0)
time_mid_to_peak_s = np.argmin(circ_mid_s, axis=1)#*df['rep_time'].values
time_mid_to_peak_a = np.argmin(circ_mid_a, axis=1)#*df['rep_time'].values
time_mid_to_peak_l = np.argmin(circ_mid_l, axis=1)#*df['rep_time'].values
time_mid_to_peak_p = np.argmin(circ_mid_p, axis=1)#*df['rep_time'].values


# %%
import matplotlib.pyplot as plt
f, ax = plt.subplots()
sns.distplot(time_bas_to_peak_s, bins=range(200, 500, 40), kde=False, ax=ax)
ax.set_xlabel('Time to peak [ms], circ-bas-s')

# %%
time_bas_to_peak_salp = np.hstack([
    time_bas_to_peak_s.reshape(-1, 1),
    time_bas_to_peak_a.reshape(-1, 1),
    time_bas_to_peak_l.reshape(-1, 1),
    time_bas_to_peak_p.reshape(-1, 1)
    ])

sns.distplot(np.std(time_bas_to_peak_salp, axis=1), kde=False, bins=range(0, 50, 1))

std_bas = np.std(time_bas_to_peak_salp, axis=1)
df['std_bas'] = std_bas

# %%
import os
import h5py
import logging
import pickle
import blosc
import PIL

mri_notebooks_dir = '/home/pdiachil/ml/notebooks/mri/'
storage_client = storage.Client('broad-ml4cvd')
bucket = storage_client.get_bucket('ml4cvd')

df_bas = df.sort_values('std_bas')
for i in range(1, 100, 1):
    row = df_bas.iloc[len(df_bas)//100*i]
    patient = int(row['sample_id'])
    prefix = f'mdrk/ukb_bulk/cardiac/ingested/BROAD_ml4h_mdrk__cardiac__{patient}'
    blobs = storage_client.list_blobs(bucket, prefix=prefix)
        
    blob_cnt = 0
    for blob in blobs:
        filename = os.path.basename(blob.name)
        blob.download_to_filename(os.path.join(mri_notebooks_dir, filename))
        blob_cnt += 1

    hd5 = h5py.File(os.path.join(mri_notebooks_dir, filename), 'r')
    try:
        slices = hd5['field/20211/instance/2'].keys()
    except KeyError:
        logging.warning(f'Patient {patient} does not have strain images. Skipping')
        continue

    for sss, (slice, slice_name) in enumerate(zip(slices, ['bas', 'mid', 'api'])):
        serieses = hd5[f'field/20211/instance/2/{slice}/series'].keys()
        for series in serieses:
            if 'pandas' in series:
                df_slice = pickle.loads(blosc.decompress(
                        hd5[f'field/20211/instance/2/{slice}/series/{series}'][()]
                        ))
            else:
                img = np.frombuffer(blosc.decompress(
                        hd5[f'field/20211/instance/2/{slice}/series/{series}'][()]
                        ), 
                        dtype=np.uint16
                        )
        series = series.split('_')[0]
        sample = df_slice.sample(1)
        nrows = sample['Rows'].values[0]
        ncols = sample['Columns'].values[0]
        img = img.reshape(-1, nrows, ncols)

        layers = []
        for frame in range(img.shape[0]):
            f, ax = plt.subplots(1, 2)
            f.set_size_inches(12, 6)
            ax[0].imshow(img[frame, :, :], cmap='gray')
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            for sec in ['S', 'A', 'L', 'P']:
                masked = np.ma.masked_array(row[circ_cols['bas'][sec]], mask=row[circ_cols['bas'][sec]]<0.1)
                ax[1].plot(range(40), masked, label=sec)
            ax[1].legend()
            ax[1].set_title(f'Dissynchrony: {row["std_bas"]:.0f} ms')
            
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            f.canvas.draw()
            layers.append(PIL.Image.frombytes(
                'RGB', 
                f.canvas.get_width_height(), 
                f.canvas.tostring_rgb()
            ))
            plt.close(f)
        layers[0].save(f'{patient}_{slice_name}_{i}.gif', append_images=layers[1:], save_all=True, loop=0)
        break

    os.remove(os.path.join(mri_notebooks_dir, filename))


# %%
import matplotlib.pyplot as plt
f, ax = plt.subplots()
sns.distplot(time_bas_to_peak_s, bins=range(200, 500, 40), kde=False, ax=ax)
ax.set_xlabel('Time to peak [ms], circ-bas-s')

# %%
time_mid_to_peak_salp = np.hstack([
    time_mid_to_peak_s.reshape(-1, 1),
    time_mid_to_peak_a.reshape(-1, 1),
    time_mid_to_peak_l.reshape(-1, 1),
    time_mid_to_peak_p.reshape(-1, 1)
    ])

sns.distplot(np.std(time_mid_to_peak_salp, axis=1), kde=False, bins=range(1, 5, 1))
# %%
std1 = np.std(time_mid_to_peak_salp, axis=1) == 3
time_mid_to_peak_salp
# %%
f, ax = plt.subplots()
ax.plot(circ_mid_s[np.where(std1)[0][0]], '*')
ax.plot(circ_mid_a[np.where(std1)[0][0]])
ax.plot(circ_mid_l[np.where(std1)[0][0]])
ax.plot(circ_mid_p[np.where(std1)[0][0]])
# %%
