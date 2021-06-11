# %%
import pandas as pd

df_4ch = pd.read_csv('/home/pdiachil/projects/manifests/manifest_4ch.tsv', sep='\t', usecols=['sample_id', 'instance', 'dicom_file', 'series', 'instance_number'])
df_sax = pd.read_csv('/home/pdiachil/projects/manifests/manifest_sax.tsv', sep='\t', usecols=['sample_id', 'instance', 'dicom_file', 'series', 'instance_number'])
df_petersen = pd.read_csv('/home/pdiachil/returned_lv_mass.tsv', sep='\t')

df_4ch = df_petersen[['sample_id']].merge(df_4ch, on='sample_id')
df_sax = df_petersen[['sample_id']].merge(df_sax, on='sample_id')
# %%
df_sax_4ch = pd.concat([df_4ch, df_sax])
df_sax_4ch.to_csv('/home/pdiachil/projects/manifests/manifest_sax_4ch_petersen.csv', index=False)
# %%
import imageio

img = imageio.imread('/home/pdiachil/ml/tmp.png')
# %%
import matplotlib.pyplot as plt
plt.imshow(img)
# %%
len(df_sax_4ch.groupby('sample_id'))
# %%
img = imageio.imread('/home/pdiachil/1.3.12.2.1107.5.2.18.141243.2017042612121958067001384.dcm.png.mask.png')
# %%
plt.imshow(img==11)
# %%
