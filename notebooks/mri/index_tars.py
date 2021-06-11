# %%
import glob
import tarfile
import pandas as pd

# %%

tars = glob.glob('/mnt/disks/segmented-sax-v20201203-lax-v20201122/tar/*.tar.gz')
tars_df = pd.DataFrame({'dicom_file': [], 'tar_file': []})
for tar in sorted(tars):
    print(tar)
    with tarfile.open(tar, 'r:gz') as tar_info:
        file_list = tar_info.extractfile('file.list')
        df_list = pd.read_csv(file_list, names=['dicom_file'])
        df_list = df_list[:-1]
        df_list['tar_file'] = tar
        tars_df = pd.concat([tars_df, df_list])

# %%
tars_df.to_csv('map_tar_to_dcm.csv', index=False)
# %%
tars_df = pd.read_csv('/home/pdiachil/projects/manifests/map_tar_to_dcm.csv')
manifest = pd.read_csv('/home/pdiachil/projects/manifests/manifest_sax_4ch.tsv', sep='\t',
                       usecols=['sample_id', 'instance', 'dicom_file', 'series', 'instance_number'])
# %%
tars_df['dicom_file'] = tars_df['dicom_file'].str.replace('.png.mask.png', '')
manifest_tars = manifest.merge(tars_df, on='dicom_file', how='left')
# %%
manifest_tars.to_csv('/home/pdiachil/projects/manifests/manifest_sax_4ch_tar.csv', index=False)
# %%
