import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

output_folder = '/afs/csail.mit.edu/u/n/ndiamant/public_html/'

df = pd.read_csv('/storage/ndiamant/ml/explorations/ecg_vae_cohort_09-15/tensors_all_union.csv')
print(df.describe())

plt.figure(figsize=(7, 7))
sns.kdeplot(df['partners_ecg_age_newest'], fill=True)
plt.xlabel('Age')
plt.savefig(os.path.join(output_folder, f'age_distribution.png',), dpi=300)

plt.figure(figsize=(7, 7))
sns.kdeplot(df['partners_ecg_rate_md_newest'], fill=True)
plt.xlabel('HR')
plt.savefig(os.path.join(output_folder, f'hr_distribution.png',), dpi=300)


