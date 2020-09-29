import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#FOLDER = 'train_runs/vae_deep_v1'
#MODEL_TYPE = 'vae'
#PREDICATE = 'ecg_vae_'
FOLDER = 'train_runs/ae_deep_v0'
MODEL_TYPE = 'autoencoder'
PREDICATE = 'ecg_ae_'
#FOLDER = 'train_runs/simclr_deep_v1'
#MODEL_TYPE = 'simclr'
#PREDICATE = 'ecg_simclr_'
#FOLDER = 'train_runs/supervised_deep_v0'
#MODEL_TYPE = 'supervised'
#PREDICATE = 'ecg_supervised_'


dfs = []
for f in filter(lambda x: x.startswith(PREDICATE), os.listdir(FOLDER)):
    path = os.path.join(FOLDER, f, 'metric_history.tsv')
    if not os.path.exists(path):
        print(f'Skipping {f}')
        continue
    df = pd.read_csv(path, sep='\t') 
    df['epoch'] = list(range(len(df)))
    num_latents = df['run_id'].iloc[0].replace(PREDICATE, '')
    df['latents'] = num_latents
    dfs.append(df)
df = pd.concat(dfs)


_, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax1.set_title(f'{MODEL_TYPE} training curves')
#sns.lineplot(data=df, x="epoch", y="mse", hue="latents", ax=ax1, legend='full')
#sns.lineplot(data=df, x="epoch", y="KL_divergence", hue="latents", ax=ax2, legend='full')
#sns.lineplot(data=df, x="epoch", y="loss", hue="latents", ax=ax1, legend='full')
#sns.lineplot(data=df, x="epoch", y="simclr_accuracy", hue="latents", ax=ax2, legend='full')
#sns.lineplot(data=df, x="epoch", y="loss", hue="latents", ax=ax1, legend='full')
#sns.lineplot(data=df, x="epoch", y="val_loss", hue="latents", ax=ax2, legend='full')
sns.lineplot(data=df, x="epoch", y="mse", hue="latents", ax=ax1, legend='full')
sns.lineplot(data=df, x="epoch", y="val_mse", hue="latents", ax=ax2, legend='full')
plt.savefig(f'/afs/csail.mit.edu/u/n/ndiamant/public_html/train_history_{MODEL_TYPE}s.png', dpi=300)

