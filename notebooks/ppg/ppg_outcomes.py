# %%
import pandas as pd
import pandas as pd
import numpy as np
import scipy.stats as ss
import tensorflow as tf
from notebooks.genetics.outcome_association_utils import odds_ratios, hazard_ratios, plot_or_hr, unpack_disease, regression_model, random_forest_model, plot_rsquared_covariates

# %%
df = pd.read_csv('/home/pdiachil/projects/ppgs/instance0_notch_vector_paolo_121820.csv')
cols = [f't_{t:03d}' for t in range(2, 102)]

x = np.asarray(df[cols].values, dtype=np.float32)
y = np.asarray(df['absent_notch'].values, dtype=np.float32)
y_smooth = np.copy(y)
y_smooth[y_smooth <= 0.5] = 0.05
y_smooth[y_smooth > 0.5] = 0.95

n_train = int(len(x) * 0.7)
n_val = int(len(x)*0.2)
n_test = len(x) - n_train - n_val

x_train = x[:n_train].reshape(-1, 100, 1)
y_train = y[:n_train]
y_train_smooth = y_smooth[:n_train]
x_val = x[n_train:n_train+n_val].reshape(-1, 100, 1)
y_val = y[n_train:n_train+n_val]
y_val_smooth = y_smooth[n_train:n_train+n_val]
x_test = x[n_train+n_val:].reshape(-1, 100, 1)
y_test = y[n_train+n_val:]
y_test_smooth = y_smooth[n_train+n_val:]

sample_ids = df['app7089'].values
sample_ids_test = sample_ids[n_train+n_val:]

print(
    "Number of samples in train and validation and test are %d and %d and %d."
    % (x_train.shape[0], x_val.shape[0], y_test.shape[0])
)

# %%
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_loader_smooth = tf.data.Dataset.from_tensor_slices((x_train, y_train_smooth))
validation_loader_smooth = tf.data.Dataset.from_tensor_slices((x_val, y_val_smooth))
test_loader_smooth = tf.data.Dataset.from_tensor_slices((x_test, y_test_smooth))

batch_size = 32

train_dataset = (
    train_loader.shuffle(len(x_train))
    .batch(batch_size)
    .prefetch(10)
)

train_dataset_smooth = (
    train_loader_smooth.shuffle(len(x_train))
    .batch(batch_size)
    .prefetch(10)
)

train_dataset_one = (
    train_loader.shuffle(len(x_train))
    .batch(batch_size)
    .prefetch(10)
)

train_dataset_two = (
    train_loader.shuffle(len(x_train))
    .batch(batch_size)
    .prefetch(10)
)

train_dataset_zip = tf.data.Dataset.zip((train_dataset_one, train_dataset_two))

validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .batch(batch_size)
    .prefetch(10)
)

validation_dataset_smooth = (
    validation_loader_smooth.shuffle(len(x_val))
    .batch(batch_size)
    .prefetch(10)
)

test_dataset = (
    test_loader.shuffle(len(x_test))
    .batch(batch_size)
    .prefetch(10)
)

test_dataset_smooth = (
    test_loader_smooth.shuffle(len(x_test))
    .batch(batch_size)
    .prefetch(10)
)

# %%
from sklearn.decomposition import PCA
from tensorflow import keras

# Residual unit
class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation='relu', **kwargs):
        super().__init__(**kwargs)        
        self.activation = activation
        self.filters = filters
        self.strides = strides
        self.main_layers = [
            keras.layers.Conv1D(filters, 3, strides=strides, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.activations.get(self.activation),
            keras.layers.Conv1D(filters, 3, strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv1D(filters, 1, strides=strides, padding='same', use_bias=False),
                keras.layers.BatchNormalization()]       

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'activation': self.activation,
            'filters': self.filters,
            'strides': self.strides
        })
        return config
            
    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return keras.activations.get(self.activation)(Z + skip_Z)

model = keras.models.Sequential()
model.add(keras.layers.Conv1D(64, 7, strides=2, input_shape=[100, 1], padding='same', use_bias=False))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'))
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1, activation='sigmoid'))

# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "ppg_notch_classification_nomu_032021.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

probs = model.output.op.inputs[0]
func = keras.backend.function([model.input], [probs])


# %%
model_weights = {
    'resnet': 'ppg_notch_classification_nomu_032021.h5',
    'mu_02': 'ppg_notch_classification_yesmu_032021.h5',
    'mu_04': 'ppg_notch_classification_yesmu_alpha04_032022.h5',
    'smooth': 'ppg_notch_classification_smooth_032021.h5'
}


def rank_to_normal(rank, c, n):
    # Standard quantile function
    x = (rank - c) / (n - 2*c + 1)
    return ss.norm.ppf(x)

dic_probs = {'app7089': sample_ids, 'absent_notch': df['absent_notch'].values[:]}

for model_type, weights in model_weights.items():
    model.load_weights(weights)
    probs = model.output.op.inputs[0]
    func = keras.backend.function([model.input], [probs])
    probs_test = func([x.reshape(-1, 100, 1)])
    c=3.0/8
    rank = ss.rankdata(probs_test, method="average")
    rank = pd.Series(rank)
    transformed = rank.apply(rank_to_normal, c=c, n=len(rank))
    dic_probs[model_type] = transformed.values
    dic_probs[f'{model_type}_raw'] = probs_test[0].ravel()

# %%
model.load_weights(model_weights['resnet'])
dic_probs['resnet_categorical'] = model.predict(x.reshape(-1, 100, 1)).ravel()


# %%
outcomes = pd.read_csv('ppg_mi_cad_outcomes.csv')
df_probs = pd.DataFrame(dic_probs)
df_probs['ensemble_raw'] = df_probs[['resnet', 'mu_02', 'smooth']].mean(axis=1)
rank = ss.rankdata(df_probs['ensemble_raw'], method="average")
rank = pd.Series(rank)
transformed = rank.apply(rank_to_normal, c=c, n=len(rank))
df_probs['ensemble'] = transformed
outcomes = outcomes.merge(df_probs, on='app7089')
# outcomes['resnet_categorical'] = (outcomes['resnet'] > 0.5).apply(float)
outcomes['mi_cad_prevalent'] = outcomes['mi_cad_prev']
outcomes['sample_id'] = outcomes['app7089']
outcomes['age'] = outcomes['enroll_age']
outcomes['instance0_date'] = pd.to_datetime('2/1/2000')
outcomes['mi_cad_censor_date'] = pd.to_datetime(outcomes['instance0_date'] + pd.to_timedelta(outcomes['mi_cad_survival']*365, unit='d'))
outcomes.loc[outcomes['mi_cad_censor_date'].isna(), 'mi_cad_censor_date'] = pd.to_datetime('1/1/2000')
outcomes.loc[outcomes['mi_cad_incident'].isna(), 'mi_cad_incident'] = 0.0
# outcomes['mi_cad_incident'] = outcomes['mi_cad_incident'].apply(int)
outcomes = outcomes.dropna(subset=['mi_cad_prevalent'])

# %%
import seaborn as sns
import matplotlib.pyplot as plt

label_dic = {    
    'resnet': ['ML4H (ranking)',''],
    'mu_02': ['ML4H Mixup ($\\alpha$=0.2)',''],
    'mu_04': ['ML4H Mixup ($\\alpha$=0.4)',''],
    'smooth': ['ML4H Smoothing',''],
    'ensemble': ['ML4H Ensemble', '']
}

f, ax = plt.subplots()
sns.heatmap(outcomes[[label for label in label_dic]].corr(), 
            annot=True, cmap='gray', cbar=False)
ax.set_xticklabels([item[0] for label, item in label_dic.items()], rotation=45, ha='right')
ax.set_yticklabels([item[0] for label, item in label_dic.items()], rotation=0)
plt.tight_layout()
f.savefig('ranking_correlation.png', dpi=500)
# %%
label_dic = {
    'absent_notch': ['UKB (categorical)', ''],
    'resnet_categorical': ['ML4H (categorical)',''],
    'resnet': ['ML4H (ranking)',''],
    'mu_02': ['ML4H Mixup ($\\alpha$=0.2)',''],
    'mu_04': ['ML4H Mixup ($\\alpha$=0.4)',''],
    'smooth': ['ML4H Smoothing',''],
    'ensemble': ['ML4H Ensemble', ''],   
    'age': ['age', '']
}

dont_scale=['absent_notch', 'resnet_categorical']
odds_ratios_age = odds_ratios(
    outcomes, 
    outcomes, label_dic,
    [['mi_cad', 'MI']], covariates=[], instance=0, dont_scale=dont_scale,
)

hazard_ratios_age = hazard_ratios(
    outcomes, 
    outcomes, label_dic,
    [['mi_cad', 'MI']], covariates=[], instance=0, dont_scale=dont_scale,
)

plot_or_hr(odds_ratios_age, label_dic, [['mi_cad', 'MI']], f'or', occ='prevalent', horizontal_line_y=1.5)
# %%
plot_or_hr(hazard_ratios_age, label_dic, [['mi_cad', 'MI']], f'or', occ='incident', horizontal_line_y=1.5)
# %%
cv_dic = {}
strat_dic = {'resnet': '', 
             'mixup': 'mu_', 
             'smooth': 'smooth_'}
for split_id in range(5):
    cv_dic[split_id] = {}
    split = np.load(f'splits_{split_id}.npz')
    test_index = split['test_index']        
    for train_strat, train_strat_str in strat_dic.items():
        model.load_weights(f'ppg_notch_classification_042021_{train_strat_str}split{split_id}.h5')        
        cv_dic[split_id]['sample_id'] = sample_ids[test_index]
        cv_dic[split_id][f'{train_strat}_categorical'] = model.predict(x[test_index].reshape(-1, 100, 1)).ravel()
        probs = model.output.op.inputs[0]
        func = keras.backend.function([model.input], [probs])
        probs_test = func([x[test_index].reshape(-1, 100, 1)])
        # c=3.0/8
        # rank = ss.rankdata(probs_test, method="average")
        # rank = pd.Series(rank)
        # transformed = rank.apply(rank_to_normal, c=c, n=len(rank))
        cv_dic[split_id][f'{train_strat}_continuous'] = probs_test[0].ravel()
# %%
cv_df_dic = {'sample_id': []}
for strat in strat_dic:
    cv_df_dic[f'{strat}_continuous'] = []
    cv_df_dic[f'{strat}_categorical'] = []
for split_id in range(5):
    cv_df_dic['sample_id'].extend(cv_dic[split_id]['sample_id'].tolist())
    for strat in strat_dic:
        cv_df_dic[f'{strat}_continuous'].extend(cv_dic[split_id][f'{strat}_continuous'].tolist())
        cv_df_dic[f'{strat}_categorical'].extend(cv_dic[split_id][f'{strat}_categorical'].tolist())

# %%
def rank_to_normal(rank, c, n):
    # Standard quantile function
    x = (rank - c) / (n - 2*c + 1)
    return ss.norm.ppf(x)

cv_df = pd.DataFrame(cv_df_dic)
cv_df['ensemble_continuous'] = (cv_df['resnet_continuous'] + cv_df['mixup_continuous'] + cv_df['smooth_continuous']) / 3.0
c=3.0/8
rank = ss.rankdata(cv_df['ensemble_continuous'], method="average")
rank = pd.Series(rank)
transformed = rank.apply(rank_to_normal, c=c, n=len(rank))
cv_df['ensemble_continuous'] = transformed
# %%
outcomes = pd.read_csv('ppg_mi_cad_outcomes.csv')
outcomes['sample_id'] = outcomes['app7089']
outcomes = outcomes.merge(cv_df, on='sample_id')
outcomes = outcomes.merge(df[['app7089', 'absent_notch']], on='app7089')
# outcomes['resnet_categorical'] = (outcomes['resnet'] > 0.5).apply(float)
outcomes['mi_cad_prevalent'] = outcomes['mi_cad_prev']
outcomes['age'] = outcomes['enroll_age']
outcomes['instance0_date'] = pd.to_datetime('2/1/2000')
outcomes['mi_cad_censor_date'] = pd.to_datetime(outcomes['instance0_date'] + pd.to_timedelta(outcomes['mi_cad_survival']*365, unit='d'))
outcomes.loc[outcomes['mi_cad_censor_date'].isna(), 'mi_cad_censor_date'] = pd.to_datetime('1/1/2000')
outcomes.loc[outcomes['mi_cad_incident'].isna(), 'mi_cad_incident'] = 0.0
# outcomes['mi_cad_incident'] = outcomes['mi_cad_incident'].apply(int)
outcomes = outcomes.dropna(subset=['mi_cad_prevalent'])
# %%
label_dic = {
    'absent_notch': ['UKB (categorical)', ''],
    'resnet_categorical': ['ML4H (categorical)',''],
    'ensemble_continuous': ['ML4H (ranking)', ''],
    'resnet_continuous': ['ML4H (ranking)',''],
    'mixup_continuous': ['ML4H (MU ranking)',''],
    'smooth_continuous': ['ML4H (SM ranking)',''],
    'mixup_categorical': ['ML4H (MU categorical)',''],
    'smooth_categorical': ['ML4H (SM categorical)',''],
    'age': ['age', '']
}

dont_scale=['absent_notch', 'resnet_categorical']
odds_ratios_age = odds_ratios(
    outcomes, 
    outcomes, label_dic,
    [['mi_cad', 'MI']], covariates=[], instance=0, dont_scale=dont_scale,
)

hazard_ratios_age = hazard_ratios(
    outcomes, 
    outcomes, label_dic,
    [['mi_cad', 'MI']], covariates=[], instance=0, dont_scale=dont_scale,
)

plot_or_hr(odds_ratios_age, label_dic, [['mi_cad', 'MI']], f'or_cv', occ='prevalent', horizontal_line_y=1.5)
# %%
plot_or_hr(hazard_ratios_age, label_dic, [['mi_cad', 'MI']], f'hr_cv', occ='incident', horizontal_line_y=1.5)
# %%
from sklearn.metrics import confusion_matrix
confusion_matrix(outcomes['absent_notch']>0.5, outcomes['resnet_categorical']>0.5)
# %%
fn = (outcomes['absent_notch']>=0.5) & (outcomes['resnet_categorical']<0.5)
fp = (outcomes['absent_notch']<0.5) & (outcomes['resnet_categorical']>=0.5)
outcomes[fn]
# %%
f, ax = plt.subplots(1, 3)
ax[0].plot(df[df['app7089'] == outcomes[fn].iloc[0]['app7089']][cols].values.ravel(), 'k', linewidth=3)
ax[1].plot(df[df['app7089'] == outcomes[fn].iloc[1]['app7089']][cols].values.ravel(), 'k', linewidth=3)
ax[2].plot(df[df['app7089'] == outcomes[fn].iloc[2]['app7089']][cols].values.ravel(), 'k', linewidth=3)
ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
ax[2].set_xticklabels([])
ax[0].set_yticklabels([])
ax[1].set_yticklabels([])
ax[2].set_yticklabels([])
f.savefig('FN.png')
# %%
f, ax = plt.subplots(1, 3)
ax[0].plot(df[df['app7089'] == outcomes[fp].iloc[3]['app7089']][cols].values.ravel(), 'k', linewidth=3)
ax[1].plot(df[df['app7089'] == outcomes[fp].iloc[1]['app7089']][cols].values.ravel(), 'k', linewidth=3)
ax[2].plot(df[df['app7089'] == outcomes[fp].iloc[2]['app7089']][cols].values.ravel(), 'k', linewidth=3)
ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
ax[2].set_xticklabels([])
ax[0].set_yticklabels([])
ax[1].set_yticklabels([])
ax[2].set_yticklabels([])
f.savefig('FP.png')
# %%
