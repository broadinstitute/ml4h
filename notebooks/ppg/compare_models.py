# %%
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns

df = pd.read_csv('/home/pdiachil/projects/ppgs/instance0_notch_vector_paolo_121820.csv')
cols = [f't_{t:03d}' for t in range(2, 102)]

x = np.asarray(df[cols].values, dtype=np.float32)
y = np.asarray(df['absent_notch'].values, dtype=np.float32)

n_train = int(len(x) * 0.7)
n_val = int(len(x)*0.2)
n_test = len(x) - n_train - n_val

x_train = x[:n_train].reshape(-1, 100, 1)
y_train = y[:n_train]
x_val = x[n_train:n_train+n_val].reshape(-1, 100, 1)
y_val = y[n_train:n_train+n_val]
x_test = x[n_train+n_val:].reshape(-1, 100, 1)
y_test = y[n_train+n_val:]

print(
    "Number of samples in train and validation and test are %d and %d and %d."
    % (x_train.shape[0], x_val.shape[0], y_test.shape[0])
)

# %%
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))

batch_size = 32

train_dataset = (
    train_loader.shuffle(len(x_train))
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

test_dataset = (
    test_loader.shuffle(len(x_test))
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

initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# %%
import matplotlib.pyplot as plt
print("Evaluate on test data")
model.load_weights("/home/pdiachil/ml/notebooks/ppg/ppg_notch_classification_nomu_032021.h5")
# results = model.evaluate(x_test, y_test, batch_size=128)
# print("test loss, test acc:", results)

probs = model.output.op.inputs[0]
func = keras.backend.function([model.input], [probs])
probs_test1 = func([x_test.reshape(-1, 100, 1)])
f, ax = plt.subplots()
sns.distplot(probs_test1, ax=ax, kde=False)

# %%
print("Evaluate on test data")
model.load_weights("/home/pdiachil/ml/notebooks/ppg/ppg_notch_classification_yesmu_032021.h5")
# results = model.evaluate(x_test, y_test, batch_size=128)
# print("test loss, test acc:", results)

probs = model.output.op.inputs[0]
func = keras.backend.function([model.input], [probs])
probs_test2 = func([x_test.reshape(-1, 100, 1)])
f, ax = plt.subplots()
sns.distplot(probs_test2, ax=ax, kde=False)

# %%
print("Evaluate on test data")
model.load_weights("/home/pdiachil/ml/notebooks/ppg/ppg_notch_classification_smooth_032021.h5")
# results = model.evaluate(x_test, y_test, batch_size=128)
# print("test loss, test acc:", results)

probs = model.output.op.inputs[0]
func = keras.backend.function([model.input], [probs])
probs_test3 = func([x_test.reshape(-1, 100, 1)])
f, ax = plt.subplots()
sns.distplot(probs_test3, ax=ax, kde=False)

# %%
import seaborn as sns
sns.distplot(model.predict(x_test))

# %%
np.corrcoef(probs_test2[0].ravel(), probs_test3[0].ravel())

# %%
import scipy.stats as ss

def rank_to_normal(rank, c, n):
    # Standard quantile function
    x = (rank - c) / (n - 2*c + 1)
    return ss.norm.ppf(x)

c=3.0/8
rank = ss.rankdata(probs_test2, method="average")
rank = pd.Series(rank)
transformed = rank.apply(rank_to_normal, c=c, n=len(rank))

# %%
sorted_idxs = np.argsort(transformed)
f, ax = plt.subplots(5, 5)
j = 0
for i in range(99, 1, -4):
    pos = int(len(sorted_idxs)*i/100.)
    ax[j//5, j%5].plot(x_test[sorted_idxs[pos]], linewidth=3, color='black')
    label = 'abs.' if y_test[sorted_idxs[pos]] > 0.5 else 'pres.'
    title_str = f'{i}% | {probs_test2[0].ravel()[sorted_idxs[pos]]:.2f}'
    ax[j//5, j%5].set_title(title_str)
    ax[j//5, j%5].set_xticklabels([])
    ax[j//5, j%5].set_yticklabels([])

    j += 1
plt.tight_layout()
plt.savefig('ppg_notch_grade_resnet_mixup.png', dpi=500)
# %%
# %%
import scipy.stats as ss

def rank_to_normal(rank, c, n):
    # Standard quantile function
    x = (rank - c) / (n - 2*c + 1)
    return ss.norm.ppf(x)

c=3.0/8
rank = ss.rankdata(probs_test, method="average")
rank = pd.Series(rank)
transformed = rank.apply(rank_to_normal, c=c, n=len(rank))

# %%
sorted_idxs = np.argsort(transformed)
f, ax = plt.subplots(5, 5)
j = 0
for i in range(99, 1, -4):
    pos = int(len(sorted_idxs)*i/100.)
    ax[j//5, j%5].plot(x_test[sorted_idxs[pos]], linewidth=3, color='black')
    label = 'abs.' if y_test[sorted_idxs[pos]] > 0.5 else 'pres.'
    title_str = f'{i}% | {probs_test2[0].ravel()[sorted_idxs[pos]]:.2f}'
    ax[j//5, j%5].set_title(title_str)
    ax[j//5, j%5].set_xticklabels([])
    ax[j//5, j%5].set_yticklabels([])

    j += 1
plt.tight_layout()
plt.savefig('ppg_notch_grade_resnet_nomixup.png', dpi=500)

# %%
import scipy.stats as ss

def rank_to_normal(rank, c, n):
    # Standard quantile function
    x = (rank - c) / (n - 2*c + 1)
    return ss.norm.ppf(x)

c=3.0/8.0
rank = ss.rankdata(probs_test, method="average")
rank = pd.Series(rank)
transformed = rank.apply(rank_to_normal, c=c, n=len(rank))

# %%
sorted_idxs = np.argsort(transformed)
f, ax = plt.subplots(5, 5)
j = 0
for i in range(99, 1, -4):
    pos = int(len(sorted_idxs)*i/100.)
    ax[j//5, j%5].plot(x_test[sorted_idxs[pos]], linewidth=3, color='black')
    label = 'abs.' if y_test[sorted_idxs[pos]] > 0.5 else 'pres.'
    title_str = f'{i}% | {probs_test2[0].ravel()[sorted_idxs[pos]]:.2f}'
    ax[j//5, j%5].set_title(title_str)
    ax[j//5, j%5].set_xticklabels([])
    ax[j//5, j%5].set_yticklabels([])

    j += 1
plt.tight_layout()
plt.savefig('ppg_notch_grade_resnet_nomixup.png', dpi=500)