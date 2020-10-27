# %%
import pandas as pd
import numpy as np

df = pd.read_csv('/home/pdiachil/jon/instance0_ppg_vector.tsv', sep='\t')

# %%
from sklearn.decomposition import PCA

cols = [f't_{t:03d}' for t in range(2, 102)]
pcs = PCA(n_components=5)
pcs.fit(df[cols].values)
# %%
import matplotlib.pyplot as plt

f, ax = plt.subplots(5, 1)
for i, component in enumerate(pcs.components_):
    ax[i].plot(component)
# %%
# PCA via autoencoder
from tensorflow import keras
encoder = keras.models.Sequential([keras.layers.Dense(5, input_shape=[100]),
                                   # keras.layers.Dense(5, activation='selu'),
                                  ])
decoder = keras.models.Sequential([keras.layers.Dense(100, input_shape=[5]),
                                  ])
autoencoder = keras.models.Sequential([encoder, decoder])
autoencoder.compile(loss='mse', optimizer=keras.optimizers.SGD(lr=0.1))

# %%
from sklearn.preprocessing import StandardScaler
long_values = np.zeros((len(df[cols].values), 100))
long_values[:, :100] = df[cols].values
maxes = np.max(long_values, axis=1)
long_values = long_values[maxes<=10000]
scaler = StandardScaler()
long_values_scaled = scaler.fit_transform(long_values)
# long_values_scaled = long_values / 10000.
history = autoencoder.fit(long_values_scaled, long_values_scaled, epochs=6)
# %%
codings_pca = encoder.predict(long_values_scaled)
reconstructions_scaled = autoencoder.predict(long_values_scaled)
reconstructions_pca = scaler.inverse_transform(reconstructions_scaled)
# reconstructions_pca = reconstructions_scaled * 10000.
# %%
err = np.linalg.norm(reconstructions_pca - long_values, axis=1)
argmax = np.argsort(err)

# %%
f, ax = plt.subplots()
i = argmax[int(len(argmax)*0.9)]
ax.plot(long_values[i])
ax.plot(reconstructions_pca[i])

# # %%
# # Convolutional autoencoder
# conv_encoder = keras.models.Sequential([
#     keras.layers.Reshape([128, 1], input_shape=[128]),
#     keras.layers.Conv1D(2, kernel_size=7, padding='same', activation='selu'),
#     keras.layers.MaxPool1D(pool_size=2),
#     keras.layers.Conv1D(4, kernel_size=3, padding='same', activation='selu'),
#     keras.layers.MaxPool1D(pool_size=2),
#     keras.layers.Conv1D(8, kernel_size=3, padding='same', activation='selu'),
#     keras.layers.MaxPool1D(pool_size=2)
# ])

# conv_decoder = keras.models.Sequential([
#     keras.layers.UpSampling1D(size=2),
#     keras.layers.Conv1D(4, kernel_size=3, strides=1, padding='same', activation='selu'),
#     keras.layers.UpSampling1D(size=2),
#     keras.layers.Conv1D(2, kernel_size=3, strides=1, padding='same', activation='selu'),
#     keras.layers.UpSampling1D(size=2),
#     keras.layers.Conv1D(1, kernel_size=7, padding='same', activation='selu'),
#     keras.layers.Reshape([128])
# ])

# conv_autoencoder = keras.models.Sequential([conv_encoder, conv_decoder])
# conv_autoencoder.compile(loss='binary_crossentropy', optimizer=keras.optimizers.SGD(lr=0.015))
# conv_autoencoder.fit(long_values, long_values, epochs=20)

# # %%
# import matplotlib.pyplot as plt
# long_values = np.zeros((len(df[cols].values), 128))
# long_values[:, :100] = df[cols].values
# plt.plot(long_values[96728])
# # %%
# maxes = np.max(long_values, axis=1)
# %%
# Variational autoencoder
from tensorflow.keras import backend as K
import tensorflow as tf

class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean


coding_size = 5
inputs = keras.layers.Input(shape=[100])
z = keras.layers.Dense(100, activation='selu')(inputs)
z = keras.layers.Dense(50, activation='selu')(z)
codings_mean = keras.layers.Dense(coding_size)(z)
codings_log_var = keras.layers.Dense(coding_size)(z)
codings = Sampling()([codings_mean, codings_log_var])
variational_encoder = keras.Model(
    inputs=[inputs], outputs=[codings_mean, codings_log_var, codings],
)

decoder_inputs = keras.layers.Input(shape=[coding_size])
x = keras.layers.Dense(50, activation='selu')(decoder_inputs)
outputs = keras.layers.Dense(100, activation='sigmoid')(x)
variational_decoder = keras.Model(inputs=[decoder_inputs], outputs=[outputs])

_, _, codings = variational_encoder(inputs)
reconstructions = variational_decoder(codings)
variational_ae = keras.Model(inputs=[inputs], outputs=[reconstructions])

latent_loss = -0.5 * K.sum(
    1 + codings_log_var - K.exp(codings_log_var) - K.square(codings_mean),
    axis=-1,
)
variational_ae.add_loss(K.mean(latent_loss) / 100.)
variational_ae.compile(loss='binary_cross_entropy', optimizer=keras.optimizers.RMSprop(learning_rate=0.001))

history = variational_ae.fit(long_values/10000., long_values/10000., epochs=10, batch_size=32)
# %%
long_values_ae = long_values/10000.
reconstructions_ae = variational_ae.predict(long_values_ae)
codings_ae = variational_encoder.predict(long_values_ae)

# %%
err = np.linalg.norm(reconstructions_ae - long_values_ae, axis=1)
argmax = np.argsort(err)

# %%
f, ax = plt.subplots()
i = argmax[int(len(argmax)*0.95)]
ax.plot(long_values_ae[i])
ax.plot(reconstructions_ae[i])

# %%
df[maxes<=10000.][cols].values.shape
# %%
ppg_pcs = pd.DataFrame()
ppg_pcs['sample_id'] = df[maxes<=10000.]['app7089']
for i in range(5):
    ppg_pcs[f'pc_pca_{i}'] = codings_pca[:, i]
# for i in range(5):
#     ppg_pcs[f'pc_ae_{i}'] = codings_ae[0][:, i]
# %%
ppg_pcs.to_csv('/home/pdiachil/ml/notebooks/ppg/ppgs_pcs.csv', index=False)
# %%
ppg_pcs = pd.read_csv('/home/pdiachil/ml/notebooks/ppg/ppgs_pcs.csv')

# %%
cols_pca = [f'pc_pca_{i}' for i in range(5)]
cols_ae = [f'pc_ae_{i}' for i in range(5)]

for mm, (cols_model, cur_decoder) in enumerate(zip([cols_pca], [decoder,])):
    f, ax = plt.subplots(5, 3)
    f.set_size_inches(8, 4.5)
    avg = ppg_pcs[cols_model].values.mean(axis=0)
    std = ppg_pcs[cols_model].values.std(axis=0)
    for i, col_model in enumerate(cols_model):
        arr = np.zeros(5)
        arr[i] = 1.0
        tmp_codings_minus = avg - 1.0*std*arr
        tmp_codings_plus = avg + 1.0*std*arr
        trace_minus = cur_decoder.predict(tmp_codings_minus.reshape(1, -1))
        trace_plus = cur_decoder.predict(tmp_codings_plus.reshape(1, -1))
        if mm == 0:
            trace_minus = scaler.inverse_transform(trace_minus)
            trace_plus = scaler.inverse_transform(trace_plus)
        ax[i, 0].plot(trace_minus[0], color='black')
        ax[i, 1].plot(trace_plus[0], color='black')
        ax[i, 2].plot(trace_plus[0]-trace_minus[0], color='black')
        ax[i, 0].set_ylim([0.0, 11000.])
        ax[i, 0].set_yticks([0.0, 5000., 10000.])
        ax[i, 0].set_xticklabels([])
        ax[i, 1].set_xticklabels([])
        ax[i, 2].set_xticklabels([])
        ax[i, 1].set_ylim([0.0, 11000.])
        ax[i, 1].set_yticks([0.0, 5000., 10000.])
        ax[i, 1].set_yticklabels([])
        ax[i, 2].set_ylim([-5000.0, 5000.])
        ax[i, 0].set_xlim([0, 100.])
        ax[i, 1].set_xlim([0, 100.])
        ax[i, 2].set_xlim([0, 100.])
    ax[0, 0].set_title('$\mu - \sigma$')
    ax[0, 1].set_title('$\mu + \sigma$')
    ax[0, 2].set_title('Difference')
    ax[0, 0].set_ylabel('PC1')
    ax[1, 0].set_ylabel('PC2')
    ax[2, 0].set_ylabel('PC3')
    ax[3, 0].set_ylabel('PC4')
    ax[4, 0].set_ylabel('PC5')
    ax[4, 0].set_xticks([0, 100.])
    ax[4, 1].set_xticks([0, 100.])
    ax[4, 2].set_xticks([0, 100.])
    ax[4, 0].set_xticklabels(['0', '100'])
    ax[4, 1].set_xticklabels(['0', '100'])
    ax[4, 2].set_xticklabels(['0', '100'])
    ax[4, 0].set_xlabel('Samples')
    ax[4, 1].set_xlabel('Samples')
    ax[4, 2].set_xlabel('Samples')
    plt.tight_layout()
    f.savefig('PCs_linear_autoencoder.png', dpi=500)

# %%
