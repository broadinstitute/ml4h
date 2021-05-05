# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
import json
from datetime import datetime

# %%
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

df = pd.read_csv('/home/pdiachil/projects/ppgs/instance0_ppg_vector&label_paolo_050321.csv')
cols = [f't_{t:03d}' for t in range(2, 102)]

x = np.asarray(df[cols].values, dtype=np.float32)
y = np.asarray(df['absent_notch'].values, dtype=np.float32)
y_smooth = np.copy(y)
y_smooth[y_smooth <= 0.5] = 0.05
y_smooth[y_smooth > 0.5] = 0.95

# # %% 
# # Needed only once
# kf = KFold(n_splits=5, random_state=9, shuffle=True)
# for i, (train_index, test_index) in enumerate(kf.split(x)):
#     np.savez(f'splits_{i}.npz', train_index=train_index, test_index=test_index)
# %%
for split_id in range(5):
    split_id=0
    logdir = f"logs_split{split_id}_" + datetime.now().strftime("%Y%m%d-%H%M%S")    
    split = np.load(f'splits_{split_id}.npz')
    train_index = split['train_index']
    test_index = split['test_index']
    n_train = int(len(train_index) * 0.8)
    n_val = int(len(train_index) - n_train)
    n_test = int(len(test_index))
    
    x_train = x[train_index][:n_train].reshape(-1, 100, 1)
    y_train = y[train_index][:n_train]
    y_train_smooth = y_smooth[train_index][:n_train]
    x_val = x[train_index][n_train:].reshape(-1, 100, 1)
    y_val = y[train_index][n_train:]
    y_val_smooth = y_smooth[train_index][n_train:]
    x_test = x[test_index].reshape(-1, 100, 1)
    y_test = y[test_index]
    y_test_smooth = y_smooth[test_index]

    print(
        "Number of samples in train and validation and test are %d and %d and %d."
        % (x_train.shape[0], x_val.shape[0], y_test.shape[0])
    )


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

    def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
        gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
        gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
        return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


    def mix_up(ds_one, ds_two, alpha=0.2):
        # Unpack two datasets
        images_one, labels_one = ds_one
        images_two, labels_two = ds_two
        batch_size = tf.shape(images_one)[0]

        # Sample lambda and reshape it to do the mixup
        l = sample_beta_distribution(batch_size, alpha, alpha)
        x_l = tf.reshape(l, (batch_size, 1, 1,))
        y_l = tf.reshape(l, (batch_size,))

        # Perform mixup on both images and labels by combining a pair of images/labels
        # (one from each dataset) into one image/label
        images = images_one * x_l + images_two * (1 - x_l)
        labels = labels_one * y_l + labels_two * (1 - y_l)
        return (images, labels)

    # First create the new dataset using our `mix_up` utility
    train_dataset_mu = train_dataset_zip.map(
        lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=0.2)
    )

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


    # TRADITIONAL
    # Define callbacks.
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        f"ppg_notch_classification_050321_split{split_id}.h5", save_best_only=True
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    # Train the model, doing validation at the end of each epoch
    model.load_weights('initial_weights.h5')
    epochs = 100
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=True,
        callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_callback],
    )

    # # Mixup
    # # Define callbacks.
    # checkpoint_cb = keras.callbacks.ModelCheckpoint(
    #     f"ppg_notch_classification_042021_mu_split{split_id}.h5", save_best_only=True
    # )
    # early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir='mu_'+logdir)

    # # Train the model, doing validation at the end of each epoch
    # model.load_weights('initial_weights.h5')
    # epochs = 100
    # history = model.fit(
    #     train_dataset_mu,
    #     validation_data=validation_dataset,
    #     epochs=epochs,
    #     shuffle=True,
    #     callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_callback],
    # )

    # # Smooth
    # # Mixup
    # # Define callbacks.
    # checkpoint_cb = keras.callbacks.ModelCheckpoint(
    #     f"ppg_notch_classification_042021_smooth_split{split_id}.h5", save_best_only=True
    # )
    # early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir='smooth_'+logdir)

    # # Train the model, doing validation at the end of each epoch
    # model.load_weights('initial_weights.h5')
    # epochs = 100
    # history = model.fit(
    #     train_dataset_smooth,
    #     validation_data=validation_dataset_smooth,
    #     epochs=epochs,
    #     shuffle=True,
    #     callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_callback],
    # )

# # %%

# # Previous best model
# # model.load_weights("ppg_notch_classification.h5")


# # %%
# print("Evaluate on test data")
# model.load_weights("ppg_notch_classification_yesmu_032021.h5")
# results = model.evaluate(x_test, y_test, batch_size=128)
# print("test loss, test acc:", results)

# # %%
# print("Generate predictions for 3 samples")
# predictions = model.predict(x_test)
# print("predictions shape:", predictions.shape)


# # %%
# import seaborn as sns

# sns.distplot(predictions)

# # %%
# probs = model.output.op.inputs[0]
# func = keras.backend.function([model.input], [probs])
# probs_test = func([x_test.reshape(-1, 100, 1)])
# f, ax = plt.subplots()
# sns.distplot(probs_test, ax=ax, kde=False)
# f.savefig('/home/pdiachil/dist_yesmu.png', dpi=500)

# # %%

# def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
#     gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
#     gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
#     return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


# def mix_up(ds_one, ds_two, alpha=0.2):
#     # Unpack two datasets
#     images_one, labels_one = ds_one
#     images_two, labels_two = ds_two
#     batch_size = tf.shape(images_one)[0]

#     # Sample lambda and reshape it to do the mixup
#     l = sample_beta_distribution(batch_size, alpha, alpha)
#     x_l = tf.reshape(l, (batch_size, 1, 1,))
#     y_l = tf.reshape(l, (batch_size,))

#     # Perform mixup on both images and labels by combining a pair of images/labels
#     # (one from each dataset) into one image/label
#     images = images_one * x_l + images_two * (1 - x_l)
#     labels = labels_one * y_l + labels_two * (1 - y_l)
#     return (images, labels)

# # First create the new dataset using our `mix_up` utility
# train_dataset_mu = train_dataset_zip.map(
#     lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=0.4)
# )

# # Let's preview 9 samples from the dataset
# sample_images, sample_labels = next(iter(train_dataset_mu))
# plt.figure(figsize=(10, 10))
# for i, (image, label) in enumerate(zip(sample_images[:9], sample_labels[:9])):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.plot(image.numpy().squeeze())
#     print(label.numpy().tolist())
#     plt.axis("off")

# # # %%
# # import os
# # print(os.getcwd())
# # model.load_weights('/home/pdiachil/ml/notebooks/ppg/initial_weights.h5')
# # checkpoint_cb = keras.callbacks.ModelCheckpoint(
# #     "/home/pdiachil/ml/notebooks/ppg/ppg_notch_classification_yesmu_alpha04_032022.h5", save_best_only=True
# # )
# # epochs = 100
# # history = model.fit(
# #     train_dataset_mu,
# #     validation_data=validation_dataset,
# #     epochs=epochs,
# #     shuffle=True,
# #     callbacks=[checkpoint_cb, early_stopping_cb],
# # )


# # # %%
# # import os
# # print(os.getcwd())
# # model.load_weights('/home/pdiachil/ml/notebooks/ppg/initial_weights.h5')
# # checkpoint_cb = keras.callbacks.ModelCheckpoint(
# #     "/home/pdiachil/ml/notebooks/ppg/ppg_notch_classification_smooth_032021.h5", save_best_only=True
# # )
# # epochs = 100
# # history = model.fit(
# #     train_dataset_smooth,
# #     validation_data=validation_dataset_smooth,
# #     epochs=epochs,
# #     shuffle=True,
# #     callbacks=[checkpoint_cb, early_stopping_cb],
# # )
# # %%
# import scipy.stats as ss

# def rank_to_normal(rank, c, n):
#     # Standard quantile function
#     x = (rank - c) / (n - 2*c + 1)
#     return ss.norm.ppf(x)

# c=3.0/8
# rank = ss.rankdata(probs_test, method="average")
# rank = pd.Series(rank)
# transformed = rank.apply(rank_to_normal, c=c, n=len(rank))
# # %%
# sns.distplot(transformed)
# # %%
# sorted_idxs = np.argsort(transformed)
# f, ax = plt.subplots(5, 5)
# j = 0
# for i in range(99, 1, -4):
#     pos = int(len(sorted_idxs)*i/100.)
#     ax[j//5, j%5].plot(x[sorted_idxs[pos]], linewidth=3, color='black')
#     label = 'abs.' if y[sorted_idxs[pos]] > 0.5 else 'pres.'
#     title_str = f'{label} | {i}%'
#     ax[j//5, j%5].set_title(title_str)
#     ax[j//5, j%5].set_xticklabels([])
#     ax[j//5, j%5].set_yticklabels([])

#     j += 1
# plt.tight_layout()
# plt.savefig('/home/pdiachil/ppg_notch_grade_resnet_yesmu.png', dpi=500)


# # # %%
# # transformed
# # # %%
# # df['resnet_notch'] = probs_test[0]
# # df['resnet_notch_grade'] = transformed
# # # %%
# # # df.to_csv('/home/pdiachil/projects/ppgs/instance0_notch_vector_paolo_ml_grade_012121.csv')
# # # %%

# # %%

# %%
