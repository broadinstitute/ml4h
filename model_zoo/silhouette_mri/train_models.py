# ML4H is released under the following BSD 3-Clause License:
# Copyright (c) 2020, Broad Institute, Inc. and The General Hospital Corporation.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name Broad Institute, Inc. or The General Hospital Corporation
#   nor the names of its contributors may be used to endorse or promote products
#   derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import sys
import cv2
import blosc
import plyvel
import pandas as pd
import numpy as np
import os
import time
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    TensorBoard,
)
from ml4h.TensorMap import TensorMap
# External functionality maintained by MDRK
import callbacks
import shrinkage_loss

# Data sources
db_path = '/tf/silhouette/data/'
data_path = './'
db_path_i3 = '/tf/silhouette/data/contours_instance3_flattened/'
leveldb_source = os.path.join(db_path,"BROAD_ml4h_mdrk_dixon_contour_flattened_valid_shapes__leveldb__26f63ef10be14648aab657a6279258b0")

# Acquire fold and intrafold from sys arguments
fold = sys.argv[1]
intra_fold = sys.argv[2]
# Target model name
target_phenotype = "silhouette_asat_vat_gfat"

# Read k-fold data
folds = [pd.read_parquet(os.path.join(data_path,f'BROAD_ml4h_mdrk__dixon_mri_i2__data_fold{fold}__31e126c6eed64e8d9fa31c6262a19b8d.pq')) for fold in range(5)]
fold_layout = np.roll(np.arange(5),2*int(fold))
holdout_data = folds[fold_layout[-1]]

fold_layout = np.roll(fold_layout[0:4],1+int(intra_fold))
test_data = folds[fold_layout[-1]]
train_data = pd.concat([folds[f] for f in fold_layout[0:3]])

# Processing-specific
def leveldb_uncompress_given_shape(t,stored_dtype=np.uint8, shape=(256,237)):
    return np.frombuffer(blosc.decompress(t),dtype=stored_dtype).reshape(shape)


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result  = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def translate_image(image, shape, steps):
    M = np.float32([[1, 0, steps], [0, 1, steps]])
    image = cv2.warpAffine(image, M, shape)
    return image


def mdrk_silhouette_flattened_2d_side_by_side_1channel_leveldb(
    augment: bool = False,
):
    def _mdrk_silhouette_flattened_2d_side_by_side_1channel_leveldb(tensor):
        do_augment = False
        rand_angle = 0.0
        rand_move = 0.0
        if augment:
            do_augment = True
            rand_angle = np.random.randint(-5, 5)
            rand_move  = np.random.randint(-16, 16)
        #
        tensor = tensor.astype(np.float32)
        #
        if do_augment:
            tensor = translate_image(tensor, (237, 256), rand_move)
            tensor = rotate_image(tensor, rand_angle)
        #
        tensor = tensor[...,np.newaxis]
        return tensor
    return _mdrk_silhouette_flattened_2d_side_by_side_1channel_leveldb


actual_train_tm = TensorMap(
    "mdrk_silhouette",
    tensor_from_file=mdrk_silhouette_flattened_2d_side_by_side_1channel_leveldb(
        augment=True
    ),
    shape=(256, 237, 1),
    normalization=None,
)

actual_test_tm = TensorMap(
    "mdrk_silhouette",
    tensor_from_file=mdrk_silhouette_flattened_2d_side_by_side_1channel_leveldb(
        augment=False
    ),
    shape=(256, 237, 1),
    normalization=None,
)


# Data generator
class leveldb_generator:
    def __init__(self, path, tm, data_shape, stored_dtype=np.uint8, prefix: str = "input"):
        self.path = path
        self.db = plyvel.DB(path, create_if_missing=False)
        self.callbacks = []
        for t in tm:
            self.callbacks.append(t.tensor_from_file)
        self.tm = tm
        self.prefix = prefix
        self.data_shape = data_shape
        self.stored_dtype = stored_dtype
    #
    def __call__(self, path, tm_id):
        ret = dict()
        for t in self.tm:
            ret[self.prefix + "_" + t.name + f"_{t.interpretation.name.lower()}"] = list()
        #
        for p in path:
            data = leveldb_uncompress_given_shape(pickle.loads(self.db.get(str(p).encode())),
                shape=self.data_shape,stored_dtype=self.stored_dtype)
            #
            for t in tm_id:
                tm = self.tm[t]
                tensor = tm.tensor_from_file(data)
                if tm.normalization is not None:
                    tensor = tm.normalization.normalize(tensor)
                #
                ret[self.prefix + "_" + tm.name + f"_{tm.interpretation.name.lower()}"].append(
                    tensor
                )
            #
        yield ret


def pandas_load_wrapper_dict(df: pd.DataFrame, targets):
    """Wrapper function for retrieving records from an in-memory Pandas
    `DataFrame. Intended use is exclusively with the `tf.Data.DataSet` API
    for creating data generator pipeplines.

    Args:
        df (pd.DataFrame): Source Pandas DataFrame to retrieve records from.
        targets ([type]): Target projection names (column names) to retrieve data for.
    """
    def pandas_load_batch(batch):
        def internal(batch):
            # Returns (batch, dimension) variables and (batch,) labels
            # return list(df[targets].loc[batch].to_dict('list').values())
            # return np.array(list(df[targets].loc[batch].to_dict('list').values()))[...,np.newaxis]
            # return df[targets].loc[batch].T.values.tolist()
            return [np.array(a)[...,np.newaxis] for a in list(df[targets].loc[batch].to_dict('list').values())]
        #
        flattened = tf.py_function(internal, [batch], [tf.float32] * len(targets))
        for f in range(len(targets)):
            flattened[f].set_shape(tf.TensorShape([batch.shape[0], 1]))
        #
        ret = {k: v for k,v in zip(targets, flattened)}
        return ret
    #
    return pandas_load_batch


gen = leveldb_generator(
    leveldb_source,
    [actual_train_tm, actual_test_tm],
    data_shape=(256, 237),
)

# Train generator
train_body = (
    tf.data.Dataset.from_tensor_slices(train_data.index.values)
    .batch(32)
)
train_dx = train_body.interleave(
    lambda filename: tf.data.Dataset.from_generator(
        gen,  # Generator class
        ({"input_" + k.name + "_continuous": tf.float32 for k in [actual_train_tm]}),
        (
            {
                "input_"
                + k.name
                + "_continuous": tf.TensorShape(tuple([None] + list(k.shape)))
                for k in [actual_train_tm]
            }
        ),  # Output shape
        args=(
            filename,
            [0],
        ),  # Pass tf.tensor to execute as string with eager execution
    ),
)
train_dy = train_body.map(
    pandas_load_wrapper_dict(train_data, ['vat_ensemble','asat_ensemble','gfat_ensemble','vat_asat_ensemble']),
)
train_ds = tf.data.Dataset.zip((train_dx, train_dy)).prefetch(5)

# Test generator
test_body = (
    tf.data.Dataset.from_tensor_slices(test_data.index.values)
    .batch(32)
)
test_dx = test_body.interleave(
    lambda filename: tf.data.Dataset.from_generator(
        gen,  # Generator class
        ({"input_" + k.name + "_continuous": tf.float32 for k in [actual_test_tm]}),
        (
            {
                "input_"
                + k.name
                + "_continuous": tf.TensorShape(tuple([None] + list(k.shape)))
                for k in [actual_test_tm]
            }
        ),  # Output shape
        args=(
            filename,
            [1],
        ),  # Pass tf.tensor to execute as string with eager execution
    ),
)
test_dy = test_body.map(
    pandas_load_wrapper_dict(test_data, ['vat_ensemble','asat_ensemble','gfat_ensemble','vat_asat_ensemble']),
)
test_ds = tf.data.Dataset.zip((test_dx, test_dy)).prefetch(5)


METRICS = [
    tf.keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None),
    tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error", dtype=None),
]

# Custom model factory
input = tf.keras.Input(
    shape=(256, 237, 1),
    name="input_mdrk_silhouette_continuous",
)

# 121: [6, 12, 24, 16]
base_model = tf.keras.applications.DenseNet121(include_top=False, input_tensor=input)
for layer in base_model.layers:
    layer.trainable = False

x = tf.keras.layers.Dense(512,activation="relu",name="latent_space",kernel_regularizer=tf.keras.regularizers.l2(1e-4))(base_model.layers[-1].output)

arm1 = tf.keras.layers.Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
arm1 = tf.keras.layers.Dense(32, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(arm1)
output1 = tf.keras.layers.Dense(1, name="vat_ensemble", activation='linear')(arm1)

arm2 = tf.keras.layers.Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
arm2 = tf.keras.layers.Dense(32, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(arm2)
output2 = tf.keras.layers.Dense(1, name="asat_ensemble", activation='linear')(arm2)

arm3 = tf.keras.layers.Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
arm3 = tf.keras.layers.Dense(32, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(arm3)
output3 = tf.keras.layers.Dense(1, name="gfat_ensemble", activation='linear')(arm3)

arm4 = tf.keras.layers.concatenate([arm1,arm2])
arm4 = tf.keras.layers.Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(arm4)
arm4 = tf.keras.layers.Dense(32, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-4))(arm4)
output4 = tf.keras.layers.Dense(1, name="vat_asat_ensemble", activation='linear')(arm4)

# Model
model = tf.keras.models.Model(inputs=input, outputs=[output1,output2,output3,output4])

# Print model summary
model.summary()

########
loss = shrinkage_loss.ShrinkageLoss(a=10.0, c=0.2)
opt = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0)
model.compile(optimizer=opt, loss=loss, metrics=METRICS)

basedir = "/tf/models"
timestr = str(round(time.time()))
output_path = os.path.join(basedir, target_phenotype + "_" + timestr + '_fold' + str(fold) + '_intrafold' + str(intra_fold) + "_enemble_mri_sliding")
model_file = os.path.join(output_path, "model")
if not os.path.exists(os.path.dirname(model_file)):
    os.makedirs(os.path.dirname(model_file))


n_epochs = 50

decay = tf.keras.experimental.CosineDecay(
    initial_learning_rate=1e-4, decay_steps=n_epochs, alpha=0.0
)

loss_history = callbacks.LossHistory(decay)
lrate = tf.keras.callbacks.LearningRateScheduler(decay, verbose=1)

callback_list = [
    loss_history,
    lrate,
    ModelCheckpoint(
        filepath=model_file,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
    ),
    TensorBoard(
        log_dir=output_path,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq=10,
        profile_batch=0,
    ),
]

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=n_epochs,
    verbose=1,
    callbacks=callback_list,
)

history_min = min([len(l) for k, l in model.history.history.items()])
history_df = pd.DataFrame(
    {k: l[:history_min] for k, l in model.history.history.items()}
)
history_df.to_csv(f"{model_file}/history.tsv", sep="\t")

# Reload best model
model = tf.keras.models.load_model(model_file, compile=False)
model.compile(optimizer=opt, loss=loss, metrics=METRICS)
metrics_dict = {m.name: m for m in model.metrics}
logger = callbacks.BatchMetricsLogger(metrics=metrics_dict)

# Test data generator
test_body = (
    tf.data.Dataset.from_tensor_slices(test_data.index.values)
    .batch(1)
)
test_dx = test_body.interleave(
    lambda filename: tf.data.Dataset.from_generator(
        gen,  # Generator class
        ({"input_" + k.name + "_continuous": tf.float32 for k in [actual_test_tm]}),
        (
            {
                "input_"
                + k.name
                + "_continuous": tf.TensorShape(tuple([None] + list(k.shape)))
                for k in [actual_test_tm]
            }
        ),  # Output shape
        args=(
            filename,
            [1],
        ),  # Pass tf.tensor to execute as string with eager execution
    ),
)
test_dy = test_body.map(
    pandas_load_wrapper_dict(test_data, ['vat_ensemble','asat_ensemble','gfat_ensemble','vat_asat_ensemble']),
)
test_ds = tf.data.Dataset.zip((test_dx, test_dy)).prefetch(512)


eval = model.evaluate(test_ds, callbacks=[logger], verbose=1)
eval = {out: eval[i] for i, out in enumerate(model.metrics_names)}
eval = pd.DataFrame(eval, index=[model_file])
eval.to_csv(f"{model_file}/evaluate.tsv", sep="\t")
eval_batch = pd.DataFrame(logger.storage, index=test_data.index)
eval_batch.to_csv(f"{model_file}/evaluate_batch.tsv", sep="\t")

# Compute predictions
pred = model.predict(test_ds, verbose=1)
pred = pd.concat([test_data[['vat_ensemble','asat_ensemble','gfat_ensemble','vat_asat_ensemble']], 
    pd.DataFrame(np.hstack(pred),index=test_data.index,
        columns=[a+'_pred' for a in ['vat_ensemble','asat_ensemble','gfat_ensemble','vat_asat_ensemble']])], axis=1)


pred.to_csv(f"{model_file}/predictions.tsv", sep="\t")

# Join batch-wise metric evaluations with y, and y_hat
joint = pd.concat([eval_batch, pred], axis=1)
joint.to_csv(f"{model_file}/joint.tsv", sep="\t")

model2 = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer('latent_space').output)
pred_latent = model2.predict(test_ds, verbose=1)
pred_latent = pd.DataFrame(pred_latent,index=test_data.index)
pred_latent.columns = ['latent_' + str(x) for x in np.arange(pred_latent.shape[1])]
pred_latent.to_parquet(f"{model_file}/predictions_latent_test.pq")

model2 = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer('latent_space').output)
pred_latent = model2.predict(train_ds, verbose=1)
pred_latent = pd.DataFrame(pred_latent,index=train_data.index)
pred_latent.columns = ['latent_' + str(x) for x in np.arange(pred_latent.shape[1])]
pred_latent.to_parquet(f"{model_file}/predictions_latent_train.pq")

# compute pearson and spearman
from scipy.stats import spearmanr

def get_correlations(a,b,field):
    return pd.DataFrame({
        "pearson_r": np.corrcoef(a, b)[0][1],
        "pearson_r2": np.corrcoef(a,b)[0][1]**2,
        "pearson_spearman": spearmanr(a,b)[0],
    },index=[field])

def get_correlation_df(source):
    return pd.concat([
        get_correlations(source.vat_ensemble,source.vat_ensemble_pred,'vat'),
        get_correlations(source.asat_ensemble,source.asat_ensemble_pred,'asat'),
        get_correlations(source.gfat_ensemble,source.gfat_ensemble_pred,'gfat'),
        get_correlations(source.vat_asat_ensemble,source.vat_asat_ensemble_pred,'vat_asat'),
        get_correlations(source.vat_ensemble/source.asat_ensemble,source.vat_ensemble_pred/source.asat_ensemble_pred,'vat_asat_post'),
    ])

corr = get_correlation_df(pred)
corr.to_csv(f"{model_file}/correlations.tsv", sep="\t")
corr_female = get_correlation_df(pred.loc[test_data.sex==0])
corr_female.to_csv(f"{model_file}/correlations_female.tsv", sep="\t")
corr_male = get_correlation_df(pred.loc[test_data.sex==1])
corr_male.to_csv(f"{model_file}/correlations_male.tsv", sep="\t")

# Holdout data generator
test_body = (
    tf.data.Dataset.from_tensor_slices(holdout_data.index.values)
    .batch(1)
)
test_dx = test_body.interleave(
    lambda filename: tf.data.Dataset.from_generator(
        gen,  # Generator class
        ({"input_" + k.name + "_continuous": tf.float32 for k in [actual_test_tm]}),
        (
            {
                "input_"
                + k.name
                + "_continuous": tf.TensorShape(tuple([None] + list(k.shape)))
                for k in [actual_test_tm]
            }
        ),  # Output shape
        args=(
            filename,
            [1],
        ),  # Pass tf.tensor to execute as string with eager execution
    ),
)
test_dy = test_body.map(
    pandas_load_wrapper_dict(holdout_data, ['vat_ensemble','asat_ensemble','gfat_ensemble','vat_asat_ensemble']),
)
test_ds = tf.data.Dataset.zip((test_dx, test_dy)).prefetch(512)



metrics_dict = {m.name: m for m in model.metrics}
logger = callbacks.BatchMetricsLogger(metrics=metrics_dict)

eval = model.evaluate(test_ds, callbacks=[logger], verbose=1)
eval = {out: eval[i] for i, out in enumerate(model.metrics_names)}
eval = pd.DataFrame(eval, index=[model_file])
eval.to_csv(f"{model_file}/evaluate_holdout.tsv", sep="\t")
eval_batch = pd.DataFrame(logger.storage, index=holdout_data.index)
eval_batch.to_csv(f"{model_file}/evaluate_batch_holdout.tsv", sep="\t")

# Compute predictions
pred = model.predict(test_ds, verbose=1)
pred = pd.concat([holdout_data[['vat_ensemble','asat_ensemble','gfat_ensemble','vat_asat_ensemble']], 
    pd.DataFrame(np.hstack(pred),index=holdout_data.index,
        columns=[a+'_pred' for a in ['vat_ensemble','asat_ensemble','gfat_ensemble','vat_asat_ensemble']])], axis=1)

pred.to_csv(f"{model_file}/predictions_holdout.tsv", sep="\t")

# Join batch-wise metric evaluations with y, and y_hat
joint = pd.concat([eval_batch, pred], axis=1)
joint.to_csv(f"{model_file}/joint_holdout.tsv", sep="\t")

# compute pearson and spearman
corr = get_correlation_df(pred)
corr.to_csv(f"{model_file}/correlations_holdout.tsv", sep="\t")
corr_female = get_correlation_df(pred.loc[holdout_data.sex==0])
corr_female.to_csv(f"{model_file}/correlations_holdout_female.tsv", sep="\t")
corr_male = get_correlation_df(pred.loc[holdout_data.sex==1])
corr_male.to_csv(f"{model_file}/correlations_holdout_male.tsv", sep="\t")

#
model2 = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer('latent_space').output)
pred_latent = model2.predict(test_ds, verbose=1)
pred_latent = pd.DataFrame(pred_latent,index=holdout_data.index)
pred_latent.columns = ['latent_' + str(x) for x in np.arange(pred_latent.shape[1])]
pred_latent.to_parquet(f"{model_file}/predictions_latent_holdout.pq")

y_names = ['vat_ensemble','asat_ensemble','gfat_ensemble','vat_asat_ensemble']
yhat_names = ['vat_ensemble_pred','asat_ensemble_pred','gfat_ensemble_pred','vat_asat_ensemble_pred']

parts = []
for a,b in zip(y_names, yhat_names):
    ll = [[
        np.corrcoef(pred.loc[holdout_data.index][a], pred.loc[holdout_data.index][b])[0][1],
        np.corrcoef(pred[holdout_data.age<50][a], pred[holdout_data.age<50][b])[0][1],
        np.corrcoef(pred[(holdout_data.age>50)&(holdout_data.age<60)][a], pred[(holdout_data.age>50)&(holdout_data.age<60)][b])[0][1],
        np.corrcoef(pred[(holdout_data.age>60)&(holdout_data.age<70)][a], pred[(holdout_data.age>60)&(holdout_data.age<70)][b])[0][1],
        np.corrcoef(pred[holdout_data.age>70][a], pred[holdout_data.age>70][b])[0][1],
    ]]
    for g in [0,1]:
        ll.append([
            np.corrcoef(pred[(holdout_data.sex==g)][a], pred[(holdout_data.sex==g)][b])[0][1],
            np.corrcoef(pred[(holdout_data.age<50)&(holdout_data.sex==g)][a], pred[(holdout_data.age<50)&(holdout_data.sex==g)][b])[0][1],
            np.corrcoef(pred[(holdout_data.age>50)&(holdout_data.age<60)&(holdout_data.sex==g)][a], pred[(holdout_data.age>50)&(holdout_data.age<60)&(holdout_data.sex==g)][b])[0][1],
            np.corrcoef(pred[(holdout_data.age>60)&(holdout_data.age<70)&(holdout_data.sex==g)][a], pred[(holdout_data.age>60)&(holdout_data.age<70)&(holdout_data.sex==g)][b])[0][1],
            np.corrcoef(pred[(holdout_data.age>70)&(holdout_data.sex==g)][a], pred[(holdout_data.age>70)&(holdout_data.sex==g)][b])[0][1],
        ])
    parts.append(pd.DataFrame(ll,columns=["any_age","under_50","50-60","60-70","over_70"],index=[a + '_' + b for b in ['both','female','male']]).T)


corr_sub = pd.concat(parts,axis=1)
corr_sub.to_csv(f"{model_file}/correlations_sub_holdout.tsv", sep="\t")

###
nparts = []
for a,b in zip(y_names, yhat_names):
    ll = [[
        pred.loc[holdout_data.index][a].shape[0],
        pred[holdout_data.age<50][a].shape[0],
        pred[(holdout_data.age>50)&(holdout_data.age<60)][a].shape[0],
        pred[(holdout_data.age>60)&(holdout_data.age<70)][a].shape[0],
        pred[holdout_data.age>70][a].shape[0],
    ]]
    for g in [0,1]:
        ll.append([
            pred[(holdout_data.sex==g)][a].shape[0],
            pred[(holdout_data.age<50)&(holdout_data.sex==g)][a].shape[0],
            pred[(holdout_data.age>50)&(holdout_data.age<60)&(holdout_data.sex==g)][a].shape[0],
            pred[(holdout_data.age>60)&(holdout_data.age<70)&(holdout_data.sex==g)][a].shape[0],
            pred[(holdout_data.age>70)&(holdout_data.sex==g)][a].shape[0],
        ])
    nparts.append(pd.DataFrame(ll,columns=["any_age","under_50","50-60","60-70","over_70"],index=[a + '_' + b for b in ['both','female','male']]).T)


nparts_sub = pd.concat(nparts,axis=1)
nparts_sub.to_csv(f"{model_file}/correlations_sub_counts_holdout.tsv", sep="\t")


pretty = dict()
pretty_square = dict()
for a,b in zip(corr_sub,nparts_sub):
    internal = []
    internal_square = []
    for c,d in zip(corr_sub[a].values,nparts_sub[b].values):
        internal.append(f"{np.round(c,3)} ({d})")
        internal_square.append(f"{np.round(c**2,3)} ({d})")
    pretty[a] = internal
    pretty_square[a] = internal_square


pretty = pd.DataFrame(pretty,index=nparts_sub.index)
pretty_square = pd.DataFrame(pretty_square,index=nparts_sub.index)
pretty.to_csv(f"{model_file}/correlations_sub_pretty_holdout.tsv", sep="\t")
pretty_square.to_csv(f"{model_file}/correlations_sub_pretty_square_holdout.tsv", sep="\t")
