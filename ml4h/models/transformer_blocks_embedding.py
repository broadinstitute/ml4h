import logging
from typing import Dict, List
import numpy as np

import keras
from keras import layers
import tensorflow as tf

from ml4h.models.Block import Block
from ml4h.TensorMap import TensorMap

tf.random.set_seed(1234)
Tensor = tf.Tensor


# Register serializable functions at module level
@keras.saving.register_keras_serializable(package='ml4h')
class PositionIndexLayer(keras.layers.Layer):
    """Custom layer to generate position indices for positional encoding."""

    def __init__(self, max_len, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len

    def call(self, inputs):
        """Generate position indices."""
        b = keras.ops.shape(inputs)[0]
        pos = keras.ops.arange(0, self.max_len)
        pos = keras.ops.tile(keras.ops.expand_dims(pos, 0), (b, 1))
        return pos

    def get_config(self):
        config = super().get_config()
        config.update({'max_len': self.max_len})
        return config


@keras.saving.register_keras_serializable(package='ml4h')
class ExpandDimsLayer(keras.layers.Layer):
    """Custom layer to expand dimensions with specified axis."""

    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        """Expand dimensions."""
        return keras.ops.expand_dims(inputs, self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({'axis': self.axis})
        return config


@keras.saving.register_keras_serializable(package='ml4h')
class LogicalAndLayer(keras.layers.Layer):
    """Custom layer to apply logical AND to two inputs."""

    def call(self, inputs):
        """Apply logical AND."""
        return keras.ops.logical_and(inputs[0], inputs[1])


@keras.saving.register_keras_serializable(package='ml4h')
class CastToFloatLayer(keras.layers.Layer):
    """Custom layer to cast boolean to float32."""

    def call(self, inputs):
        """Cast to float32."""
        return keras.ops.cast(inputs, 'float32')


@keras.saving.register_keras_serializable(package='ml4h')
class ApplyVeryNegativeLayer(keras.layers.Layer):
    """Custom layer to apply very negative values to masked positions."""

    def call(self, inputs):
        """Apply very negative values."""
        return (1.0 - inputs) * (-1e9)


@keras.saving.register_keras_serializable(package='ml4h')
class SumOverTimeLayer(keras.layers.Layer):
    """Custom layer to sum over time dimension."""

    def call(self, inputs):
        """Sum over axis 1."""
        return keras.ops.sum(inputs, axis=1)





class TransformerEncoderEmbedding(Block):
    #this version directly intakes in tokens in the shape of (batch_size, N_tokens,embedim)
    def __init__(
        self, *, tensor_map: TensorMap, dense_layers: List[int],
        dense_regularize_rate: float, attention_heads: int, transformer_size: int, **kwargs
    ):
        self.tensor_map = tensor_map
        if not self.can_apply():
            return

        self.dropout = tf.keras.layers.Dropout(rate=dense_regularize_rate)
        self.padding_mask_layer = tf.keras.layers.Lambda(
            create_padding_mask, output_shape=(256, None),
            name='encoder_padding_mask',
        )

        self.encoder_layers = encoder(
            vocab_size=len(tensor_map.channel_map),
            window_size=tensor_map.shape[0],
            num_layers=len(dense_layers),
            units=dense_layers[0],
            d_model=transformer_size,
            num_heads=attention_heads,
            dropout=dense_regularize_rate,
            input_name=tensor_map.input_name(),
        )

    def can_apply(self):
        return True

    def __call__(self, x: Tensor, intermediates: Dict[TensorMap, List[Tensor]] = None) -> Tensor:
        if not self.can_apply():
            return x

        padded = self.padding_mask_layer(x)
        y = self.encoder_layers(inputs=[x, padded])
        intermediates[self.tensor_map.dependent_map].extend([x, y])
        y = tf.keras.backend.mean(y, 1)  # batch size x output size
        return y


def scaled_dot_product_attention(query, key, value, mask):
    """Calculate the attention weights. """
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask to zero out padding tokens
    mask = None
    if mask is not None:
        logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_weights, value)
    return output


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention", **kwargs):
        super(MultiHeadAttention, self).__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'d_model': self.d_model, 'num_heads': self.num_heads})
        return config

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth),
        )
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(
            scaled_attention,
            (batch_size, -1, self.d_model),
        )

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs


def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - \
        tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)


class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model
        self.position = position
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'d_model': self.d_model, 'position': self.position})
        return config

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(
            10000, (2 * (i // 2)) /
            tf.cast(d_model, tf.float32),
        )
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model,
        )
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer", input_name="inputs"):
    inputs = tf.keras.Input(shape=(None, d_model), name=input_name)
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttention(
        d_model, num_heads, name="attention",
    )({
            'query': inputs,
            'key': inputs,
            'value': inputs,
            'mask': padding_mask,
    })

    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(
        epsilon=1e-6,
    )(inputs + attention)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6,
    )(attention + outputs)

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name,
    )


def encoder(
    vocab_size,
    window_size,
    num_layers,
    units,
    d_model,
    num_heads,
    dropout,
    name="encoder",
    input_name="inputs",
):

    inputs = tf.keras.Input(shape=(None, 3), name=input_name)
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Dense(units=d_model, activation='relu')(inputs)
    embeddings = PositionalEncoding(window_size, d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)
    for i in range(num_layers):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name=f"encoder_layer_{i}",
            input_name=input_name,
        )([outputs, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name,
    )


def build_embedding_transformer(
        INPUT_NUMERIC_COLS,
        REGRESSION_TARGETS,
        BINARY_TARGETS,
        MAX_LEN,
        EMB_DIM,
        TOKEN_HIDDEN,
        TRANSFORMER_DIM,
        NUM_HEADS,
        NUM_LAYERS,
        DROPOUT,
        view2id,
):
    Feat = len(INPUT_NUMERIC_COLS)

    inp_num = keras.Input(shape=(MAX_LEN, Feat), dtype='float32', name='num')
    inp_mask = keras.Input(shape=(MAX_LEN,), dtype='bool', name='mask')  # True = valid

    if view2id is not None:
        inp_view = keras.Input(shape=(MAX_LEN,), dtype='int32', name='view')
        view_emb = layers.Embedding(
            input_dim=int(max(view2id.values())) + 1,  # include PAD
            output_dim=EMB_DIM,
            mask_zero=True,
            name='view_embedding'
        )(inp_view)  # (B,T,EMB_DIM)
        # Token features: [embed(view) || numeric features]
        x = layers.Concatenate(name='token_concat')([view_emb, inp_num])  # (B,T,EMB_DIM+F)
        x = layers.Dense(TOKEN_HIDDEN, activation='relu', name='token_proj')(x)
    else:
        x = layers.Dense(TOKEN_HIDDEN, activation='relu', name='token_proj')(inp_num)
    x = layers.Dropout(DROPOUT)(x)

    # Positional embedding (learnable)
    if view2id is not None:
        pos_idx = PositionIndexLayer(max_len=MAX_LEN, name='pos_idx')(inp_view)
        pos_emb = layers.Embedding(input_dim=MAX_LEN, output_dim=TOKEN_HIDDEN, name='pos_embedding')(pos_idx)
        x = layers.Add(name='add_pos')([x, pos_emb])

    # Build (B,T,T) attention mask from (B,T)
    m_q = ExpandDimsLayer(axis=2, name='mask_q')(inp_mask)
    m_k = ExpandDimsLayer(axis=1, name='mask_k')(inp_mask)
    mask_2d = LogicalAndLayer(name='mask_qk')([m_q, m_k])

    # Transformer blocks
    for i in range(NUM_LAYERS):
        attn = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=TRANSFORMER_DIM // NUM_HEADS,
                                         dropout=DROPOUT, name=f'mha_{i}')(x, x, attention_mask=mask_2d)
        attn = layers.Dropout(DROPOUT)(attn)
        x = layers.LayerNormalization(epsilon=1e-6, name=f'ln1_{i}')(layers.Add()([x, attn]))

        ff = layers.Dense(TRANSFORMER_DIM, activation='relu', name=f'ff1_{i}')(x)
        ff = layers.Dropout(DROPOUT)(ff)
        ff = layers.Dense(TOKEN_HIDDEN, name=f'ff2_{i}')(ff)
        x = layers.LayerNormalization(epsilon=1e-6, name=f'ln2_{i}')(layers.Add()([x, ff]))

    # Attention pooling over time (mask-aware via very negative)
    score_h = layers.Dense(TOKEN_HIDDEN, activation='tanh', name='attn_h')(x)  # (B,T,D)
    score = layers.Dense(1, name='attn_score')(score_h)  # (B,T,1)
    score = layers.Reshape((MAX_LEN,), name='attn_score_squeeze')(score)  # (B,T)

    mask_f = CastToFloatLayer(name='mask_cast')(inp_mask)

    very_neg = ApplyVeryNegativeLayer(name='veryneg')(mask_f)
    score_m = layers.Add(name='score_masked')([score, very_neg])
    wts = layers.Softmax(axis=-1, name='attn_wts')(score_m)  # (B,T)
    wts_e = layers.Reshape((MAX_LEN, 1), name='wts_e')(wts)
    ctx = layers.Multiply(name='apply_wts')([x, wts_e])  # (B,T,D)
    ctx = SumOverTimeLayer(name='pool')(ctx)  # (B,D)

    # Shared tower
    h = layers.Dense(128, activation='relu')(ctx)
    h = layers.Dropout(DROPOUT)(h)

    # Task heads (names must match keys used in y/sample_weight dicts)
    outputs = {}
    for t in REGRESSION_TARGETS:
        outputs[t] = layers.Dense(1, name=t)(h)  # linear
    for t in BINARY_TARGETS:
        outputs[t] = layers.Dense(1, activation='sigmoid', name=t)(h)

    if view2id is not None:
        model = keras.Model(inputs={'view': inp_view, 'num': inp_num, 'mask': inp_mask}, outputs=outputs)
    else:
        model = keras.Model(inputs={'num': inp_num, 'mask': inp_mask}, outputs=outputs)
    # Losses / metrics
    losses = {t: 'mse' for t in REGRESSION_TARGETS}
    losses.update({t: 'binary_crossentropy' for t in BINARY_TARGETS})

    metrics = {t: [keras.metrics.MeanAbsoluteError(name='mae'),
                   keras.metrics.MeanSquaredError(name='mse')] for t in REGRESSION_TARGETS}
    metrics.update({t: [keras.metrics.AUC(name='auroc', curve='ROC'),
                        keras.metrics.AUC(name='auprc', curve='PR'),
                        keras.metrics.BinaryAccuracy(name='acc')] for t in BINARY_TARGETS})

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=losses,
        metrics=metrics
    )

    return model

from sklearn.metrics import r2_score, roc_auc_score, average_precision_score, accuracy_score

def evaluate_multitask_on_dataset(
    name,
    model,
    dataset,
    REGRESSION_TARGETS,
    BINARY_TARGETS,
    steps=None,   # if dataset is repeated(), pass the number of batches to consume
    verbose=True,
):
    # Accumulators
    y_true = {t: [] for t in REGRESSION_TARGETS + BINARY_TARGETS}
    y_pred = {t: [] for t in REGRESSION_TARGETS + BINARY_TARGETS}
    w      = {t: [] for t in REGRESSION_TARGETS + BINARY_TARGETS}
    performance_data = []

    def _consume():
        if steps is None:
            for x, y, sw in dataset:
                outs = model(x, training=False)
                for t in y_true.keys():
                    y_true[t].append(tf.convert_to_tensor(y[t]).numpy())
                    w[t].append(tf.convert_to_tensor(sw[t]).numpy())
                    # model outputs dict of tensors; ensure 1D
                    yp = tf.convert_to_tensor(outs[t]).numpy().reshape(-1)
                    y_pred[t].append(yp)
        else:
            it = iter(dataset)
            for _ in range(int(steps)):
                try:
                    x, y, sw = next(it)
                except StopIteration:
                    break
                outs = model(x, training=False)
                for t in y_true.keys():
                    y_true[t].append(tf.convert_to_tensor(y[t]).numpy())
                    w[t].append(tf.convert_to_tensor(sw[t]).numpy())
                    yp = tf.convert_to_tensor(outs[t]).numpy().reshape(-1)
                    y_pred[t].append(yp)

    _consume()

    # Concatenate
    for t in y_true.keys():
        y_true[t] = np.concatenate(y_true[t], axis=0) if len(y_true[t]) else np.array([])
        y_pred[t] = np.concatenate(y_pred[t], axis=0) if len(y_pred[t]) else np.array([])
        w[t]      = np.concatenate(w[t],      axis=0) if len(w[t])      else np.array([])

    # Metrics
    results = {}

    # Regression tasks
    for t in REGRESSION_TARGETS:
        if y_true[t].size == 0:
            #results[t] = {"MAE": np.nan, "MSE": np.nan, "R2": np.nan}
            continue
        msk = w[t] > 0
        if msk.sum() == 0:
            #results[t] = {"MAE": np.nan, "MSE": np.nan, "R2": np.nan}
            continue
        yt = y_true[t][msk].astype("float32")
        yp = y_pred[t][msk].astype("float32")
        mae = float(np.mean(np.abs(yp - yt)))
        mse = float(np.mean((yp - yt) ** 2))
        try:
            r2  = float(r2_score(yt, yp))
        except ValueError:
            r2 = float("nan")
        results[t] = {"MAE": mae, "MSE": mse, "R2": r2}
        performance_data.append({"Model": name, "Task": t, "Metric": 'R^2', "Score": r2})
    # Binary tasks
    for t in BINARY_TARGETS:
        if y_true[t].size == 0:
            results[t] = {"AUROC": np.nan, "AUPRC": np.nan, "ACC": np.nan}
            continue
        msk = w[t] > 0
        if msk.sum() == 0:
            results[t] = {"AUROC": np.nan, "AUPRC": np.nan, "ACC": np.nan}
            continue
        yt = (y_true[t][msk] > 0.5).astype("int32")
        prob = y_pred[t][msk].astype("float32")
        try:
            auroc = float(roc_auc_score(yt, prob))
        except ValueError:
            auroc = float("nan")
        try:
            auprc = float(average_precision_score(yt, prob))
        except ValueError:
            auprc = float("nan")
        acc = float(accuracy_score(yt, (prob >= 0.5).astype("int32")))
        results[t] = {"AUROC": auroc, "AUPRC": auprc, "ACC": acc}
        performance_data.append({"Model": name, "Task": t, "Metric": 'auROC', "Score": auroc})
        performance_data.append({"Model": name, "Task": t, "Metric": 'auPRC', "Score": auprc})
    if verbose:
        logging.info("\n=== Evaluation on dataset ===")
        for t in REGRESSION_TARGETS:
            r = results[t]
            logging.info(f"{t:30s}  MAE: {r['MAE']:.4f}  MSE: {r['MSE']:.4f}  R^2: {r['R2']:.4f}")
        for t in BINARY_TARGETS:
            r = results[t]
            logging.info(f"{t:30s}  AUROC: {r['AUROC']:.4f}  AUPRC: {r['AUPRC']:.4f}  ACC: {r['ACC']:.4f}")

    return performance_data
