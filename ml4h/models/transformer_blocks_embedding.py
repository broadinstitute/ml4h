from typing import Dict, List, Tuple

import tensorflow as tf

from ml4h.models.Block import Block
from ml4h.TensorMap import TensorMap

tf.random.set_seed(1234)
Tensor = tf.Tensor


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

        y=tf.keras.backend.mean(y,1)#batch size x output size
        return y


def scaled_dot_product_attention(query, key, value, mask):
    """Calculate the attention weights. """
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask to zero out padding tokens
    mask =None
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
    inputs = tf.keras.Input(shape=(None,3), name=input_name)
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
            name="encoder_layer_{}".format(i),
            input_name=input_name,
        )([outputs, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name,
    )


