"""One self-attention block over the clip embeddings in an echo study."""

import tensorflow as tf


def build_study_attention_model(embedding_dim, spec, num_heads=4, key_dim=64,
                                dropout=0.1, pooling='attention'):
    """Predict the same DROID heads once per study from a variable-length clip set.

    The mask is explicit so padded clips cannot act as keys or enter the pooled
    representation. No position encoding is used: clip order has no clinical meaning.
    """
    embeddings = tf.keras.Input(shape=(None, embedding_dim), name='embeddings')
    mask = tf.keras.Input(shape=(None,), dtype=tf.bool, name='mask')
    attended = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, dropout=dropout,
        name='study_self_attention',
    )(embeddings, embeddings, attention_mask=tf.expand_dims(mask, axis=1))
    features = tf.keras.layers.LayerNormalization(name='attention_norm')(embeddings + attended)
    valid = tf.cast(tf.expand_dims(mask, axis=-1), features.dtype)
    if pooling == 'attention':
        scores = tf.keras.layers.Dense(1, name='attention_pool_score')(features)
        scores = tf.where(tf.expand_dims(mask, axis=-1), scores,
                          tf.fill(tf.shape(scores), tf.cast(-1e9, scores.dtype)))
        weights = tf.nn.softmax(scores, axis=1)
        pooled = tf.reduce_sum(features * weights, axis=1)
    elif pooling == 'mean':
        pooled = tf.reduce_sum(features * valid, axis=1) / tf.reduce_sum(valid, axis=1)
    else:
        raise ValueError("pooling must be 'attention' or 'mean'.")
    pooled = tf.keras.layers.Dropout(dropout, name='study_dropout')(pooled)

    outputs = []
    if spec['n_output_features']:
        outputs.append(tf.keras.layers.Dense(
            spec['n_output_features'], name='echolab',
        )(pooled))
    for name in spec['category_order'] or []:
        outputs.append(tf.keras.layers.Dense(
            spec['categories'][name], activation='softmax', name=f'cls_{name}',
        )(pooled))
    for name, intervals in spec['survival_heads'].items():
        outputs.append(tf.keras.layers.Dense(
            intervals, activation='sigmoid', name=f'survival_{name}',
        )(pooled))
    if not outputs:
        raise ValueError('The source run has no prediction heads.')
    return tf.keras.Model(
        inputs={'embeddings': embeddings, 'mask': mask}, outputs=outputs,
        name='droid_study_attention',
    )
