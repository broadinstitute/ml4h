"""Parallel, finite DROID input datasets with bounded video buffers."""

import numpy as np
import tensorflow as tf


def make_dataset(input_dd, output_dd, sample_ids, batch_size, output_signature,
                 shuffle, workers=4, seed=None):
    """Load clips in parallel; prepare immutable label tensors once.

    Shuffle IDs/labels before decoding. Never cache decoded clips, open LMDB
    environments or randomized augmentations. Validation stays finite; callers
    may repeat the training dataset and prefetch an explicit number of batches.
    """
    if workers < 1 or batch_size < 1:
        raise ValueError('workers and batch_size must be positive')
    if getattr(output_dd, 'transforms', None):
        raise ValueError('Precomputed labels require an output description without transforms')
    sample_ids = list(sample_ids)
    if len(sample_ids) < batch_size:
        raise ValueError('Dataset needs at least one full batch')
    image_spec, label_spec = output_signature
    multiple_outputs = isinstance(label_spec, (list, tuple))
    specs = list(label_spec) if multiple_outputs else [label_spec]
    labels = [np.empty((len(sample_ids), *spec.shape[1:]), np.float32) for spec in specs]
    for row, sample_id in enumerate(sample_ids):
        values = output_dd.get_raw_data(sample_id)
        values = list(values) if isinstance(values, (tuple, list)) else [values]
        if len(values) != len(labels):
            raise ValueError(f'Expected {len(labels)} label heads, got {len(values)}')
        for target, value in zip(labels, values):
            target[row] = np.asarray(value, dtype=np.float32).reshape(target.shape[1:])

    @tf.autograph.experimental.do_not_convert
    def load_clip(sample_id):
        # Existing augmentation classes are eager and share parameters across
        # the frames of each clip. Keep their work off the training GPU.
        with tf.device('/CPU:0'):
            return input_dd.get_raw_data(sample_id.numpy())

    def load(sample_id, targets):
        video = tf.py_function(load_clip, [sample_id], Tout=tf.float32)
        video.set_shape(image_spec.shape[1:])
        return video, targets

    with tf.device('/CPU:0'):
        targets = tuple(labels) if multiple_outputs else labels[0]
        dataset = tf.data.Dataset.from_tensor_slices((sample_ids, targets))
        if shuffle:
            dataset = dataset.shuffle(len(sample_ids), seed=seed, reshuffle_each_iteration=True)
        dataset = dataset.map(load, num_parallel_calls=workers, deterministic=True)
        dataset = dataset.batch(batch_size, drop_remainder=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = workers
    options.threading.max_intra_op_parallelism = 1
    options.autotune.enabled = False
    options.experimental_optimization.inject_prefetch = False
    return dataset.with_options(options)
