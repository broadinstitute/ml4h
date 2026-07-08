
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Dropout,
    BatchNormalization,
    Activation,
    Add,
    Dense,
    GlobalAveragePooling1D,
)
import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np


def PCLR_model() -> Model:
    """
    Builds and compiles the PCLR model used in the pre-print.
    """
    model = get_model(
        projection_layer_shapes=[320, 320],
        ecg_len=4096,
        dropout=0.,
        activation="relu",
        num_channels=12,
    )
    optimizer = tf.keras.optimizers.Adam(1e-1)
    model.compile(
        loss=[simclr_loss],
        metrics=[simclr_accuracy],
        optimizer=optimizer,
    )
    return model


def CLOCS_model() -> Model:
    """
    Builds and compiles the CLOCS model used for comparison.
    """
    model = get_model(
        projection_layer_shapes=[320, 320],
        ecg_len=4096,
        dropout=0.,
        activation="relu",
        num_channels=12,
    )
    optimizer = tf.keras.optimizers.Adam(1e-2)
    model.compile(
        loss=[simclr_loss],
        metrics=[simclr_accuracy],
        optimizer=optimizer,
    )
    return model


def CAE_model() -> Model:
    """
    Builds and compiles the convolutional autoencoder used for comparison against PCLR
    """
    model = get_model(
        projection_layer_shapes=[1], dropout=0., ecg_len=4096, activation="swish",
    )
    # upsampling for reconstruction
    blocks = [
        [
            tf.keras.layers.Conv1DTranspose(filters=filters, kernel_size=16, padding="same", strides=4),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("swish"),
        ]
        for filters in [256, 196, 128, 64]
    ]
    blocks = sum(blocks, [])

    # make bottleneck the correct shape for upsampling
    new_layers = [
        tf.keras.layers.Dense(16 * 320),
        tf.keras.layers.Reshape((16, 320)),
    ] + blocks + [
        tf.keras.layers.Conv1D(filters=12, padding="same", kernel_size=1, name="ecg_out"),
    ]
    # the embedding is the representation layer of the PCLR architecture
    embed = model.layers[-2].output
    x = embed
    for new_layer in new_layers:
        x = new_layer(x)
    model = tf.keras.models.Model(
        model.input, x,
    )
    optimizer = tf.keras.optimizers.Adam(1e-4)
    model.compile(
        loss=[tf.keras.losses.log_cosh],
        metrics=[tf.keras.losses.MSE],
        optimizer=optimizer,
    )
    return model


def ribeiro_r_model() -> Model:
    """
    Builds and compiles the classification baseline used for comparison against PCLR
    """
    model = get_model(
        projection_layer_shapes=[1], ecg_len=4096, activation="relu",
    )
    # the embedding is the representation layer of the PCLR architecture
    old_out = model.layers[-2].output
    classification_targets = [
        "rbbb",
        "lbbb",
        "avb",
        "af",
        'sb',
        'st',
    ]
    model = tf.keras.models.Model(
        model.input,
        [
            tf.keras.layers.Dense(2, name=target, activation="sigmoid")(old_out)
            for target in classification_targets
        ],
    )
    optimizer = tf.keras.optimizers.Adam(1e-2)
    model.compile(
        loss=[tf.keras.metrics.categorical_crossentropy for _ in classification_targets],
        metrics=tf.keras.metrics.categorical_accuracy,
        optimizer=optimizer,
    )
    return model


class ResidualUnit:
    """
    Residual unit block (unidimensional).
    Parameters
    ----------
    n_samples_out: int
        Number of output samples.
    n_filters_out: int
        Number of output filters.
    kernel_initializer: str, optional
        Initializer for the weights matrices. See Keras initializers. By default it uses
        'he_normal'.
    dropout_keep_prob: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. Default is 17.
    preactivation: bool, optional
        When preactivation is true use full preactivation architecture proposed
        in [1]. Otherwise, use architecture proposed in the original ResNet
        paper [2]. By default it is true.
    postactivation_bn: bool, optional
        Defines if you use batch normalization before or after the activation layer (there
        seems to be some advantages in some cases:
        https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md).
        If true, the batch normalization is used before the activation
        function, otherwise the activation comes first, as it is usually done.
        By default it is false.
    activation_function: string, optional
        Keras activation function to be used. By default 'relu'.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027 [cs], Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(
        self,
        n_samples_out,
        n_filters_out,
        kernel_initializer="he_normal",
        dropout=0.2,
        kernel_size=17,
        preactivation=True,
        postactivation_bn=False,
        activation="relu",
    ):
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = dropout
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation

    def _skip_connection(self, y, downsample, n_filters_in):
        """Implement skip connection."""
        # Deal with downsampling
        if downsample > 1:
            y = MaxPooling1D(downsample, strides=downsample, padding="same")(y)
        elif downsample == 1:
            y = y
        else:
            raise ValueError("Number of samples should always decrease.")
        # Deal with n_filters dimension increase
        if n_filters_in != self.n_filters_out:
            # This is one of the two alternatives presented in ResNet paper
            # Other option is to just fill the matrix with zeros.
            y = Conv1D(
                self.n_filters_out,
                1,
                padding="same",
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
            )(y)
        return y

    def _batch_norm_plus_activation(self, x):
        if self.postactivation_bn:
            x = Activation(self.activation_function)(x)
            x = BatchNormalization(center=False, scale=False)(x)
        else:
            x = BatchNormalization()(x)
            x = Activation(self.activation_function)(x)
        return x

    def __call__(self, inputs):
        """Residual unit."""
        x, y = inputs
        n_samples_in = y.shape[1]
        downsample = n_samples_in // self.n_samples_out
        n_filters_in = y.shape[2]
        y = self._skip_connection(y, downsample, n_filters_in)
        # 1st layer
        x = Conv1D(
            self.n_filters_out,
            self.kernel_size,
            padding="same",
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
        )(x)
        x = self._batch_norm_plus_activation(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)

        # 2nd layer
        x = Conv1D(
            self.n_filters_out,
            self.kernel_size,
            strides=downsample,
            padding="same",
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
        )(x)
        if self.preactivation:
            x = Add()([x, y])  # Sum skip connection and main connection
            y = x
            x = self._batch_norm_plus_activation(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
        else:
            x = BatchNormalization()(x)
            x = Add()([x, y])  # Sum skip connection and main connection
            x = Activation(self.activation_function)(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
            y = x
        return [x, y]


def get_model(
    projection_layer_shapes,
    dropout=0.2,
    ecg_len=4096,
    activation="relu",
    num_channels: int = 12,
):
    """
    modified code from https://github.com/antonior92/automatic-ecg-diagnosis
    """
    kernel_size = 16
    kernel_initializer = "he_normal"
    signal = Input(shape=(ecg_len, num_channels), dtype=np.float32, name="ecg")
    x = signal
    x = Conv1D(
        64,
        kernel_size,
        padding="same",
        use_bias=False,
        kernel_initializer=kernel_initializer,
    )(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x, y = ResidualUnit(
        1024,
        128,
        kernel_size=kernel_size,
        kernel_initializer=kernel_initializer,
        dropout=dropout,
        activation=activation,
    )([x, x])
    x, y = ResidualUnit(
        256,
        196,
        kernel_size=kernel_size,
        kernel_initializer=kernel_initializer,
        dropout=dropout,
        activation=activation,
    )([x, y])
    x, y = ResidualUnit(
        64,
        256,
        kernel_size=kernel_size,
        kernel_initializer=kernel_initializer,
        dropout=dropout,
        activation=activation,
    )([x, y])
    x, _ = ResidualUnit(
        16,
        320,
        kernel_size=kernel_size,
        kernel_initializer=kernel_initializer,
        dropout=dropout,
        activation=activation,
    )([x, y])

    x = GlobalAveragePooling1D(name="embed")(x)
    for i, projection_size in enumerate(projection_layer_shapes[:-1]):
        x = Dense(
            projection_size,
            activation="relu",
            kernel_initializer=kernel_initializer,
            name=f"projection_{i}",
        )(x)
    diagn = Dense(
        projection_layer_shapes[-1],
        kernel_initializer=kernel_initializer,
        name="projection",
    )(x)
    model = Model(
        signal,
        diagn,
        name=f"ribeiro_{activation}_{'_'.join(map(str, projection_layer_shapes))}",
    )
    return model


def simclr_loss(
    _,
    hidden,  # batch size (n) x 320 matrix
):
    """
    https://arxiv.org/abs/2002.05709
    https://github.com/google-research/simclr/
    """
    temperature = 0.1
    large_num = 1e9
    hidden = tf.math.l2_normalize(hidden, -1)  # n x 320
    # n / 2 x 320, n / 2 x 320
    hidden1, hidden2 = tf.split(
        hidden,
        2,
        0,
    )  # hidden is created from 2 * batch_size tensors
    # n / 2
    batch_size = tf.shape(hidden1)[0]
    # n / 2, n
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    # n / 2, n / 2 (identity matrix)
    masks = tf.one_hot(
        tf.range(batch_size),
        batch_size,
    )  # masks diagonals, aka self similarities

    # n / 2 x n / 2 (matrix of cosine similarities between first half of batch)
    logits_aa = tf.matmul(hidden1, hidden1, transpose_b=True) / temperature
    # remove self similarities
    logits_aa = logits_aa - masks * large_num

    # same thing for second half of batch
    logits_bb = tf.matmul(hidden2, hidden2, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * large_num

    # n / 2 x n / 2 (cosine similarity between first and second half of batch)
    logits_ab = tf.matmul(hidden1, hidden2, transpose_b=True) / temperature
    logits_ba = tf.matmul(hidden2, hidden1, transpose_b=True) / temperature
    loss_a = tf.compat.v1.losses.softmax_cross_entropy(
        # n / 2 x n, n / 2 x (n / 2 + n / 2)
        labels,
        tf.concat([logits_ab, logits_aa], 1),
    )
    loss_b = tf.compat.v1.losses.softmax_cross_entropy(
        labels,
        tf.concat([logits_ba, logits_bb], 1),
    )
    return tf.add(loss_a, loss_b)


def simclr_accuracy(_, hidden):
    hidden = tf.math.l2_normalize(hidden, -1)
    large_num = 1e9
    # hidden is created from 2 * batch_size tensors
    hidden1, hidden2 = tf.split(hidden, 2, 0)
    batch_size = tf.shape(hidden1)[0]
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(
        tf.range(batch_size),
        batch_size,
    )  # masks diagonals, aka self similarities
    logits_aa = tf.matmul(hidden1, hidden1, transpose_b=True)
    logits_aa = logits_aa - masks * large_num
    logits_bb = tf.matmul(hidden2, hidden2, transpose_b=True)
    logits_bb = logits_bb - masks * large_num
    logits_ab = tf.matmul(hidden1, hidden2, transpose_b=True)
    logits_ba = tf.matmul(hidden2, hidden1, transpose_b=True)
    loss_a = tf.keras.metrics.categorical_accuracy(
        labels,
        tf.concat([logits_ab, logits_aa], 1),
    )
    loss_b = tf.keras.metrics.categorical_accuracy(
        labels,
        tf.concat([logits_ba, logits_bb], 1),
    )
    return tf.add(tf.reduce_mean(loss_a), tf.reduce_mean(loss_b)) / 2
