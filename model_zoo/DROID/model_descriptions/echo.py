import numpy as np
import tensorflow as tf

# from official.common import flags as tfm_flags
from official.vision.beta.projects.movinet.modeling import movinet, movinet_model

learning_rate = 0.0001
hidden_units = 256
dropout_rate = 0.5
temperature = 0.05


class DDGenerator:
    def __init__(self, input_dd, output_dd, fill_empty=False):
        self.input_dd = input_dd
        self.output_dd = output_dd
        self.fill_empty = fill_empty

    def __call__(self, sample_ids):
        ret_input = []
        ret_output = []
        for sample_id in sample_ids:
            ret_input.append(
                self.input_dd.get_raw_data(sample_id)
            )
            if self.output_dd is not None:
                ret_output.append(
                    self.output_dd.get_raw_data(sample_id)
                )
            if self.fill_empty:
                ret_output.append(np.NaN)

        if self.output_dd is None and self.fill_empty == False:
            yielded = (ret_input,)
        else:
            yielded = (ret_input, ret_output)
        yield yielded


def create_movinet_classifier(
        n_input_frames,
        batch_size,
        checkpoint_dir,
        num_classes,
        freeze_backbone=False
):
    backbone = movinet.Movinet(model_id='a2')
    model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
    model.build([1, 1, 1, 1, 3])
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(checkpoint_path)
    status.assert_existing_objects_matched()

    model = movinet_model.MovinetClassifier(
        backbone=backbone,
        num_classes=num_classes
    )
    model.build([batch_size, n_input_frames, 224, 224, 3])

    if freeze_backbone:
        for layer in model.layers[:-1]:
            layer.trainable = False
        model.layers[-1].trainable = True

    return model, backbone


def create_regressor(encoder, trainable=True, input_shape=(224, 224, 3), n_output_features=1):
    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = tf.keras.Input(shape=input_shape, name='image')
    features = encoder(inputs)
    features = tf.keras.layers.Dropout(dropout_rate)(features)
    features = tf.keras.layers.Dense(hidden_units, activation="relu")(features)
    features = tf.keras.layers.Dropout(dropout_rate)(features)
    outputs = tf.keras.layers.Dense(n_output_features, activation=None, name='echolab')(features)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="regressor")

    return model
