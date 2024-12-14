import numpy as np
import tensorflow as tf
from official.vision.beta.projects.movinet.modeling import movinet, movinet_model

hidden_units = 256
dropout_rate = 0.5

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

def create_regressor_classifier(encoder, trainable=True, input_shape=(224, 224, 3), n_output_features=0, categories={},
                                category_order=None, add_dense={'regressor': False, 'classifier': False}):
    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = tf.keras.Input(shape=input_shape, name='image')
    features = encoder(inputs)
    features = tf.keras.layers.Dropout(dropout_rate)(features)
    features = tf.keras.layers.Dense(hidden_units, activation="relu")(features)
    features = tf.keras.layers.Dropout(dropout_rate)(features)

    outputs = []
    if n_output_features > 0:
        if add_dense['regressor']:
            features_reg = tf.keras.layers.Dense(hidden_units, activation="relu")(features)
            features_reg = tf.keras.layers.Dropout(dropout_rate)(features_reg)
            outputs.append(tf.keras.layers.Dense(n_output_features, activation=None, name='echolab')(features_reg))
        else:
            outputs.append(tf.keras.layers.Dense(n_output_features, activation=None, name='echolab')(features))
    if len(categories.keys()) > 0:
        if add_dense['classifier']:
            features = tf.keras.layers.Dense(hidden_units, activation="relu")(features)
            features = tf.keras.layers.Dropout(dropout_rate)(features)
        for category in category_order:
            activation = 'softmax'
            n_classes = categories[category]
            outputs.append(tf.keras.layers.Dense(n_classes, name='cls_'+category, activation=activation)(features))

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="regressor_classifier")

    return model

def rescale_droid_rv_outputs(droid_rv_output):
    droid_rv_output[0][0,0] =  droid_rv_output[0][0,0] * 15.51761856 + 64.43979878
    droid_rv_output[0][0,1] =  droid_rv_output[0][0,1] * 6.88963822 + 42.52320993


def rescale_droid_rvef_outputs(droid_rvef_output):
    droid_rvef_output[0][0,0] =  droid_rvef_output[0][0,0] * 8.658711 + 53.40699
    droid_rvef_output[0][0,1] =  droid_rvef_output[0][0,1] * 46.5734 + 130.8913
    droid_rvef_output[0][0,2] =  droid_rvef_output[0][0,2] * 31.6643 + 62.87321
    droid_rvef_output[0][0,3] =  droid_rvef_output[0][0,3] * 22.99643 + 47.18989
    return droid_rvef_output
