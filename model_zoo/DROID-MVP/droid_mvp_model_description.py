import numpy as np
import tensorflow as tf

# from official.common import flags as tfm_flags
#from official.vision.beta.projects.movinet.modeling import movinet, movinet_model

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

        if self.output_dd is not None and isinstance(ret_output[0], list):
            ret_output = [np.vstack([ret_output[i][j] for i in range(len(sample_ids))])
                          for j in range(len(ret_output[0]))]
            ret_output = tuple(ret_output)

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


# ---------- Adaptation for regression + classification ---------- #
def create_regressor_classifier(encoder, trainable=True, input_shape=(224, 224, 3), n_output_features=0, categories={},
                                category_order=None, add_dense={'regressor': False, 'classifier': False}):
#     for layer in encoder.layers:
#         layer.trainable = trainable

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
            # added a variable - category_order to make sure the ordering is correct
            # (dictionary items ordering is not necessarily consistent)
            activation = 'softmax'
            n_classes = categories[category]
            outputs.append(tf.keras.layers.Dense(n_classes, name='cls_'+category, activation=activation)(features))

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="regressor_classifier")

    return model
# ---------------------------------------------------------------- #


def train_model(
        model,
        train_loader,
        valid_loader,
        epochs,
        n_train_steps,
        n_valid_steps,
        output_folder,
        es_flags,
        class_weight=None
):
    tb_callback = tf.keras.callbacks.TensorBoard(f'{output_folder}/logs', profile_batch=[160, 170])
    es_callback = tf.keras.callbacks.EarlyStopping(monitor=es_flags['es_loss2monitor'],
                                                   patience=es_flags['es_patience'],
                                                   mode=es_flags['es_mode'])
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'{output_folder}/model/chkp',
        monitor=es_flags['es_loss2monitor'],
        save_best_only=True,
        save_weights_only=True,
        mode=es_flags['es_mode']
    )
    model.fit(
        train_loader,
        validation_data=valid_loader,
        callbacks=[tb_callback, es_callback, cp_callback],
        epochs=epochs,
        steps_per_epoch=n_train_steps,
        validation_steps=n_valid_steps,
        workers=1,
        max_queue_size=1,
        use_multiprocessing=False,
        class_weight=class_weight
    )

    model.load_weights(
        f'{output_folder}/model/chkp'
    )

    return model
