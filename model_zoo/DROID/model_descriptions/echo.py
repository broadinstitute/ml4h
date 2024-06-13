import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

# from official.common import flags as tfm_flags
from official.vision.beta.projects.movinet.modeling import movinet, movinet_model

learning_rate = 0.0001
hidden_units = 256
dropout_rate = 0.5
temperature = 0.05

ONE_HOT_2CLS = False

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
                # print(len(ret_output))
            if self.fill_empty:
                ret_output.append(np.NaN)

        if (self.output_dd is not None) and isinstance(ret_output[0], list):
            # print(f'Ouput is a list: {len(ret_output[0])}, {ret_ouput[0][0].shape}')
            ret_output = [np.vstack([ret_output[i][j] for i in range(len(sample_ids))])
                          for j in range(len(ret_output[0]))]
            ret_output = tuple(ret_output)

        if (self.output_dd is None) and self.fill_empty == False:
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


# ---------- Adaptation for regression + classification + Adaptation for survival loss ---------- #
def create_regressor_classifier(encoder, trainable=True, input_shape=(224, 224, 3), n_output_features=0, categories={},
                                category_order=None, add_dense={'regressor': False, 'classifier': False}, survival_shapes={}):
    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = tf.keras.Input(shape=input_shape, name='image')
    features = encoder(inputs)
    features = tf.keras.layers.Dropout(dropout_rate)(features)
    features = tf.keras.layers.Dense(hidden_units, activation="relu")(features)
    features = tf.keras.layers.Dropout(dropout_rate)(features)

    outputs = []
    if survival_shapes:
        for s_name in survival_shapes:
            outputs.append(tf.keras.layers.Dense(survival_shapes[s_name], activation='sigmoid', name='survival_'+s_name)(features))
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
            if ONE_HOT_2CLS:
                n_classes = categories[category]
                activation = 'softmax'
            else:
                n_classes = categories[category] if categories[category] > 2 else 1
                activation = 'softmax' if n_classes > 2 else 'sigmoid'
            outputs.append(tf.keras.layers.Dense(n_classes, name='cls_' + category, activation=activation)(features))

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
                                                   patience=es_flags['es_patience'])
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'{output_folder}/model/chkp',
        monitor=es_flags['es_loss2monitor'],
        save_best_only=True,
        save_weights_only=True,
        mode='min'
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


def survival_likelihood_loss(n_intervals):
    """Create custom Keras loss function for neural network survival model.

    This function is tightly coupled with the function _survival_tensor defined in tensor_from_file.py which builds the y_true tensor.

    Arguments
        n_intervals: the number of survival time intervals
    Returns
        Custom loss function that can be used with Keras
    """

    def loss(y_true, y_pred):
        """
        To play nicely with the Keras framework y_pred is the same shape as y_true.
        However, we only consider the first half (n_intervals) of y_pred.
        Arguments
            y_true: Tensor.
              First half of the values are 1 if individual survived that interval, 0 if not.
              Second half of the values are for individuals who failed, and are 1 for time interval during which failure occurred, 0 for other intervals.
              For example given n_intervals = 3 a sample with prevalent disease will have y_true [0, 0, 0, 1, 0, 0]
              a sample with incident disease occurring in the last time bin will have y_true [1, 1, 0, 0, 0, 1]
              a sample who is lost to follow up (censored) in middle time bin will have y_true [1, 0, 0, 0, 0, 0]
            y_pred: Tensor, predicted survival probability (1-hazard probability) for each time interval.
        Returns
            Vector of losses for this minibatch.
        """
        # print(y_true.shape)
        # print(y_pred.shape)
        failure_likelihood = 1. - (y_true[:, n_intervals:] * y_pred[:, 0:n_intervals])  # Loss only for individuals who failed
        survival_likelihood = y_true[:, 0:n_intervals] * y_pred[:, 0:n_intervals]  # Loss for intervals that were survived
        survival_likelihood += 1. - y_true[:, 0:n_intervals]  # No survival loss if interval was censored or failed
        return K.sum(-K.log(K.clip(K.concatenate((survival_likelihood, failure_likelihood)), K.epsilon(), None)), axis=-1)  # return -log likelihood

    return loss
