import tensorflow as tf
import tensorflow_hub as hub


HIDDEN_UNITS = 256
DROPOUT_RATE = 0.5


def create_video_encoder(
    path='https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3',
    input_shape=(40, 224, 224, 3),
    trainable=True
    ):
    
    inputs = tf.keras.layers.Input(
        shape=input_shape,
        dtype=tf.float32,
        name='image'
    )
    
    movinet = hub.KerasLayer(path, trainable=trainable)
    outputs = movinet({
        'image': inputs
    })

    model = tf.keras.Model(inputs, outputs, name='movinet')
    
    return model


def create_classifier(encoder, trainable=True, input_shape=(224, 224, 3), categories={'output': 10}):

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = tf.keras.Input(shape=input_shape, name='image')
    features = encoder(inputs)
    features = tf.keras.layers.Dropout(DROPOUT_RATE)(features)
    features = tf.keras.layers.Dense(HIDDEN_UNITS, activation="relu")(features)
    features = tf.keras.layers.Dropout(DROPOUT_RATE)(features)
    outputs = []
    for category, n_classes in categories.items():
        activation = 'softmax' if n_classes > 1 else 'sigmoid'
        outputs.append(tf.keras.layers.Dense(n_classes, name=category, activation=activation)(features))
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="classifier")
    return model
