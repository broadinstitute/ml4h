from keras import optimizers
from keras_radam import RAdam


def get_optimizer(name: str, lr: float):
    name = str.lower(name)
    try:
        opt = optimizers.get(name)
        opt.__init__(learning_rate=lr)
        return opt
    except ValueError:
        print('Using optimizer not in keras')
    return NON_KERAS_OPTIMIZERS[name](lr)


NON_KERAS_OPTIMIZERS = {
    'radam': RAdam,
}
