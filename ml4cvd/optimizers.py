from keras import optimizers
from keras_radam import RAdam


def get_optimizer(name: str):
    name = str.lower(name)
    try:
        opt = optimizers.get(name)
        return opt
    except ValueError:
        print('Using optimizer not in keras')
    return NON_KERAS_OPTIMIZERS[name]


NON_KERAS_OPTIMIZERS = {
    'radam': RAdam,
}
