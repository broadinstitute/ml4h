from tensorflow.keras import optimizers
from tensorflow_addons.optimizers import RectifiedAdam, TriangularCyclicalLearningRate, Triangular2CyclicalLearningRate


def get_optimizer(name: str, lr: float, steps_per_epoch: int = None, lr_schedule: str = None, optimizer_kwargs=None):
    if not optimizer_kwargs:
        optimizer_kwargs = {}
    name = str.lower(name)
    lr = _get_learning_rate_schedule(lr, lr_schedule, steps_per_epoch)
    try:
        opt = optimizers.get(name)
        opt.__init__(lr, **optimizer_kwargs)
        return opt
    except ValueError:
        pass
    if name in NON_KERAS_OPTIMIZERS:
        return NON_KERAS_OPTIMIZERS[name](lr, **optimizer_kwargs)
    raise ValueError(f'Unknown optimizer {name}')


def _get_learning_rate_schedule(lr: float, lr_schedule: str = None, steps_per_epoch: int = None):
    if lr_schedule is None:
        return lr
    if lr_schedule == 'triangular':
        return TriangularCyclicalLearningRate(initial_learning_rate=lr / 5, maximal_learning_rate=lr,
                                              step_size=steps_per_epoch)
    if lr_schedule == 'triangular2':
        return Triangular2CyclicalLearningRate(initial_learning_rate=lr / 5, maximal_learning_rate=lr,
                                               step_size=steps_per_epoch)
    else:
        raise ValueError(f'Learning rate schedule {lr_schedule} unknown.')


NON_KERAS_OPTIMIZERS = {
    'radam': RectifiedAdam,
}
