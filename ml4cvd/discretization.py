"""Functions for discretizing continuous data
"""

from keras.utils import to_categorical
import logging
import numpy as np


def discretization_from_boundaries(boundaries: [float]):
    if boundaries:
        logging.info(f'Discretization will be applied with bin boundaries: {boundaries}')

        def discretization(tensor: np.ndarray):
            return to_categorical(np.digitize(tensor, bins=boundaries), num_classes=len(boundaries)+1)
        return discretization
    else:
        logging.info('No discretization boundaries specified so no discretization will be applied.')
        return lambda t: t
