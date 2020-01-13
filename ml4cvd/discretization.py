"""Functions for discretizing continuous data
"""

import logging
import numpy as np

from ml4cvd.TensorMap import TensorMap


def discretization_from_boundaries(boundaries: [float]):
    if boundaries:
        logging.info(f'Discretization will be applied with bin boundaries: {boundaries}')

        def discretization(tensor: np.ndarray):
            return np.digitize(tensor, bins=boundaries)
        return discretization
    else:
        logging.info('No discretization boundaries specified so no discretization will be applied.')
        return lambda tm, t: t
