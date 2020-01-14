"""Functions for discretizing continuous data
"""

from keras.utils import to_categorical
import logging
import numpy as np


class Discretization:

    def __init__(self, boundaries: [float]):
        self.boundaries = boundaries

    def __call__(self, tensor: np.ndarray):
        return to_categorical(np.digitize(tensor, bins=self.boundaries), num_classes=len(self.boundaries) + 1)

    def __len__(self):
        return len(self.boundaries)+1


# def discretization_from_boundaries(boundaries: [float]):
#     """Produces a discretization function that bins values according to the specified boundaries
#     """
#     logging.info(f'Discretization will be applied with bin boundaries: {boundaries}')
#
#     def discretization(tensor: np.ndarray):
#         return to_categorical(np.digitize(tensor, bins=boundaries), num_classes=len(boundaries)+1)
#     return discretization
