"""Minimal metrics utilities required for import."""
import numpy as np


def fitness(x):
    """Return fitness for hyperparameter evolution.

    Args:
        x (array-like): Metrics array where first four columns correspond
            to [precision, recall, mAP@0.5, mAP@0.5:0.95].
    Returns:
        numpy.ndarray: Weighted fitness value per row.
    """
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    w = np.array([0.0, 0.0, 0.1, 0.9])
    return (x[:, :4] * w).sum(1)
