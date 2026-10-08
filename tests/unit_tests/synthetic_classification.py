"""Small, separable cases shared by classifier and model-selection tests."""
import numpy as np
from si.data.dataset import Dataset
from si.models.logistic_regression import LogisticRegression


def training_data():
    return Dataset(np.tile([[-2., -1.], [-2., 1.], [2., -1.], [2., 1.]], (3, 1)),
                   y=np.tile([0, 0, 1, 1], 3), features=['signal', 'noise'], label='class')

def evaluation_data():
    return Dataset(np.array([[-3., 0.], [3., 0.], [-4., .5], [4., -.5]]),
                   y=np.array([0, 1, 0, 1]), features=['signal', 'noise'], label='class')

def logistic_model():
    return LogisticRegression(l2_penalty=.1, alpha=.1, max_iter=200)
