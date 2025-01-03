import numpy as np
from si.metrics.mse import mse

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Root Mean Square Error.

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset

    Returns
    -------
    rmse: float
        The root mean squared error
    """
    return np.sqrt(mse(y_true, y_pred))