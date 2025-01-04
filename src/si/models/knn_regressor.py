from typing import Callable, Union

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor(Model):
    """
    KNN Regressor
    This regression method is a non-parametric machine learning method suitable for regression problems.
    It classifies new sample based on a similarity measure, predicting the class of the new sample by looking at the values of the k-nearest samples in the training data.
    The k-Nearest Neighbors regressor predicts continuous values based on the mean of the k nearest neighbors in the training data.

    Parameters
    ----------
    k: int
        The number of nearest neighbors to use
    distance: Callable
        The distance function to use

    Attributes
    ----------
    dataset: Dataset
        The training dataset
    """
    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        """
        Initialize the KNN classifier
        Parameters
        ----------
        k: int
            The number of k nearest example to consider
        distance: Callable
            Function that calculates the distance between a sample and the samples
            in the training dataset
        """        
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        Stores the training dataset. Fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: KNNRegressor
            The fitted model
        """
        self.dataset = dataset
        return self

    def _get_closest_value(self, sample: np.ndarray) -> Union[int, float]:
        """
        It returns the closest label of the given sample
        Parameters
        ----------
        sample: np.ndarray
            The sample to get the closest value of
        Returns
        -------
        value: int or float
            The closest value
        """

        # compute the distance between the sample and the training dataset
        distances = self.distance(sample, self.dataset.X)

        # get the k nearest neighbors
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # get the values of the k nearest neighbors
        k_nearest_neighbors_label_values = self.dataset.y[k_nearest_neighbors]

        # get the average value of the k nearest neighbors
        value = np.sum(k_nearest_neighbors_label_values) / self.k

        return value

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict Label Values for a Dataset
        Generates predictions for the label values of the provided testing dataset.
        Parameters
        ----------
        dataset : Dataset
            The dataset for which label values are to be predicted.
        Returns:
        -----------
        predictions : np.ndarray
            An array containing the predicted label values for the testing dataset.
        """
        # compute the predictions for each row(sample) of the testing dataset
        predictions = np.apply_along_axis(self._get_closest_value, axis=1, arr=dataset.X)
        return predictions

    def _score(self, dataset: Dataset, predictions:np.ndarray) -> float:
        """
        It calculates the root mean square error (RMSE) between the actual and predicted values

        Parameters
        ----------
        dataset: Dataset
            The dataset containing the actual values (testing dataset)
        predictions: np.ndarray
            The predicted values for the testing dataset

        Returns
        -------
        score: float
            The RMSE score
        """
        return rmse(dataset.y,predictions)