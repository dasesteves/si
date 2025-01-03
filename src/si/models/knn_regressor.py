from typing import Callable

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor(Model):
    """
    KNN Regressor
    The k-Nearest Neighbors regressor predicts continuous values based on the mean
    of the k nearest neighbors in the training data.

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
    def __init__(self, k: int = 3, distance: Callable = euclidean_distance):
        super().__init__()
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        Stores the training dataset.

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

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts target values for dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict

        Returns
        -------
        predictions: np.ndarray
            The predicted values
        """
        predictions = np.zeros(dataset.shape()[0])
        
        for i, sample in enumerate(dataset.X):
            distances = self.distance(sample, self.dataset.X)
            k_nearest_indices = np.argsort(distances)[:self.k]
            predictions[i] = np.mean(self.dataset.y[k_nearest_indices])
            
        return predictions

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Returns the RMSE score for the predictions.

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate
        predictions: np.ndarray
            The predicted values

        Returns
        -------
        rmse: float
            The RMSE score
        """
        return rmse(dataset.y, predictions)


if __name__ == '__main__':
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split

    # Load and split dataset
    dataset = Dataset.from_random(600, 100, 2)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)

    # Train and evaluate model
    knn = KNNRegressor(k=3)
    knn.fit(train_dataset)
    score = knn.score(test_dataset)
    print(f'The RMSE score is: {score}')