import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse

class LassoRegression(Model):
    """
    Lasso Regression is a linear regression model with L1 regularization.
    The L1 regularization term penalizes the absolute values of the coefficients,
    encouraging sparsity by shrinking or eliminating less important features.

    Parameters:
    ----------
    l1_penalty : float, default=1
        The L1 regularization parameter.
    max_iter : int, default=1000
        Maximum number of iterations for gradient descent.
    patience : int, default=5
        Early stopping patience: number of iterations without improvement.
    scale : bool, default=True
        Whether to scale the dataset before fitting the model.
    """

    def __init__(self, l1_penalty=1, max_iter=1000, patience=5, scale=True, **kwargs):
        super().__init__(**kwargs)
        self.l1_penalty = l1_penalty
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        # Model attributes
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _fit(self, dataset: Dataset) -> 'LassoRegression':
        """
        Fit the Lasso Regression model to the given dataset.

        Parameters:
        ----------
        dataset : Dataset
            The training dataset containing features (X) and target (y).

        Returns:
        -------
        self : LassoRegression
            The fitted Lasso Regression model.
        """
        X = dataset.X
        if self.scale:
            self.mean = np.nanmean(X, axis=0)
            self.std = np.nanstd(X, axis=0)
            X = (X - self.mean) / self.std

        m, n = dataset.shape()
        self.theta = np.zeros(n)
        self.theta_zero = 0

        early_stopping = 0
        for i in range(self.max_iter):
            if early_stopping >= self.patience:
                break

            y_pred = np.dot(X, self.theta) + self.theta_zero

            # Update coefficients using soft-thresholding
            for feature in range(n):
                residual = np.dot(X[:, feature], dataset.y - y_pred + X[:, feature] * self.theta[feature])
                self.theta[feature] = self.soft_threshold(residual, self.l1_penalty) / np.sum(X[:, feature] ** 2)

            # Update intercept
            self.theta_zero = np.mean(dataset.y - np.dot(X, self.theta))

            # Calculate and store cost
            cost = self.cost(dataset)
            self.cost_history[i] = cost
            if i > 0 and cost >= self.cost_history[i - 1]:
                early_stopping += 1
            else:
                early_stopping = 0

        return self

    def cost(self, dataset: Dataset) -> float:
        """
        Compute the cost function for Lasso Regression with L1 regularization.

        Parameters:
        ----------
        dataset : Dataset
            The dataset to compute the cost on.

        Returns:
        -------
        cost : float
            The computed cost value.
        """
        y_pred = self.predict(dataset)
        mse_cost = np.mean((dataset.y - y_pred) ** 2) / 2
        l1_cost = self.l1_penalty * np.sum(np.abs(self.theta))
        return mse_cost + l1_cost

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the target values for the given dataset.

        Parameters:
        ----------
        dataset : Dataset
            The dataset containing the features to predict.

        Returns:
        -------
        predictions : np.ndarray
            The predicted target values.
        """
        X = (dataset.X - self.mean) / self.std if self.scale else dataset.X
        return np.dot(X, self.theta) + self.theta_zero

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Compute the Mean Squared Error (MSE) score for the model.

        Parameters:
        ----------
        dataset : Dataset
            The dataset to compute the MSE on.
        predictions : np.ndarray
            The predicted values.

        Returns:
        -------
        mse_score : float
            The computed Mean Squared Error score.
        """
        return mse(dataset.y, predictions)

    #Used @staticmethod for the soft_threshold method as it does not rely on instance attributes
    @staticmethod
    def soft_threshold(residual: float, l1_penalty: float) -> float:
        """
        Perform soft-thresholding for Lasso regularization.

        Parameters:
        ----------
        residual : float
            The residual value for a feature.
        l1_penalty : float
            The L1 regularization parameter.

        Returns:
        -------
        soft_threshold : float
            The updated value after applying soft-thresholding.
        """
        if residual > l1_penalty:
            return residual - l1_penalty
        elif residual < -l1_penalty:
            return residual + l1_penalty
        else:
            return 0.0
