import numpy as np
from si.metrics.accuracy import accuracy
from si.data.dataset import Dataset
from si.base.model import Model


class StackingClassifier(Model):
    """
    A stacking classifier is an ensemble learning technique that combines multiple base models to create a more powerful and accurate predictive model. It uses a meta-learning approach where a meta-model learns to combine the predictions of the base models.

    How it Works:

    1. Base Models:
    Multiple base models (e.g., decision trees, random forests, support vector machines, neural networks) are trained on the same training dataset. Each model learns patterns and relationships within the data and makes predictions on a given dataset.

    2. Generating Predictions:
    Each base model makes predictions on the same dataset, generating a set of predictions for each data point.

    3. Creating a New Dataset:
    The predictions from all base models for each data point are combined into a new dataset. Each row in this new dataset represents a data point, and each column represents the prediction from a specific base model.

    4. Training the Meta-Model:
    A meta-model (e.g., logistic regression, another decision tree, or a neural network) is trained on this new dataset. The meta-model learns to combine the predictions from the base models to make a final, more accurate prediction.

    Key Advantages of Stacking Classifiers:

    - Improved Performance: Stacking often leads to significantly improved performance compared to individual models, especially when the base models are diverse and complementary.
    - Reduced Overfitting: By combining multiple models, stacking can help mitigate overfitting, which occurs when a model is too complex and fits the training data too closely.
    - Enhanced Robustness: Stacking can make the overall model more robust to noise and variations in the data.
    - Better Generalization: Stacking can improve the model's ability to generalize to unseen data.
    """


    def __init__(self, models:list, final_model,**kwargs):
        """
        Initialize the Stacking Classifier ensemble model
        Parameters
        ----------
        models : list
            Array-like of base models to be combined in the ensemble.
            Each model should be an instance of a Model class.
        final_model :
            Model to be used as the meta-model and create the final predictions.
            The model must be an instance of a Model class
        """

        # parameters
        super().__init__(**kwargs)
        self.models = models
        self.final_model = final_model

        # attributes
        self.new_dataset = None

    def _fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Fit the StackingClassifier ensemble model to the given training data.
        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the model to (training dataset)
        Returns
        -------
        self : StackingClassifier
            The fitted model
        """
        # Fit the base models
        for model in self.models:
            model.fit(dataset)

        # Genarate the base models predictions
        base_predictions = [model.predict(dataset) for model in self.models]
        base_predictions = np.array(base_predictions).T

        # Create a new dataset with the base models predictions
        self.new_dataset = Dataset(X=base_predictions, y=dataset.y, features = [f"{model}" for model in self.models] ,label= dataset.label)

        # Fit the final model (meta-model)
        self.final_model.fit(self.new_dataset)

        return self

    def _predict(self, dataset:Dataset) -> np.ndarray:
        """
        Predict class labels for samples in X.
        Parameters
        ----------
        dataset : Dataset
            The dataset to make predictions on
        Returns
        -------
        np.ndarray
            The predicted class labels for the samples in X
        """
        # Base models predictions
        base_predictions = [model.predict(dataset) for model in self.models]
        base_predictions = np.array(base_predictions).T

        # Create a new dataset with the base models predictions
        new_dataset = Dataset(X=base_predictions, y=dataset.y, features = [f"{model}" for model in self.models] ,label=dataset.label)

        # Make predictions with the final model (meta-model)
        return self.final_model.predict(new_dataset)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Returns the mean accuracy on the given test data and labels.
        Parameters
        ----------
        dataset : Dataset
            The test data.
        predictions: np.ndarray
            Predictions
        Returns
        -------
        score : float
            Mean accuracy
        """
        return accuracy(dataset.y, predictions)