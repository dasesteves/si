import numpy as np
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.data.dataset import Dataset
from si.base.model import Model

class RandomForestClassifier(Model):
    """
    Random Forest Classifier is an ensemble machine learning model that combines
    multiple decision trees to enhance prediction accuracy, robustness, and reduce overfitting.

    Parameters:
    ----------
    n_estimators : int, default=100
        The number of decision trees in the forest.
    max_features : int, optional
        The maximum number of features to consider when splitting a node. If None, sqrt(n_features) is used.
    min_sample_split : int, default=5
        The minimum number of samples required to split a node.
    max_depth : int, default=10
        The maximum depth of the decision trees in the forest.
    mode : str, default="gini"
        The criterion for calculating information gain. Options: "gini", "entropy".
    seed : int, default=123
        The random seed for reproducibility.
    """

    def __init__(self, n_estimators=100, max_features=None, min_sample_split=5, max_depth=10, mode="gini", seed=123, **kwargs):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.seed = seed

        # Validate mode
        if isinstance(mode, str) and mode.lower() in {"gini", "entropy"}:
            self.mode = mode
        else:
            raise ValueError(f'Invalid mode: {mode}. Valid modes are: "gini", "entropy"')

        # Model attributes
        self.trees = []

    def _fit(self, dataset: Dataset) -> "RandomForestClassifier":
        """
        Train the Random Forest by fitting multiple decision trees on bootstrapped datasets.

        Parameters:
        ----------
        dataset : Dataset
            The dataset used to train the random forest.

        Returns:
        -------
        self : RandomForestClassifier
            The trained random forest model.
        """
        np.random.seed(self.seed)
        if self.max_features is None:
            self.max_features = int(np.sqrt(dataset.shape()[1]))

        for _ in range(self.n_estimators):
            # Create a bootstrap dataset
            sample_indices = np.random.choice(dataset.shape()[0], size=dataset.shape()[0], replace=True)
            feature_indices = np.random.choice(dataset.shape()[1], size=self.max_features, replace=False)
            selected_features = [dataset.features[idx] for idx in feature_indices]

            bootstrap_dataset = Dataset(
                X=dataset.X[sample_indices][:, feature_indices],
                y=dataset.y[sample_indices],
                features=selected_features,
                label=dataset.label
            )

            # Train a decision tree on the bootstrap dataset
            tree = DecisionTreeClassifier(
                min_sample_split=self.min_sample_split,
                max_depth=self.max_depth,
                mode=self.mode
            )
            tree.fit(bootstrap_dataset)

            # Store the trained tree and its features
            self.trees.append((selected_features, tree))

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict target values using the trained random forest.

        Parameters:
        ----------
        dataset : Dataset
            The dataset to predict target values for.

        Returns:
        -------
        predictions : np.ndarray
            The predicted target values.
        """
        # Collect predictions from each tree
        tree_predictions = []
        for features, tree in self.trees:
            # Subset the dataset to match the features used by the current tree
            feature_mask = [feature in features for feature in dataset.features]
            subset_dataset = Dataset(
                X=dataset.X[:, feature_mask],
                y=dataset.y,
                features=features,
                label=dataset.label
            )
            tree_predictions.append(tree.predict(subset_dataset))

        # Aggregate predictions across trees
        predictions = np.array(tree_predictions).T

        # Determine the most frequent prediction for each sample
        final_predictions = np.apply_along_axis(
            lambda row: np.bincount(row).argmax(),
            axis=1,
            arr=predictions
        )

        return final_predictions

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculate the accuracy of the model on the dataset.

        Parameters:
        ----------
        dataset : Dataset
            The dataset to evaluate.
        predictions : np.ndarray
            Predicted values for the dataset.

        Returns:
        -------
        score : float
            The accuracy of the model.
        """
        return accuracy(dataset.y, predictions)
