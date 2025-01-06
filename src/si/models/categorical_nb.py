#nao esta funcional erro log...

import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset

class CategoricalNB(Model):
    """
    Categorical Naive Bayes classifier for binary features.

    Parameters
    ----------
    smoothing : float, default=1.0
        Laplace smoothing to avoid zero probabilities.

    Attributes
    ----------
    class_prior : np.ndarray
        Prior probabilities for each class.
    feature_probs : np.ndarray
        Probabilities for each feature for each class being present / being 1.
    """
    def __init__(self, smoothing: float = 1.0):
        super().__init__()
        self.smoothing = smoothing
        self.class_prior = None
        self.feature_probs = None

    def _fit(self, dataset: Dataset) -> 'CategoricalNB':
        n_samples, n_features = dataset.X.shape
        n_classes = len(np.unique(dataset.y))

        class_counts = np.zeros(n_classes)
        feature_counts = np.zeros((n_classes, n_features))
        self.class_prior = np.zeros(n_classes)

        for c in range(n_classes):
            class_samples = dataset.X[dataset.y == c]
            class_counts[c] = class_samples.shape[0]
            feature_counts[c, :] = np.sum(class_samples, axis=0)

        self.class_prior = (class_counts + self.smoothing) / (n_samples + n_classes * self.smoothing)
        self.feature_probs = (feature_counts + self.smoothing) / (class_counts[:, None] + 2 * self.smoothing)

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        n_samples = dataset.X.shape[0]
        n_classes = len(self.class_prior)
        log_class_prior = np.log(self.class_prior)
        log_feature_probs = np.log(self.feature_probs)
        log_feature_probs_neg = np.log(1 - self.feature_probs)

        predictions = np.zeros(n_samples)

        for i in range(n_samples):
            sample = dataset.X[i]
            log_probs = np.zeros(n_classes)

            for c in range(n_classes):
                log_probs[c] = log_class_prior[c] + np.sum(sample * log_feature_probs[c] + (1 - sample) * log_feature_probs_neg[c])

            predictions[i] = np.argmax(log_probs)

        return predictions

    def _score(self, dataset: Dataset) -> float:
        predictions = self.predict(dataset)
        accuracy = np.mean(predictions == dataset.y)
        return accuracy