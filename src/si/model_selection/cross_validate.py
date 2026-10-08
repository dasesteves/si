from typing import List

import numpy as np

from si.data.dataset import Dataset


def k_fold_cross_validation(model, dataset: Dataset, scoring: callable = None, cv: int = 3,
                            seed: int = None) -> List[float]:
    """
    Perform k-fold cross-validation on the given model and dataset.

    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    scoring: Callable
        The scoring function to use. If None, the model's score method will be used.
    cv: int
        The number of cross-validation folds, between 2 and the sample count.
    seed: int
        The seed to use for the random number generator.

    Returns
    -------
    scores: List[float]
        The scores of the model on each fold. Every sample is tested once;
        fold sizes differ by at most one sample.

    Raises
    ------
    ValueError
        If cv is not an integer in the valid range.
    """
    num_samples = dataset.X.shape[0]
    if (not isinstance(cv, (int, np.integer)) or isinstance(cv, (bool, np.bool_))
            or not 2 <= cv <= num_samples):
        raise ValueError("cv must be an integer between 2 and the number of samples.")
    scores = []

    # Create an array of indices to shuffle the data
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    folds = np.array_split(indices, cv)
    for fold, test_indices in enumerate(folds):
        # Split the data into training and testing sets
        train_indices = np.concatenate(folds[:fold] + folds[fold + 1:])

        dataset_train = Dataset(dataset.X[train_indices], dataset.y[train_indices],
                                features=dataset.features, label=dataset.label)
        dataset_test = Dataset(dataset.X[test_indices], dataset.y[test_indices],
                               features=dataset.features, label=dataset.label)

        # Fit the model on the training set and score it on the test set
        model.fit(dataset_train)
        fold_score = scoring(dataset_test.y, model.predict(dataset_test)) if scoring is not None else model.score(
            dataset_test)
        scores.append(fold_score)

    return scores
