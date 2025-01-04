from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test


def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = None) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into train and test sets while preserving class proportions.

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
        
    A tuple where the first element is the training dataset and the second element is the testing dataset
    """
    if test_size <0 or test_size > 1:
        raise ValueError("Test size must be between 0 and 1")

    # set random state
    np.random.seed(random_state)

    # get unique labels
    labels,counts = np.unique(dataset.y, return_counts= True)

    # initialize empty lists to store indices for training and testing sets
    train_idx = []
    test_idx = []

    # spliting the data based on the labels
    for label,count in zip(labels,counts):

        idx = np.where(dataset.y == label)[0]
        train_size = int(count * (1 - test_size))
        np.random.shuffle(idx)
        train_idx.extend(idx[:train_size])
        test_idx.extend(idx[train_size:])       

    train_dataset = Dataset(X=dataset.X[train_idx,:], y= dataset.y[train_idx], features= dataset.features, label= dataset.label)
    test_dataset = Dataset( X = dataset.X[test_idx,:], y= dataset.y[test_idx], features= dataset.features, label= dataset.label)    

    return train_dataset, test_dataset