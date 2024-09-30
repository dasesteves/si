import numpy as np
from scipy.stats import f_oneway
from si.data.dataset import Dataset  

def f_classification(dataset: Dataset) -> (tuple, tuple):
    """
    Analyze the variance of the dataset using F-values and p-values.

    Args:
    - dataset (Dataset): The Dataset object.

    Returns:
    - tuple: F-values
    - tuple: p-values
    """
    classes = dataset.get_classes()
    samples_by_class = [dataset.select_samples(class_) for class_ in classes]

    f_values = []
    p_values = []
    for i in range(len(samples_by_class)):
        for j in range(i + 1, len(samples_by_class)):
            f_value, p_value = f_oneway(samples_by_class[i], samples_by_class[j])
            f_values.append(f_value)
            p_values.append(p_value)

    return tuple(f_values), tuple(p_values)