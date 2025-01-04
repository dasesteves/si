

from si.data.dataset import Dataset
import scipy


def f_classification(dataset: Dataset) -> tuple:
    """Performs a 1-way ANOVA test.
    Calculates the F-statistic and p-value for a 1-way ANOVA test.
    The test assesses whether the means of two or more groups are statistically different.

    Args:
        dataset (Dataset): The dataset containing the groups to compare.

    Returns:
        tuple: A tuple containing the F-statistic and the p-value.
    """
    
    classes = dataset.get_classes()

    groups = []
    for class_ in classes:
        mask = dataset.y == class_
        group = dataset.X[mask, :]
        groups.append(group)

    return scipy.stats.f_oneway(*groups)
