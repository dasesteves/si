<<<<<<< HEAD
import numpy as np

from si.base.transformer import Transformer
from si.statistics.f_classification import f_classification 
from si.data.dataset import Dataset

class SelectKBest(Transformer):
    
    def __init__(self, k: int, score_func=callable, **kwargs):
=======
from typing import Callable

import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectKBest(Transformer):
    """
    Select features according to the k highest scores.
    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value between label/feature for classification tasks.
        - f_regression: F-value obtained from F-value of r's pearson correlation coefficients for regression tasks.

    Parameters
    ----------
    score_func: callable
        Function taking dataset and returning a pair of arrays (scores, p_values)
    k: int, default=10
        Number of top features to select.

    Attributes
    ----------
    F: array, shape (n_features,)
        F scores of features.
    p: array, shape (n_features,)
        p-values of F-scores.
    """

    def __init__(self, score_func: Callable = f_classification, k: int = 10, **kwargs):
        """
        Select features according to the k highest scores.

        Parameters
        ----------
        score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values)
        k: int, default=10
            Number of top features to select.
        """
>>>>>>> 1277ea0d469379272f7a89a41c1f53b396d113e1
        super().__init__(**kwargs)
        self.k = k
        self.score_func = score_func
        self.F = None
<<<<<<< HEAD
        self.p = None        

    def _fit(self, dataset: Dataset) -> 'SelectKBest':
        self.F, self.p = self.score_func(dataset)
        return self
    
    def fit(self, dataset: Dataset) -> 'SelectKBest':
        return self._fit(dataset)
        
    def _transform(self, dataset: Dataset) -> Dataset:
        if self.F is None or self.p is None:
            raise ValueError("Você deve chamar fit() antes de transform()")
        
        idx = np.argsort(self.F)[::-1]  # Ordena em ordem decrescente
        mask = idx[:self.k]  # Seleciona os k melhores
        new_X = dataset.X[:, mask]
        new_features = [dataset.features[i] for i in mask]
        
        return Dataset(X=new_X, features=new_features, y=dataset.y, label=dataset.label)
=======
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectKBest':
        """
        It fits SelectKBest to compute the F scores and p-values.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        self: object
            Returns self.
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        It transforms the dataset by selecting the k highest scoring features.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the k highest scoring features.
        """
        idxs = np.argsort(self.F)[-self.k:]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)


if __name__ == '__main__':
    from si.data.dataset import Dataset

    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")

    selector = SelectKBest(k=2)
    selector = selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset.features)
>>>>>>> 1277ea0d469379272f7a89a41c1f53b396d113e1
