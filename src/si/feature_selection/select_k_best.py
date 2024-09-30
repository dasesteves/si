import numpy as np

from si.base.transformer import Transformer
from si.statistics.f_classification import f_classification 
from si.data.dataset import Dataset

class SelectKBest(Transformer):
    
    def __init__(self, k: int, score_func=callable, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.score_func = score_func
        self.F = None
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
