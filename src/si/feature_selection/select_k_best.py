import numpy as np

from si.base.transformer import Transformer
from si.statistics.f_classification import f_classification  
from si.data.dataset import Dataset

class SelectKBest(Transformer):
    
    def __init__(self, k : int, score_func = callable, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.score_func = score_func
        self.F = None
        self.p = None        
    def fit(self, dataset : Dataset) -> 'SelectKBest':
        self.F, self.p = self.score_func(dataset)
        return self
        
    def _transform(self, dataset : Dataset) -> Dataset:
        
        idx = np.argsort(self.F)
        mask = idx[-self.k:]
        new_X = self.dataset.X[:, mask]
        new_features = self.dataset.features[mask]
        
        return Dataset(X = new_X, features = new_features, y=self.dataset.y, label=dataset.label)
