import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile(Transformer):
    """
    Selects features based on a specified percentile of a scoring function.

    Keeps at most floor(n_features * percentile / 100) highest-scoring features.
    NaN scores are excluded, even at 100 percent. Ties retain the earliest
    features in their original order; infinite scores remain eligible.

    Args:
        percentile (float): The percentile for selecting features, must be an integer between 0 and 100.
        score_func (callable, optional): The scoring function used for feature evaluation. Defaults to f_classification.

    Raises
        ValueError: If the percentile is not an integer between 0 and 100.

    Examples:
        selector = SelectPercentile(percentile=50)
    """
        
    def __init__(self, percentile: float, score_func: callable = f_classification, **kwargs):
        """
        Selects features from the given percentile of a score function and returns a new Dataset object with the selected features
        Parameters
        ----------
        percentile: float
            Percentile for selecting features

        score_func: callable, optional
            Variance analysis function. Use the f_classification by default for
        """
        super().__init__(**kwargs)
        if not isinstance(percentile, int) or not (0 <= percentile <= 100):
            raise ValueError("Percentile must be an integer between 0 and 100")
        self.percentile = percentile
        
        self.score_func = score_func
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        Estimate the F and P values for each feature using the scoring function
        Parameters
        ----------
        dataset: Dataset
            - Dataset object where features are selected
        
        Returns
        -------
        self: object
            - Returns self instance with the F and P values for each feature calculated using the score function.
        """
        self.F,self.p = self.score_func(dataset)

        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selects features with the highest F value up to the specified percentile
        Parameters
        ----------
        dataset: Dataset
            - Dataset object where features are selected
        
        Returns
        -------
        dataset: Dataset
            - A Dataset object with the selected features
        """
        
        # NaN scores cannot rank a feature. Infinities remain meaningful scores.
        # Keep at most floor(n_features * percentile / 100), with stable ties.
        scores = np.asarray(self.F, dtype=float)
        eligible = np.flatnonzero(~np.isnan(scores))
        ranked = eligible[np.argsort(-scores[eligible], kind='stable')]
        max_features = int(len(scores) * self.percentile / 100)
        mask = np.zeros(len(scores), dtype=bool)
        mask[ranked[:max_features]] = True

        features = np.array(dataset.features)[mask]
        

        return Dataset(X=dataset.X[:, mask], y=dataset.y, features=list(features), label=dataset.label)
    


