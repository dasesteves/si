from unittest import TestCase

import numpy as np

from si.data.dataset import Dataset
from si.decomposition.pca import PCA
from si.feature_selection.variance_threshold import VarianceThreshold


class TestTransformerFitGuard(TestCase):
    def test_transform_requires_fit_before_accessing_learned_state(self):
        dataset = Dataset(X=np.array([[0., 1.], [2., 1.]]), features=['varying', 'constant'])
        transformer = VarianceThreshold(threshold=0)
        with self.assertRaises(ValueError):
            transformer.transform(dataset)
        self.assertFalse(transformer.is_fitted())
        transformer.fit(dataset)
        output = transformer.transform(dataset)
        self.assertEqual(output.features, ['varying'])
        np.testing.assert_array_equal(output.X, [[0.], [2.]])

    def test_pca_boolean_state_remains_compatible(self):
        dataset = Dataset(X=np.array([[-2., -1.], [0., 2.], [2., -1.]]), features=['x', 'y'])
        transformer = PCA(n_components=1)
        with self.assertRaises(ValueError):
            transformer.transform(dataset)
        transformer.fit(dataset)
        self.assertTrue(transformer.is_fitted)
        output = transformer.transform(dataset)
        self.assertEqual(output.features, ['PC1'])
        np.testing.assert_allclose(np.abs(output.X[:, 0]), [2., 0., 2.])
