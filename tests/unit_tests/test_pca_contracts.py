from unittest import TestCase

import numpy as np

from si.data.dataset import Dataset
from si.decomposition.pca import PCA


class TestPCAContracts(TestCase):
    def setUp(self):
        # Mean [10, 20], covariance diag(4, 3), leading direction along x.
        self.X = np.array([[8., 19.], [10., 22.], [12., 19.]])
        self.dataset = Dataset(X=self.X.copy(), y=np.array([0, 1, 0]),
                               features=['x', 'y'], label='class')

    def test_fit_preserves_caller_data(self):
        PCA(n_components=1).fit(self.dataset)
        np.testing.assert_array_equal(self.dataset.X, self.X)

    def test_fit_transform_centers_once_and_preserves_labels(self):
        output = PCA(n_components=1).fit_transform(self.dataset)
        np.testing.assert_allclose(np.abs(output.X[:, 0]), [2., 0., 2.], atol=1e-12)
        self.assertAlmostEqual(float(output.X.mean()), 0.)
        np.testing.assert_array_equal(output.y, self.dataset.y)
        self.assertEqual(output.features, ['PC1'])
        self.assertEqual(output.label, 'class')

    def test_components_are_rows_and_variance_is_a_fraction_of_total(self):
        model = PCA(n_components=1).fit(self.dataset)
        self.assertEqual(model.components.shape, (1, 2))
        np.testing.assert_allclose(model.get_covariance(), [[4., 0.], [0., 3.]])
        np.testing.assert_allclose(model.explained_variance, [4. / 7.])

    def test_invalid_component_count_is_rejected_before_fitting(self):
        for count in [-1, 0, 3]:
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    PCA(n_components=count).fit(self.dataset)
