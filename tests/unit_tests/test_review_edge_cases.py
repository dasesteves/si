import unittest
import warnings
import numpy as np

from si.data.dataset import Dataset
from si.decomposition.pca import PCA
from si.feature_selection.select_percentile import SelectPercentile
from si.models.lasso_regression import LassoRegression


class TestReviewEdgeCases(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset(np.array([[0., 1., 4., 2.], [1., 1., 3., 1.],
                                         [4., 1., 2., 1.], [5., 1., 1., 2.]]),
                               y=np.array([0, 0, 1, 1]), features=['signal', 'constant', 'other', 'noise'], label='class')

    def select_scores(self, scores, percentile):
        selector = SelectPercentile(percentile, score_func=lambda _: (np.array(scores), np.zeros(4)))
        return selector.fit_transform(self.dataset)

    def test_nan_score_does_not_discard_informative_variables(self):
        result = self.select_scores([9., np.nan, 5., 1.], 50)
        self.assertEqual(result.features, ['signal', 'other'])
        np.testing.assert_array_equal(result.X, self.dataset.X[:, [0, 2]])
        self.assertEqual(result.label, 'class')

    def test_default_scorer_with_constant_variable(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = SelectPercentile(50).fit_transform(self.dataset)
        self.assertEqual(result.features, ['signal', 'other'])

    def test_infinite_scores_remain_rankable(self):
        self.assertEqual(self.select_scores([np.inf, np.nan, 5., -np.inf], 50).features, ['signal', 'other'])

    def test_ties_select_earlier_features_deterministically(self):
        self.assertEqual(self.select_scores([4., 4., 4., 4.], 50).features, ['signal', 'constant'])

    def test_budget_rounds_down_and_does_not_select_nan(self):
        self.assertEqual(self.select_scores([4., 3., 2., np.nan], 10).shape()[1], 0)
        self.assertEqual(self.select_scores([4., np.nan, 2., 1.], 100).features, ['signal', 'other', 'noise'])

    def test_no_usable_scores_returns_empty_selection(self):
        self.assertEqual(self.select_scores([np.nan]*4, 100).shape()[1], 0)

    def test_single_feature_pca_preserves_centered_variation(self):
        dataset = Dataset(np.array([[1.], [2.], [3.]]), features=['value'])
        pca = PCA(1).fit(dataset)
        result = pca.transform(dataset)
        self.assertEqual(pca.get_covariance().shape, (1, 1))
        np.testing.assert_allclose(pca.explained_variance, [1.])
        np.testing.assert_allclose(np.abs(result.X[:, 0]), [1., 0., 1.])
        self.assertEqual(result.features, ['PC1'])

    def test_pca_rejects_non_integer_component_counts_with_value_error(self):
        for value in [1.5, 1.0, True, np.bool_(True), '1', None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                PCA(value).fit(self.dataset)

    def test_pca_accepts_numpy_integer_component_count(self):
        self.assertEqual(PCA(np.int64(1)).fit_transform(self.dataset).shape()[1], 1)

    def test_lasso_refit_keeps_only_the_current_run_history(self):
        model = LassoRegression(l1_penalty=.1, max_iter=8, patience=20, scale=False)
        model.fit(self.dataset)
        self.assertEqual(len(model.cost_history), 8)
        model.max_iter = 2
        second = Dataset(self.dataset.X.copy(), y=np.array([4., 4., 5., 5.]))
        model.fit(second)
        self.assertEqual(set(model.cost_history), {0, 1})
        fresh = LassoRegression(l1_penalty=.1, max_iter=2, patience=20, scale=False).fit(second)
        self.assertEqual(model.cost_history, fresh.cost_history)
        np.testing.assert_allclose(model.predict(second), fresh.predict(second))


if __name__ == '__main__':
    unittest.main()
