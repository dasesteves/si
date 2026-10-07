from unittest import TestCase

import numpy as np

from si.data.dataset import Dataset
from si.models.lasso_regression import LassoRegression


class TestLassoContracts(TestCase):
    def test_penalty_matches_mean_squared_cost(self):
        # For mean(x)=0, mean(x*x)=1 and y=3*x+5, alpha=.5 gives w=2.5, b=5.
        dataset = Dataset(X=np.array([[-1.], [1.], [-1.], [1.]]),
                          y=np.array([2., 8., 2., 8.]))
        model = LassoRegression(l1_penalty=.5, max_iter=100, scale=False).fit(dataset)
        np.testing.assert_allclose(model.theta, [2.5], atol=1e-8)
        self.assertAlmostEqual(float(model.theta_zero), 5.)
        self.assertAlmostEqual(float(model.cost(dataset)), 1.375)

    def test_correlated_features_converge_without_increasing_objective(self):
        x = np.array([-3., -2., -1., 1., 2., 3.])
        dataset = Dataset(X=np.column_stack([x, x, x]), y=2. * x + 4.)
        model = LassoRegression(l1_penalty=0., max_iter=50, scale=False).fit(dataset)
        np.testing.assert_allclose(model.predict(dataset), dataset.y, atol=1e-8)
        costs = np.array(list(model.cost_history.values()))
        self.assertTrue(np.all(np.diff(costs) <= 1e-9))

    def test_constant_feature_is_finite_and_does_not_change_input(self):
        dataset = Dataset(X=np.array([[7., -1.], [7., 1.], [7., -1.], [7., 1.]]),
                          y=np.array([2., 8., 2., 8.]))
        original = dataset.X.copy()
        model = LassoRegression(l1_penalty=.5, max_iter=100, scale=True).fit(dataset)
        np.testing.assert_allclose(model.theta, [0., 2.5], atol=1e-8)
        self.assertTrue(np.isfinite(model.predict(dataset)).all())
        np.testing.assert_array_equal(dataset.X, original)
