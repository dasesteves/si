from unittest import TestCase
import numpy as np
from si.data.dataset import Dataset
from si.models.logistic_regression import LogisticRegression


class TestLogisticRefitAndConstants(TestCase):
    def test_constant_feature_keeps_fit_cost_and_predictions_finite(self):
        data = Dataset(np.array([[-2., 7.], [-1., 7.], [1., 7.], [2., 7.]]), np.array([0, 0, 1, 1]))
        model = LogisticRegression(alpha=.1, max_iter=20)
        try:
            with np.errstate(divide='raise', invalid='raise'):
                model.fit(data)
                predicted = model.predict(data)
                cost = model.cost(data)
        except FloatingPointError as exc:
            self.fail('Constant features must not divide by zero: ' + str(exc))
        np.testing.assert_array_equal(predicted, data.y)
        self.assertTrue(np.isfinite(cost))
        self.assertEqual(model.theta[1], 0.)

    def test_refit_replaces_history_and_matches_a_fresh_model(self):
        first = Dataset(np.array([[-3.], [-1.], [1.], [3.]]), np.array([0, 0, 1, 1]))
        second = Dataset(np.array([[10.], [12.], [14.], [16.]]), np.array([1, 1, 0, 0]))
        model = LogisticRegression(alpha=.1, max_iter=8, patience=20).fit(first)
        model.max_iter = 2
        model.fit(second)
        fresh = LogisticRegression(alpha=.1, max_iter=2, patience=20).fit(second)
        self.assertEqual(set(model.cost_history), {0, 1})
        self.assertEqual(model.cost_history, fresh.cost_history)
        np.testing.assert_allclose(model.theta, fresh.theta)
        np.testing.assert_array_equal(model.predict(second), fresh.predict(second))
