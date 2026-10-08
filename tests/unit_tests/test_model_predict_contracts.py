import unittest

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.models.knn_regressor import KNNRegressor
from si.models.lasso_regression import LassoRegression
from si.models.ridge_regression import RidgeRegression


class MeanModel(Model):
    def __init__(self, fail_fit=False):
        super().__init__()
        self.fail_fit = fail_fit
        self.prediction_calls = 0

    def _fit(self, dataset):
        self.mean = np.mean(dataset.y)
        if self.fail_fit:
            raise RuntimeError('Training failed')
        return self

    def _predict(self, dataset):
        self.prediction_calls += 1
        return np.full(len(dataset.X), self.mean)

    def _score(self, dataset, predictions):
        return np.mean((dataset.y - predictions) ** 2)


class TestModelPredictContracts(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset(
            X=np.array([[-2.], [-1.], [1.], [2.]]),
            y=np.array([-3., -1., 3., 5.]),
            features=['x'], label='y')

    def test_predict_rejects_unfitted_model_before_subclass_prediction(self):
        model = MeanModel()
        with self.assertRaisesRegex(ValueError, 'fitted'):
            model.predict(self.dataset)
        self.assertEqual(model.prediction_calls, 0)

    def test_failed_initial_fit_does_not_allow_prediction(self):
        model = MeanModel(fail_fit=True)
        with self.assertRaisesRegex(RuntimeError, 'Training failed'):
            model.fit(self.dataset)
        self.assertFalse(model.is_fitted())
        with self.assertRaisesRegex(ValueError, 'fitted'):
            model.predict(self.dataset)
        self.assertEqual(model.prediction_calls, 0)

    def test_fit_predict_sets_fitted_state_and_returns_predictions(self):
        model = MeanModel()
        np.testing.assert_array_equal(model.fit_predict(self.dataset), [1., 1., 1., 1.])
        self.assertTrue(model.is_fitted())
        self.assertEqual(model.prediction_calls, 1)

    def test_score_rejects_unfitted_model_and_works_after_fit(self):
        model = MeanModel()
        with self.assertRaises(ValueError):
            model.score(self.dataset)
        self.assertEqual(model.fit(self.dataset).score(self.dataset), 10.)

    def test_knn_predict_rejects_unfitted_model_and_works_after_fit(self):
        model = KNNRegressor(k=1)
        with self.assertRaisesRegex(ValueError, 'fitted'):
            model.predict(self.dataset)
        np.testing.assert_array_equal(model.fit_predict(self.dataset), self.dataset.y)

    def test_regression_models_reject_prediction_before_fit(self):
        for model in (RidgeRegression(), LassoRegression()):
            with self.subTest(model=type(model).__name__):
                with self.assertRaisesRegex(ValueError, 'fitted'):
                    model.predict(self.dataset)

    def assert_training_cost(self, model, penalty):
        states_during_cost = []
        original_cost = model.cost

        def record_cost(dataset):
            states_during_cost.append(model.is_fitted())
            return original_cost(dataset)

        model.cost = record_cost
        self.assertIs(model.fit(self.dataset), model)
        self.assertTrue(states_during_cost)
        self.assertFalse(any(states_during_cost))
        self.assertTrue(model.is_fitted())
        predictions = model.predict(self.dataset)
        expected = np.mean((self.dataset.y - predictions) ** 2) / 2 + penalty(model)
        self.assertTrue(np.isfinite(expected))
        self.assertAlmostEqual(original_cost(self.dataset), expected)
        self.assertAlmostEqual(model.cost_history[max(model.cost_history)], expected)

    def test_ridge_cost_can_be_evaluated_during_training(self):
        for scale in (False, True):
            with self.subTest(scale=scale):
                model = RidgeRegression(l2_penalty=0.2, alpha=0.1, max_iter=20, scale=scale)
                self.assert_training_cost(model, lambda m: m.l2_penalty * np.sum(m.theta ** 2) / (2 * len(self.dataset.y)))

    def test_lasso_cost_can_be_evaluated_during_training(self):
        for scale in (False, True):
            with self.subTest(scale=scale):
                model = LassoRegression(l1_penalty=0.2, max_iter=20, scale=scale)
                self.assert_training_cost(model, lambda m: m.l1_penalty * np.sum(np.abs(m.theta)))
