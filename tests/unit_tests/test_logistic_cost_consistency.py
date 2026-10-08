from unittest import TestCase

import numpy as np

from si.data.dataset import Dataset
from si.models.logistic_regression import LogisticRegression


class TestLogisticCostConsistency(TestCase):
    def setUp(self):
        self.data = Dataset(np.array([[10., 2.], [12., 4.], [14., 3.], [16., 8.]]),
                            np.array([0, 0, 1, 1]))

    def test_training_cost_uses_standardized_features(self):
        model = LogisticRegression(alpha=.1, max_iter=1, l2_penalty=.2).fit(self.data)
        # Independent one-step calculation, including L2 / (2 * sample count).
        self.assertAlmostEqual(model.cost(self.data), .6662858014409409)
        self.assertAlmostEqual(model.cost_history[0], .6662858014409409)

    def test_evaluation_cost_reuses_training_statistics(self):
        model = LogisticRegression(alpha=.1, max_iter=1, l2_penalty=.2).fit(self.data)
        evaluation = Dataset(np.array([[18., 1.], [20., 9.]]), np.array([0, 1]))
        standardized = (evaluation.X - np.array([13., 4.25])) / np.sqrt([5., 5.1875])
        scores = standardized @ model.theta + model.theta_zero
        probabilities = 1 / (1 + np.exp(-scores))
        expected = -np.mean(evaluation.y * np.log(probabilities)
                            + (1 - evaluation.y) * np.log(1 - probabilities))
        expected += .2 * np.sum(model.theta ** 2) / 4
        self.assertAlmostEqual(model.cost(evaluation), expected)

    def test_changing_feature_units_preserves_training(self):
        translated = Dataset(self.data.X * [2., 3.] + [50., -20.], self.data.y.copy())
        first = LogisticRegression(alpha=.1, max_iter=12, patience=2).fit(self.data)
        second = LogisticRegression(alpha=.1, max_iter=12, patience=2).fit(translated)
        self.assertEqual(len(first.cost_history), len(second.cost_history))
        np.testing.assert_allclose(list(first.cost_history.values()), list(second.cost_history.values()))
        np.testing.assert_allclose(first.theta, second.theta)

    def test_unscaled_cost_keeps_original_feature_units(self):
        model = LogisticRegression(alpha=.001, max_iter=1, l2_penalty=.2, scale=False).fit(self.data)
        scores = self.data.X @ model.theta + model.theta_zero
        probabilities = 1 / (1 + np.exp(-scores))
        expected = -np.mean(self.data.y * np.log(probabilities)
                            + (1 - self.data.y) * np.log(1 - probabilities))
        expected += .2 * np.sum(model.theta ** 2) / 8
        self.assertAlmostEqual(model.cost(self.data), expected)

    def test_saturated_probabilities_have_finite_cost(self):
        training = Dataset(np.array([[-1.], [1.]]), np.array([0, 1]))
        model = LogisticRegression(alpha=.1, max_iter=1, l2_penalty=0., scale=False).fit(training)
        model.theta = np.array([1.])
        model.theta_zero = 0.
        for labels, expected in [(np.array([0, 1]), 0.), (np.array([1, 0]), 1000.)]:
            with self.subTest(labels=labels.tolist()):
                evaluation = Dataset(np.array([[-1000.], [1000.]]), labels)
                try:
                    with np.errstate(over='raise', divide='raise', invalid='raise'):
                        actual = model.cost(evaluation)
                except FloatingPointError as exc:
                    self.fail('Cost must remain finite for finite logits: ' + str(exc))
                self.assertAlmostEqual(actual, expected)
