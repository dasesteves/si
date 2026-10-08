from unittest import TestCase
import numpy as np
from synthetic_classification import training_data, evaluation_data, logistic_model


class TestLogisticRegressor(TestCase):
    def setUp(self):
        self.train_dataset, self.test_dataset = training_data(), evaluation_data()

    def test_fit(self):
        model = logistic_model().fit(self.train_dataset)
        self.assertTrue(model.is_fitted())
        self.assertEqual(model.theta.shape, (2,))
        self.assertGreater(len(model.cost_history), 0)
        self.assertTrue(np.isfinite(list(model.cost_history.values())).all())
        np.testing.assert_array_equal(model.mean, [0., 0.])
        np.testing.assert_array_equal(model.std, [2., 1.])

    def test_predict(self):
        model = logistic_model().fit(self.train_dataset)
        np.testing.assert_array_equal(model.predict(self.test_dataset), [0, 1, 0, 1])

    def test_score(self):
        self.assertEqual(logistic_model().fit(self.train_dataset).score(self.test_dataset), 1.)
