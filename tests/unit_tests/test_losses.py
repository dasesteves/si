from unittest import TestCase
import numpy as np

from si.neural_networks.losses import BinaryCrossEntropy, MeanSquaredError


class TestLosses(TestCase):
    def test_mean_squared_error_loss(self):
        y_true, y_pred = np.array([1., 3.]), np.array([2., 5.])
        self.assertEqual(MeanSquaredError().loss(y_true, y_pred), 2.5)
        self.assertEqual(MeanSquaredError().loss(y_true, y_true), 0.)

    def test_mean_squared_error_derivative(self):
        # For two outputs, d(mean squared error)/dp is [1, 2].
        np.testing.assert_allclose(MeanSquaredError().derivative(np.array([1., 3.]), np.array([2., 5.])),
                                   [1., 2.])

    def test_binary_cross_entropy_loss(self):
        y_true, y_pred = np.array([1., 0.]), np.array([.25, .75])
        # This class uses summed binary cross entropy: -2 * log(0.25).
        self.assertAlmostEqual(BinaryCrossEntropy().loss(y_true, y_pred), 2.772588722239781)
        self.assertAlmostEqual(BinaryCrossEntropy().loss(y_true, y_true), 0.)

    def test_binary_cross_entropy_derivative(self):
        np.testing.assert_allclose(BinaryCrossEntropy().derivative(np.array([1., 0.]), np.array([.25, .75])),
                                   [-4., 4.])
