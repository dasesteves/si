from unittest import TestCase
import numpy as np

from si.neural_networks.layers import DenseLayer
from si.neural_networks.optimizers import SGD


class TestDenseLayer(TestCase):
    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.X = np.array([[1., 2.], [3., 4.]])
        self.layer = DenseLayer(n_units=2, input_shape=(2,)).initialize(SGD(learning_rate=.1))
        self.layer.weights = np.array([[1., -1.], [2., .5]])
        self.layer.biases = np.array([[.5, -.5]])

    def test_forward_propagation(self):
        output = self.layer.forward_propagation(self.X, training=False)
        np.testing.assert_allclose(output, [[5.5, -.5], [11.5, -1.5]])

    def test_backward_propagation(self):
        self.layer.forward_propagation(self.X, training=True)
        error = self.layer.backward_propagation(np.array([[1., -1.], [2., 3.]]))
        np.testing.assert_allclose(error, [[2., 1.5], [-1., 5.5]])
        np.testing.assert_allclose(self.layer.weights, [[.3, -1.8], [1., -.5]])
        np.testing.assert_allclose(self.layer.biases, [[.2, -.7]])
