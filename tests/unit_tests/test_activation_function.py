from unittest import TestCase
import numpy as np

from si.neural_networks.activation import ReLUActivation, SigmoidActivation


class TestSigmoidLayer(TestCase):
    def setUp(self):
        self.X = np.array([[-np.log(3), 0., np.log(3)], [np.log(3), 0., -np.log(3)]])

    def test_activation_function(self):
        np.testing.assert_allclose(SigmoidActivation().activation_function(self.X),
                                   [[.25, .5, .75], [.75, .5, .25]])

    def test_derivative(self):
        np.testing.assert_allclose(SigmoidActivation().derivative(self.X),
                                   [[.1875, .25, .1875], [.1875, .25, .1875]])


class TestRELULayer(TestCase):
    def test_activation_function(self):
        X = np.array([[-2., 0., 3.], [4., -5., -1.]])
        np.testing.assert_array_equal(ReLUActivation().activation_function(X),
                                      [[0., 0., 3.], [4., 0., 0.]])

    def test_derivative(self):
        # Avoid zero, where ReLU has no unique mathematical derivative.
        X = np.array([[-2., -1., 3.], [4., -5., -1.]])
        np.testing.assert_array_equal(ReLUActivation().derivative(X),
                                      [[0., 0., 1.], [1., 0., 0.]])
