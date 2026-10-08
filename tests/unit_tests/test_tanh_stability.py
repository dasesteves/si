from unittest import TestCase
import numpy as np
from si.neural_networks.activation import TanhActivation


class TestTanhStability(TestCase):
    def test_large_inputs_saturate_without_overflow_or_nan(self):
        layer = TanhActivation()
        values = np.array([[-1000., 0., 1000.]])
        try:
            with np.errstate(over='raise', invalid='raise'):
                output = layer.activation_function(values)
                derivative = layer.derivative(values)
        except FloatingPointError as exc:
            self.fail('Finite inputs must give finite Tanh outputs: ' + str(exc))
        np.testing.assert_array_equal(output, [[-1., 0., 1.]])
        np.testing.assert_array_equal(derivative, [[0., 1., 0.]])

    def test_moderate_inputs_and_backward_preserve_expected_values(self):
        layer = TanhActivation()
        values = np.array([[-np.log(3.) / 2, 0., np.log(3.) / 2]])
        np.testing.assert_allclose(layer.forward_propagation(values, True), [[-.5, 0., .5]])
        np.testing.assert_allclose(layer.backward_propagation(np.array([[2., 3., 4.]])), [[1.5, 3., 3.]])
