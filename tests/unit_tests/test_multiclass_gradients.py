from unittest import TestCase

import numpy as np

from si.data.dataset import Dataset
from si.neural_networks.activation import SoftmaxActivation
from si.neural_networks.layers import DenseLayer
from si.neural_networks.losses import CategoricalCrossEntropy
from si.neural_networks.neural_network import NeuralNetwork


def finite_difference(function, values):
    gradient = np.empty_like(values)
    step = 1e-6
    for index in np.ndindex(values.shape):
        plus, minus = values.copy(), values.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (function(plus) - function(minus)) / (2 * step)
    return gradient


class TestMulticlassGradients(TestCase):
    def setUp(self):
        self.labels = np.array([[1., 0., 0.], [0., 1., 0.]])
        self.probabilities = np.array([[.2, .3, .5], [.3, .6, .1]])
        self.logits = np.array([[.3, -.2, .7], [1.1, .4, -.1]])

    def test_categorical_derivative_matches_mean_loss(self):
        loss = CategoricalCrossEntropy()
        numerical = finite_difference(lambda p: loss.loss(self.labels, p), self.probabilities)
        np.testing.assert_allclose(loss.derivative(self.labels, self.probabilities),
                                   numerical, rtol=1e-6, atol=1e-8)

    def test_duplicate_batch_halves_each_example_gradient(self):
        loss = CategoricalCrossEntropy()
        gradient = loss.derivative(self.labels, self.probabilities)
        duplicated = loss.derivative(np.tile(self.labels, (2, 1)),
                                     np.tile(self.probabilities, (2, 1)))
        np.testing.assert_allclose(duplicated, np.tile(gradient / 2, (2, 1)))

    def test_softmax_derivative_includes_cross_class_terms(self):
        layer = SoftmaxActivation()
        jacobian = layer.derivative(self.logits)
        self.assertEqual(jacobian.shape, (2, 3, 3))
        for batch in range(2):
            for output in range(3):
                numerical = finite_difference(
                    lambda row: layer.activation_function(row.reshape(1, -1))[0, output],
                    self.logits[batch])
                np.testing.assert_allclose(jacobian[batch, output], numerical, rtol=1e-6, atol=1e-8)

    def test_softmax_backward_matches_weighted_output_derivative(self):
        layer = SoftmaxActivation()
        weights = np.array([[.4, -.5, 1.2], [-.3, .8, .5]])
        numerical = finite_difference(lambda x: np.sum(layer.activation_function(x) * weights), self.logits)
        layer.forward_propagation(self.logits, training=True)
        np.testing.assert_allclose(layer.backward_propagation(weights), numerical, rtol=1e-6, atol=1e-8)

    def test_sum_of_softmax_outputs_has_zero_derivative(self):
        layer = SoftmaxActivation()
        layer.forward_propagation(self.logits, training=True)
        np.testing.assert_allclose(layer.backward_propagation(np.ones_like(self.logits)), 0., atol=1e-14)

    def test_softmax_and_loss_match_combined_numerical_derivative(self):
        layer, loss = SoftmaxActivation(), CategoricalCrossEntropy()
        numerical = finite_difference(lambda x: loss.loss(self.labels, layer.activation_function(x)), self.logits)
        predictions = layer.forward_propagation(self.logits, training=True)
        actual = layer.backward_propagation(loss.derivative(self.labels, predictions))
        np.testing.assert_allclose(actual, numerical, rtol=1e-6, atol=1e-8)

    def test_one_training_step_follows_full_network_loss_gradient(self):
        state = np.random.get_state()
        self.addCleanup(np.random.set_state, state)
        np.random.seed(0)
        network = NeuralNetwork(epochs=1, batch_size=2, learning_rate=.1,
                                loss=CategoricalCrossEntropy, metric=None)
        dense = DenseLayer(3, input_shape=(2,))
        network.add(dense).add(SoftmaxActivation())
        X = np.array([[1., -.5], [-.3, .8]])
        weights = np.array([[.1, -.2, .3], [.4, .2, -.1]])
        biases = np.array([[.1, 0., -.1]])
        dense.weights, dense.biases = weights.copy(), biases.copy()
        loss, activation = CategoricalCrossEntropy(), SoftmaxActivation()
        weight_gradient = finite_difference(
            lambda w: loss.loss(self.labels, activation.activation_function(X @ w + biases)), weights)
        bias_gradient = finite_difference(
            lambda b: loss.loss(self.labels, activation.activation_function(X @ weights + b)), biases)
        network.fit(Dataset(X, self.labels))
        np.testing.assert_allclose(dense.weights, weights - .1 * weight_gradient, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(dense.biases, biases - .1 * bias_gradient, rtol=1e-6, atol=1e-8)
        predicted = network.predict(Dataset(X))
        np.testing.assert_allclose(predicted.sum(axis=1), 1.)
