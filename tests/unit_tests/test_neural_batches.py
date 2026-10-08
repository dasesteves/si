from unittest import TestCase
import numpy as np
from si.data.dataset import Dataset
from si.neural_networks.layers import DenseLayer
from si.neural_networks.neural_network import NeuralNetwork


class TestNeuralBatches(TestCase):
    def test_includes_last_partial_batch_with_matching_labels(self):
        X = np.arange(10.).reshape(5, 2)
        y = np.arange(5)
        batches = list(NeuralNetwork(batch_size=2)._get_mini_batches(X, y, shuffle=False))
        self.assertEqual([len(x) for x, _ in batches], [2, 2, 1])
        np.testing.assert_array_equal(np.concatenate([x for x, _ in batches]), X)
        np.testing.assert_array_equal(np.concatenate([target for _, target in batches]), y)

    def test_batch_larger_than_dataset_uses_all_available_samples(self):
        X = np.arange(6.).reshape(3, 2)
        try:
            batches = list(NeuralNetwork(batch_size=128)._get_mini_batches(X, shuffle=False))
        except AssertionError as exc:
            self.fail('Default batch size must support a small dataset: ' + str(exc))
        self.assertEqual(len(batches), 1)
        np.testing.assert_array_equal(batches[0][0], X)
        self.assertIsNone(batches[0][1])

    def test_rejects_nonpositive_noninteger_and_boolean_batch_sizes(self):
        for value in (0, -1, 1.5, True, np.bool_(False), None):
            with self.subTest(batch_size=value):
                with self.assertRaises(ValueError):
                    list(NeuralNetwork(batch_size=value)._get_mini_batches(np.ones((3, 1))))

    def test_accepts_numpy_integer_batch_size(self):
        batches = list(NeuralNetwork(batch_size=np.int64(2))._get_mini_batches(np.ones((3, 1)), shuffle=False))
        self.assertEqual([len(x) for x, _ in batches], [2, 1])

    def test_rejects_empty_training_data(self):
        with self.assertRaisesRegex(ValueError, 'sample'):
            NeuralNetwork(epochs=1, batch_size=2).fit(Dataset(np.empty((0, 1)), np.empty(0)))

    def test_training_history_accounts_for_each_sample(self):
        state = np.random.get_state()
        self.addCleanup(np.random.set_state, state)
        np.random.seed(0)
        net = NeuralNetwork(epochs=1, batch_size=2, learning_rate=0., metric=None)
        layer = DenseLayer(1, input_shape=(1,))
        net.add(layer)
        layer.weights[:] = 0.
        layer.biases[:] = 0.
        net.fit(Dataset(np.zeros((5, 1)), np.array([1., 2., 3., 4., 5.])))
        # Mean of squares over ALL five targets is 11, regardless of shuffle.
        self.assertAlmostEqual(net.history[1]['loss'], 11.)

    def test_complete_batches_keep_order_without_shuffle(self):
        X = np.arange(8.).reshape(4, 2)
        batches = list(NeuralNetwork(batch_size=2)._get_mini_batches(X, shuffle=False))
        self.assertEqual([len(x) for x, _ in batches], [2, 2])
        np.testing.assert_array_equal(np.concatenate([x for x, _ in batches]), X)

    def test_predict_before_fit_raises_the_model_contract_error(self):
        with self.assertRaisesRegex(ValueError, 'fitted'):
            NeuralNetwork().predict(Dataset(np.ones((2, 1))))
