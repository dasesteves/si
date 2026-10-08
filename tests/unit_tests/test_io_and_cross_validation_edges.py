from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import numpy as np

from si.data.dataset import Dataset
from si.io.data_file import read_data_file
from si.metrics.accuracy import accuracy
from si.model_selection.cross_validate import k_fold_cross_validation
from si.models.knn_classifier import KNNClassifier


class TestDataFileDimensions(TestCase):
    def read_text(self, text, label=False):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / 'synthetic.data'
        path.write_text(text, encoding='utf-8')
        return read_data_file(path, sep=',', label=label)

    def test_single_row_preserves_feature_axis_and_label(self):
        dataset = self.read_text('1,2,0\n', label=True)
        np.testing.assert_array_equal(dataset.X, [[1., 2.]])
        np.testing.assert_array_equal(dataset.y, [0.])

    def test_single_column_remains_a_column(self):
        dataset = self.read_text('1\n2\n3\n')
        np.testing.assert_array_equal(dataset.X, [[1.], [2.], [3.]])
        self.assertFalse(dataset.has_label())

    def test_single_value_is_a_one_by_one_dataset(self):
        dataset = self.read_text('7\n')
        self.assertEqual(dataset.shape(), (1, 1))
        np.testing.assert_array_equal(dataset.X, [[7.]])


class TestCrossValidationPartition(TestCase):
    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        self.dataset = Dataset(np.arange(10, dtype=float).reshape(-1, 1),
                               y=np.arange(10), features=['position'], label='class')

    def test_every_sample_is_tested_once_in_balanced_folds(self):
        tested, sizes = [], []

        def record_score(y_true, predictions):
            tested.extend(y_true.tolist())
            sizes.append(len(y_true))
            return accuracy(y_true, predictions)

        scores = k_fold_cross_validation(KNNClassifier(k=1), self.dataset,
                                        scoring=record_score, cv=3, seed=0)
        self.assertEqual(len(scores), 3)
        self.assertCountEqual(tested, list(range(10)))
        self.assertEqual(sorted(sizes), [3, 3, 4])

    def test_invalid_fold_counts_raise_value_error(self):
        for folds in (0, 1, -1, 11, 2.5, True, None):
            with self.subTest(folds=folds), self.assertRaises(ValueError):
                k_fold_cross_validation(KNNClassifier(k=1), self.dataset, cv=folds, seed=0)

    def test_fold_datasets_keep_feature_and_label_names(self):
        observations = []

        class RecordingKNN(KNNClassifier):
            def _fit(self, dataset):
                observations.append((dataset.features, dataset.label))
                return super()._fit(dataset)

            def _predict(self, dataset):
                observations.append((dataset.features, dataset.label))
                return super()._predict(dataset)

        k_fold_cross_validation(RecordingKNN(k=1), self.dataset, cv=3, seed=0)
        self.assertEqual(len(observations), 6)
        self.assertTrue(all(features == ['position'] and label == 'class' for features, label in observations))
