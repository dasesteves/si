from unittest import TestCase

import numpy as np
import pandas as pd

from si.data.dataset import Dataset


class TestDatasetDataframeLabels(TestCase):
    def test_label_column_is_not_counted_as_a_feature(self):
        frame = pd.DataFrame({'first': [1., 2.], 'class': [0, 1], 'last': [3., 4.]})
        before = frame.copy(deep=True)
        dataset = Dataset.from_dataframe(frame, label='class')
        self.assertEqual(dataset.features, ['first', 'last'])
        self.assertEqual(dataset.label, 'class')
        np.testing.assert_array_equal(dataset.X, [[1., 3.], [2., 4.]])
        np.testing.assert_array_equal(dataset.y, [0, 1])
        pd.testing.assert_frame_equal(frame, before)

    def test_unlabelled_frame_preserves_all_columns(self):
        frame = pd.DataFrame({'first': [1., 2.], 'last': [3., 4.]})
        dataset = Dataset.from_dataframe(frame)
        self.assertEqual(dataset.features, ['first', 'last'])
        self.assertIsNone(dataset.y)
        self.assertIsNone(dataset.label)
        np.testing.assert_array_equal(dataset.X, [[1., 3.], [2., 4.]])
