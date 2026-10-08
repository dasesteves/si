from unittest import TestCase

import numpy as np

from si.data.dataset import Dataset
from si.feature_selection.select_percentile import SelectPercentile


class TestSelectPercentileDefault(TestCase):
    def setUp(self):
        self.dataset = Dataset(
            X=np.array([[0., 3., 1., 2.], [1., 1., 2., 1.], [2., 2., 1., 3.],
                        [6., 3., 2., 4.], [7., 1., 1., 2.], [8., 2., 2., 3.]]),
            y=np.array([0, 0, 0, 1, 1, 1]),
            features=['signal', 'noise', 'small', 'medium'],
            label='class',
        )

    def test_default_scorer_fits_and_preserves_selected_data(self):
        selector = SelectPercentile(percentile=50)
        output = selector.fit_transform(self.dataset)
        np.testing.assert_allclose(selector.F, [54., 0., .5, 1.5])
        self.assertTrue(np.all(np.isfinite(selector.p)))
        self.assertTrue(np.all((selector.p >= 0) & (selector.p <= 1)))
        np.testing.assert_array_equal(output.X, self.dataset.X[:, [0, 3]])
        np.testing.assert_array_equal(output.y, self.dataset.y)
        self.assertEqual(output.features, ['signal', 'medium'])
        self.assertEqual(output.label, self.dataset.label)

    def test_default_scorer_respects_zero_and_all_features(self):
        for percentile, count in ((0, 0), (100, 4)):
            with self.subTest(percentile=percentile):
                output = SelectPercentile(percentile=percentile).fit_transform(self.dataset)
                self.assertEqual(output.X.shape, (6, count))
                self.assertEqual(len(output.features), count)
                np.testing.assert_array_equal(output.y, self.dataset.y)

    def test_explicit_scorer_keeps_its_behavior(self):
        def rank_second_and_third(dataset):
            self.assertIs(dataset, self.dataset)
            return np.array([1., 4., 3., 2.]), np.array([.9, .1, .2, .5])

        output = SelectPercentile(percentile=50, score_func=rank_second_and_third).fit_transform(self.dataset)
        np.testing.assert_array_equal(output.X, self.dataset.X[:, [1, 2]])
        self.assertEqual(output.features, ['noise', 'small'])
