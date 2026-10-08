from unittest import TestCase
import numpy as np

from si.statistics.euclidean_distance import euclidean_distance


class TestEuclideanDistance(TestCase):
    def test_euclidean_distance(self):
        x = np.array([1, 2, 3])
        y = np.array([[1, 2, 3], [4, 5, 6]])
        distances = euclidean_distance(x, y)
        self.assertEqual(distances.shape, (2,))
        # Squared coordinate differences sum to 0 and 27.
        np.testing.assert_allclose(distances, [0., 5.196152422706632])

    def test_negative_coordinates_and_three_four_five_distance(self):
        x = np.array([-1., -2.])
        y = np.array([[2., 2.], [-4., -6.], [-1., -2.]])
        np.testing.assert_allclose(euclidean_distance(x, y), [5., 5., 0.])

    def test_mismatched_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            euclidean_distance(np.array([1., 2.]), np.array([[1., 2., 3.]]))
