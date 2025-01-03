from unittest import TestCase
import numpy as np
from datasets import DATASETS_PATH
import os
from si.io.csv_file import read_csv
from si.statistics.cosine_distance import cosine_distance

class TestCosineDistance(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_cosine_distance(self):
        # Test orthogonal vectors (should give distance = 1)
        x = np.array([1, 0])
        y = np.array([[0, 1], [1, 0]])
        distances = cosine_distance(x, y)
        
        expected = np.array([1.0, 0.0])  # orthogonal: 1, same direction: 0
        self.assertTrue(np.allclose(distances, expected))
