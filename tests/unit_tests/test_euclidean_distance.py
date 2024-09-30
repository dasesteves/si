from unittest import TestCase
from datasets import DATASETS_PATH
from si.io.csv_file import read_csv

import os
import numpy as np
from src.si.statistics.euclidean_distance import euclidean_distance

class TestEuclideanDistance(TestCase):
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
        
    def test_euclidean_distance(self):
        x = np.array([1, 2, 3])
        y = np.array([[4, 5, 6], [7, 8, 9]])
        expected_output = np.array([np.sqrt(27), np.sqrt(54)])
        self.assertTrue(np.allclose(euclidean_distance(x, y), expected_output))

