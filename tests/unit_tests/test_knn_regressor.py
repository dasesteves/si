from unittest import TestCase
import numpy as np
from datasets import DATASETS_PATH
import os
from si.io.csv_file import read_csv
from si.models.knn_regressor import KNNRegressor
from si.model_selection.split import train_test_split

class TestKNNRegressor(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
        self.train_dataset, self.test_dataset = train_test_split(self.dataset, random_state=42)

    def test_fit(self):
        knn = KNNRegressor(k=3)
        knn.fit(self.dataset)
        
        self.assertTrue(np.all(self.dataset.features == knn.dataset.features))
        self.assertTrue(np.all(self.dataset.y == knn.dataset.y))

    def test_predict(self):
        knn = KNNRegressor(k=3)
        knn.fit(self.train_dataset)
        predictions = knn.predict(self.test_dataset)
        
        self.assertEqual(predictions.shape[0], self.test_dataset.y.shape[0])
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_score(self):
        knn = KNNRegressor(k=3)
        knn.fit(self.train_dataset)
        rmse_score = knn.score(self.test_dataset)
        
        self.assertIsInstance(rmse_score, float)
        self.assertGreater(rmse_score, 0)