from unittest import TestCase

from datasets import DATASETS_PATH

import os
import numpy as np

from si.io.csv_file import read_csv

from si.model_selection.split import train_test_split, stratified_train_test_split

class TestSplits(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_train_test_split(self):

        train, test = train_test_split(self.dataset, test_size = 0.2, random_state=123)
        test_samples_size = int(self.dataset.shape()[0] * 0.2)
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)

    def test_stratified_split(self):
        test_size = 0.2
        train, test = stratified_train_test_split(self.dataset, test_size=test_size, random_state=42)
        
        # Verificar tamanhos dos conjuntos
        test_samples_size = int(self.dataset.shape()[0] * test_size)
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)
        
        # Verificar proporções das classes
        original_props = np.unique(self.dataset.y, return_counts=True)[1] / len(self.dataset.y)
        train_props = np.unique(train.y, return_counts=True)[1] / len(train.y)
        test_props = np.unique(test.y, return_counts=True)[1] / len(test.y)
        
        # Verificar se as proporções são mantidas (com tolerância)
        self.assertTrue(np.allclose(original_props, train_props, atol=0.1))
        self.assertTrue(np.allclose(original_props, test_props, atol=0.1))