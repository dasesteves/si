from unittest import TestCase
import numpy as np
from datasets import DATASETS_PATH
import os
from si.io.csv_file import read_csv
from si.decomposition.pca import PCA

class TestPCA(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        n_components = 2
        pca = PCA(n_components=n_components)
        pca.fit(self.dataset)
        
        # Check if components were correctly created
        self.assertEqual(pca.components.shape[1], n_components)
        self.assertEqual(pca.components.shape[0], self.dataset.X.shape[1])
        
        # Check if explained variance was calculated
        self.assertEqual(len(pca.explained_variance), n_components)
        self.assertTrue(np.all(pca.explained_variance >= 0))
        self.assertTrue(np.allclose(np.sum(pca.explained_variance), 1))

    def test_transform(self):
        n_components = 2
        pca = PCA(n_components=n_components)
        
        # Test fit_transform
        transformed_dataset = pca.fit_transform(self.dataset)
        
        # Check transformed shape
        self.assertEqual(transformed_dataset.X.shape[0], self.dataset.X.shape[0])
        self.assertEqual(transformed_dataset.X.shape[1], n_components)
        
        # Check if features were renamed
        self.assertEqual(len(transformed_dataset.features), n_components)
        self.assertEqual(transformed_dataset.features[0], 'PC1')