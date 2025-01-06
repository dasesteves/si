from unittest import TestCase

from datasets import DATASETS_PATH

import os

from si.ensemble.stacking_classifier import StackingClassifier
from si.io.data_file import read_data_file
from si.metrics.accuracy import accuracy
from si.model_selection.split import train_test_split
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression

class TestStackingClassifier(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

        decision_tree = DecisionTreeClassifier()
        knn = KNNClassifier()
        logistic_regression = LogisticRegression()
        final_model = KNNClassifier()

        self.stacking = StackingClassifier(models=[decision_tree, knn, logistic_regression], final_model=final_model)

    def test_fit(self):

        self.stacking.fit(self.train_dataset)

        self.assertEqual(self.stacking.predictions_dataset.X.shape[0], self.train_dataset.X.shape[0])
        self.assertEqual(self.stacking.predictions_dataset.X.shape[1], len(self.stacking.models))


    def test_predict(self):
        self.stacking.fit(self.train_dataset)
        
        predictions = self.stacking.predict(self.test_dataset)

        self.assertEqual(predictions.shape[0], self.test_dataset.X.shape[0])
    
    def test_score(self):
        self.stacking.fit(self.train_dataset)
        
        accuracy_ = self.stacking.score(self.test_dataset)
        
        expected_accuracy = accuracy(self.test_dataset.y, self.stacking.predict(self.test_dataset))

        self.assertEqual(round(accuracy_, 2), round(expected_accuracy, 2))