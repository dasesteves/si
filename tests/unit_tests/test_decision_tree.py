from unittest import TestCase
import numpy as np
from si.models.decision_tree_classifier import DecisionTreeClassifier
from synthetic_classification import training_data, evaluation_data


class TestDecisionTree(TestCase):
    def setUp(self):
        self.train_dataset, self.test_dataset = training_data(), evaluation_data()

    def test_fit(self):
        model = DecisionTreeClassifier().fit(self.train_dataset)
        self.assertTrue(model.is_fitted())
        np.testing.assert_array_equal(model.predict(self.train_dataset), self.train_dataset.y)

    def test_predict(self):
        model = DecisionTreeClassifier().fit(self.train_dataset)
        np.testing.assert_array_equal(model.predict(self.test_dataset), [0, 1, 0, 1])

    def test_score(self):
        self.assertEqual(DecisionTreeClassifier().fit(self.train_dataset).score(self.test_dataset), 1.)
