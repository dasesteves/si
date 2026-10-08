from unittest import TestCase
import numpy as np
from si.ensemble.stacking_classifier import StackingClassifier
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.models.knn_classifier import KNNClassifier
from synthetic_classification import training_data, evaluation_data, logistic_model


class TestStackingClassifier(TestCase):
    def setUp(self):
        self.train_dataset, self.test_dataset = training_data(), evaluation_data()
        self.stacking = StackingClassifier([DecisionTreeClassifier(), KNNClassifier(k=3), logistic_model()],
                                           final_model=KNNClassifier(k=3))

    def test_fit(self):
        self.stacking.fit(self.train_dataset)
        self.assertTrue(self.stacking.is_fitted())
        self.assertEqual(self.stacking.new_dataset.shape(), (12, 3))
        np.testing.assert_array_equal(self.stacking.new_dataset.y, [0, 0, 1, 1] * 3)
        self.assertTrue(self.stacking.final_model.is_fitted())

    def test_predict(self):
        self.stacking.fit(self.train_dataset)
        np.testing.assert_array_equal(self.stacking.predict(self.test_dataset), [0, 1, 0, 1])

    def test_score(self):
        self.assertEqual(self.stacking.fit(self.train_dataset).score(self.test_dataset), 1.)
