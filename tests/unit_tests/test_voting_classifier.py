from unittest import TestCase
import numpy as np
from si.ensemble.voting_classifier import VotingClassifier
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.models.knn_classifier import KNNClassifier
from synthetic_classification import training_data, evaluation_data, logistic_model


class TestVotingClassifier(TestCase):
    def setUp(self):
        self.train_dataset, self.test_dataset = training_data(), evaluation_data()
        self.voting = VotingClassifier([DecisionTreeClassifier(), KNNClassifier(k=3), logistic_model()])

    def test_fit(self):
        self.voting.fit(self.train_dataset)
        self.assertTrue(self.voting.is_fitted())
        self.assertTrue(all(model.is_fitted() for model in self.voting.models))

    def test_predict(self):
        self.voting.fit(self.train_dataset)
        np.testing.assert_array_equal(self.voting.predict(self.test_dataset), [0, 1, 0, 1])

    def test_score(self):
        self.assertEqual(self.voting.fit(self.train_dataset).score(self.test_dataset), 1.)
