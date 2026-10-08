from unittest import TestCase
import numpy as np
from si.metrics.accuracy import accuracy
from si.model_selection.cross_validate import k_fold_cross_validation
from synthetic_classification import training_data, logistic_model


class TestKFoldCrossValidation(TestCase):
    def test_k_fold_cross_validation(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        scores = k_fold_cross_validation(logistic_model(), training_data(), scoring=accuracy, cv=3, seed=0)
        np.testing.assert_array_equal(scores, [1., 1., 1.])
