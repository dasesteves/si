from unittest import TestCase
import numpy as np
from si.model_selection.grid_search_cv import grid_search_cv
from synthetic_classification import training_data, logistic_model


class TestGridSearchCV(TestCase):
    def test_grid_search_k_fold_cross_validation(self):
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(0)
        grid = {'l2_penalty': (.1, .5), 'alpha': (.05, .1), 'max_iter': (100, 200)}
        results = grid_search_cv(logistic_model(), training_data(), hyperparameter_grid=grid, cv=3)
        self.assertEqual(len(results['scores']), 8)
        combinations = {tuple(sorted(params.items())) for params in results['hyperparameters']}
        self.assertEqual(len(combinations), 8)
        self.assertEqual(results['best_score'], 1.)
        self.assertIn(results['best_hyperparameters'], results['hyperparameters'])
        for params in results['hyperparameters']:
            self.assertEqual(set(params), set(grid))
            for name, value in params.items():
                self.assertIn(value, grid[name])
