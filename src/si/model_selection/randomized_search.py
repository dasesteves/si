import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
import itertools
from si.model_selection.cross_validate import k_fold_cross_validation


def randomized_search_cv(model:Model,dataset:Dataset,hyperparameter_grid:dict,cv:int,n_iter:int,scoring:callable = None)->dict:
    """
    Performs Randomized Hyperparameter Search

    Randomized hyperparameter search is an optimization technique that randomly samples
    combinations of hyperparameter values from a defined search space. This method is
    particularly advantageous for high-dimensional or large parameter spaces.

    Key Benefits:
    - **Efficiency**: Samples a wide range of hyperparameter values without exhaustively
    searching every combination.
    - **Flexibility**: Supports both continuous and discrete hyperparameter spaces.
    - **High-Dimensional Effectiveness**: Performs well with models involving numerous hyperparameters.

    Parameters:
    ----------
    model : Model
        The machine learning model to optimize.
    dataset : Dataset
        The dataset used for validation during hyperparameter tuning.
    hyperparameter_grid : dict
        A dictionary specifying hyperparameter names and their respective search ranges.
    scoring : callable
        A function to evaluate model performance during tuning.
    cv : int
        The number of cross-validation folds.
    n_iter : int
        The number of random hyperparameter combinations to evaluate.

    Returns:
    ----------
    results : dict
        A dictionary containing:
            - 'scores': Evaluation scores for each hyperparameter combination.
            - 'hyperparameters': The tested hyperparameter combinations.
            - 'best_hyperparameters': The hyperparameter combination yielding the best score.
            - 'best_score': The highest score achieved.
    """


    # check if the hyperparameter are present in the model
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")

    # select n_iter random combinations from all combinations possible for the hyperparameters
    combinations = random_combinations(hyperparameter_grid = hyperparameter_grid, n_iter=n_iter)

    # initializing the results dictionary
    results = {'scores': [], 'hyperparameters': []}

    for combination in combinations:

        # parameter configuration
        parameters = {}

        # set the parameters
        for parameter, value in zip(hyperparameter_grid.keys(), combination):
            setattr(model, parameter, value)
            parameters[parameter] = value

        # cross validate the model
        score = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        # add the score
        results['scores'].append(np.mean(score))

        # add the hyperparameters
        results['hyperparameters'].append(parameters)

    results['best_hyperparameters'] = results['hyperparameters'][np.argmax(results['scores'])]
    results['best_score'] = np.max(results['scores'])
    return results


def random_combinations(hyperparameter_grid:dict,n_iter:int)->list:
    """
    Selects Random Hyperparameter Combinations

    Randomly selects a specified number of hyperparameter combinations from all
    possible combinations defined in the hyperparameter grid.

    Parameters:
    -----------
    hyperparameter_grid : dict
        A dictionary where keys are hyperparameter names and values are lists or ranges
        of possible values for each hyperparameter.
    n_iter : int
        The number of hyperparameter combinations to randomly select.

    Returns:
    -----------
    random_combinations : list
        A list containing the randomly selected combinations of hyperparameters.
    """

    # computing all combinations of hyperparameters possible
    all_combinations = list(itertools.product(*hyperparameter_grid.values()))
    # select random indices form all combinations
    random_indices = np.random.choice(len(all_combinations),n_iter, replace=False)
    # select the random combinations from all combinations
    random_combinations = [all_combinations[idx] for idx in random_indices]

    return random_combinations