from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb


class PrefXGB:
    def __init__(self, params=None, criteria_nr=4):
        self.params = params if params else self.default_model_params(criteria_nr)
        self.model = xgb.XGBClassifier(**self.params)

    def default_model_params(self, criteria_nr: int):
        params = {
            "max_depth": criteria_nr * 2,  # Maximum depth of a tree
            "eta": 0.1,  # Learning rate
            "nthread": 2,  # Number of parallel threads
            "seed": 0,  # Random seed
            "eval_metric": "rmse",  # Evaluation metric
            "monotone_constraints": "("
            + ",".join(["1"] * criteria_nr)
            + ")",  # Monotonic constraints for each criterion (1 = increasing, -1 = decreasing, 0 = no constraint)
            "n_estimators": 1,  # Number of boosting rounds, or trees
        }
        return params

    def partial_dependency(
        self, X: np.ndarray, y: pd.Series, f_id: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the partial dependency of a feature on the predicted outcome.


        Args:
            booster (xgb.Booster): The trained XGBoost model.
            X (np.ndarray): The input feature matrix.
            y (pd.Series): The target variable.
            f_id (int): The index of the feature for which the partial dependency is calculated.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
            - grid: The values of the feature used for calculation.
            - y_pred: The predicted outcomes corresponding to each value in the grid.
        """
        grid = np.linspace(0, 1, 50)
        y_pred = np.zeros(len(grid))
        for i, val in enumerate(grid):
            X_temp = X.copy()
            X_temp[:, f_id] = val
            data = xgb.DMatrix(pd.DataFrame(X_temp))
            y_pred[i] = np.average(self.model.get_booster().predict(data))
        return grid, y_pred

def get_metric(X, y, model, metric):
        y_pred = model.predict(X)
        predictions = [round(value) for value in y_pred]
        metric_value = metric(y, predictions)
        return metric_value

def load_data(
    path: str, target_map: dict, criteria_nr: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Preprocesses the data for training a machine learning model.

    Args:
        path (str): The path to the CSV file containing the data.
        target_map (dict): A dictionary mapping target values to binary labels.
        criteria_nr (int): The number of criteria used for classification.

    Returns:
        tuple: A tuple containing the preprocessed data and the train-test split.
    """
    data = pd.read_csv(path, header=None)
    data[criteria_nr] = data[criteria_nr].apply(lambda x: target_map[x])

    data = data.drop_duplicates()

    data_input = data.iloc[:, :criteria_nr]
    data_target = data[criteria_nr]

    X_train, X_test, y_train, y_test = train_test_split(
        data_input, data_target, test_size=0.2, random_state=1234
    )

    return (X_train, X_test, y_train, y_test)
