import numpy as np
from prefect import task
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


@task
def predict_age(input_data: csr_matrix, model: LinearRegression) -> np.ndarray:
    """
    Predict using the trained model.

    Parameters:
    - input_data (csr_matrix): Input features matrix.
    - model (LinearRegression): Trained linear regression model.

    Returns:
    - np.ndarray: Predicted values.
    """
    return model.predict(input_data)


@task
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Evaluate the model using root mean squared error.

    Parameters:
    - y_true (np.ndarray): True target values.
    - y_pred (np.ndarray): Predicted target values.

    Returns:
    - float: Root mean squared error.
    """
    return mean_squared_error(y_true, y_pred, squared=False)
