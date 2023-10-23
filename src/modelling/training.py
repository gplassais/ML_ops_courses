from scipy.sparse import csr_matrix 
import numpy as np
from sklearn.linear_model import LinearRegression
from prefect import task

@task
def train_model(x_train: csr_matrix, y_train: np.ndarray) -> LinearRegression:
    """
    Train a linear regression model.
    
    Parameters:
    - x_train (csr_matrix): Features matrix.
    - y_train (np.ndarray): Target array.
    
    Returns:
    - LinearRegression: Trained linear regression model.
    """
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    return lr