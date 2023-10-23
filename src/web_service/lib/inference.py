from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression
import numpy as np
from typing import List, Tuple, Union
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix
from loguru import logger
from sklearn.feature_extraction import DictVectorizer


def load_data(path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Parameters:
    - path (str): Path to the CSV file.
    
    Returns:
    - pd.DataFrame: Loaded dataframe.
    """
    df = pd.read_csv(path)
    return df


def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 'Age' column based on the 'Rings' column and drop the 'Rings' column.
    
    Parameters:
    - df (pd.DataFrame): Input dataframe.
    
    Returns:
    - pd.DataFrame: Dataframe with 'Age' column computed and 'Rings' column dropped.
    """
    df['Age'] = df['Rings'] + 1.5
    return df.drop('Rings', axis=1)


def encode_categorical_cols(
    df: pd.DataFrame, categorical_cols: List[str] = None
) -> pd.DataFrame:
    """
    Encode categorical columns as strings.
    
    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - categorical_cols (List[str]): List of categorical columns to encode.
    
    Returns:
    - pd.DataFrame: Dataframe with categorical columns encoded.
    """
    if categorical_cols is None:
        categorical_cols = ["Sex"]
    df[categorical_cols] = df[categorical_cols].astype("str")
    return df


def extract_x_y(
    df: pd.DataFrame,
    categorical_cols: List[str] = None,
    dv: DictVectorizer = None,
    with_target: bool = True,
) -> Tuple[csr_matrix, Union[np.ndarray, None], DictVectorizer]:
    """
    Extract features and target from the dataframe.
    
    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - categorical_cols (List[str]): List of categorical columns.
    - dv (DictVectorizer): DictVectorizer instance.
    - with_target (bool): Whether to extract target or not.
    
    Returns:
    - Tuple: Features matrix, target array, and DictVectorizer instance.
    """
    if categorical_cols is None:
        categorical_cols = ["Sex"]
    dicts = df[categorical_cols].to_dict(orient="records")

    y = None
    if with_target:
        if dv is None:
            dv = DictVectorizer()
            dv.fit(dicts)
        y = df["Age"].values

    x = dv.transform(dicts)
    return x, y, dv



def process_data(filepath: str, dv=None, with_target: bool = True) -> scipy.sparse.csr_matrix:
    """
    Load data from a parquet file
    Compute target (duration column) and apply threshold filters (optional)
    Turn features to sparce matrix
    :return The sparce matrix, the target' values and the
    dictvectorizer object if needed.
    """
    df = load_data(filepath)
    if with_target:
        logger.debug(f"{filepath} | Computing target...")
        df1 = compute_target(df)
        logger.debug(f"{filepath} | Encoding categorical columns...")
        df2 = encode_categorical_cols(df1)
        logger.debug(f"{filepath} | Extracting X and y...")
        return extract_x_y(df2, dv=dv)
    else:
        logger.debug(f"{filepath} | Encoding categorical columns...")
        df1 = encode_categorical_cols(df)
        logger.debug(f"{filepath} | Extracting X and y...")
        return extract_x_y(df1, dv=dv, with_target=with_target)
    

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