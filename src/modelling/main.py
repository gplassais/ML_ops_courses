from config.config import TRAIN_PATH, TEST_PATH, PREDICT_PATH, MODEL_PATH
from loguru import logger
from training import train_model
from preprocessing import process_data
from predicting import predict_age, evaluate_model
from utils import save_pickle, load_pickle
from prefect import flow, serve
from typing import Optional
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
import numpy as np

@flow(retries=3, retry_delay_seconds=5, log_prints=True)
def train_model_workflow(trainset_path: str,
                         testset_path: str,
                         artifacts_path:Optional[str] = None) -> dict:
    """Train a model using the data at the given path and save the model (pickle)."""
    # Process training data
    logger.info("Processing training data...")
    X_train, y_train, dv = process_data(filepath=trainset_path)

    # Process testing data
    logger.info("Processing testing data...")
    X_test, y_test, _ = process_data(testset_path, dv=dv)

    # Train model
    logger.info("Training model...")
    model = train_model(X_train, y_train)

    # Evaluating model
    y_pred = predict_age(X_test, model)
    mse = evaluate_model(y_test, y_pred)

    # Save encoder and model
    if artifacts_path:
        save_pickle(artifacts_path + "/model.pkl", model)
        save_pickle(artifacts_path + "/dv.pkl", dv)
        logger.info("Model and encoder saved !")

    # Return MSE
    dict = {"model": model, "dv": dv, "mse": mse}
    return dict

@flow(retries=3, retry_delay_seconds=5, log_prints=True)
def batch_predict_workflow(
    input_path: str,
    model: Optional[LinearRegression] = None,
    dv: Optional[DictVectorizer] = None,
    artifacts_path: Optional[str] = None,
) -> np.ndarray:
    """Load input data, apply data processing, use a trained model to make predictions on the data, and return the predicted target values.

    Args:
        input_filepath (str): The file path to the input data.
        model (Optional[LinearRegression]): The trained linear regression model (default: None).
        dv (Optional[DictVectorizer]): The fitted DictVectorizer object (default: None).
        artifacts_filepath (Optional[str]): The file path to load the trained model and data vectorizer artifacts (default: None).

    Returns:
        np.ndarray: The predicted target values.

    """    
        
    if dv is None:
        dv = load_pickle(artifacts_path + "/dv.pkl")
    if model is None:
        model = load_pickle(artifacts_path + "/model.pkl")
    
    processed_data, _, _ = process_data(input_path, dv=dv, with_target=False)
    predictions = predict_age(processed_data, model)
    return predictions


if __name__ == "__main__":
    
    train_workflow = train_model_workflow(
        trainset_path=TRAIN_PATH,
        testset_path=TEST_PATH,
        artifacts_path=MODEL_PATH,
    )

    predict_workflow = batch_predict_workflow(
        input_path=PREDICT_PATH,
        artifacts_path=MODEL_PATH
        )
    serve(train_workflow)
    serve(predict_workflow)
