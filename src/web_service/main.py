# Code with FastAPI (app = FastAPI(...))


from fastapi import FastAPI
from app_config import APP_TITLE, APP_DESCRIPTION
from lib.models import InputData, OutputData
from lib.utils import load_pickle
from app_config import TRAIN_PATH, TEST_PATH, PREDICT_PATH, MODEL_PATH
from lib.inference import process_data, predict_age, compute_target, extract_x_y, encode_categorical_cols
import pandas as pd
from typing import List


# Other imports

app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION)


@app.get("/")
def home() -> dict:
    return {"health_check": "App up and running!"}


@app.post("/predict", response_model=OutputData, status_code=201)
def predict(abalones: InputData) -> dict:
    """Load input data, apply data processing, use a trained model to make predictions on the data, and return the predicted target values.

    Args:
        input_filepath (str): The file path to the input data.
        model (Optional[LinearRegression]): The trained linear regression model (default: None).
        dv (Optional[DictVectorizer]): The fitted DictVectorizer object (default: None).
        artifacts_filepath (Optional[str]): The file path to load the trained model and data vectorizer artifacts (default: None).

    Returns:
        np.ndarray: The predicted target values.

    """    
    # logger.info(f"Running inference on:\n{input_data}")
    data = {
        "Sex": "M",
        "Length": 0,
        "Diameter": 2.0,
        "Height": 3.,
        "Whole_weight": 0.5,
        "Shucked_weight": 0,
        "Viscera_weight": 0,
        "Shell_weight": 0,
        "Rings": 0
    }
    # df = pd.DataFrame([abalones])
    df = pd.DataFrame([data])

    # cols = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
    #    'Viscera weight', 'Shell weight', 'Rings']

    dv = load_pickle(MODEL_PATH + "/dv.pkl")
    model = load_pickle(MODEL_PATH + "/model.pkl")
    # processed_data, _, _ = process_data(df, dv=dv, with_target=False)
    df1 = compute_target(df)
    # logger.debug(f"{filepath} | Encoding categorical columns...")
    df2 = encode_categorical_cols(df1)
    # logger.debug(f"{filepath} | Extracting X and y...")
    x, y, _ = extract_x_y(df2, dv=dv)
    
    predictions = predict_age(x, model)
    
    return {"Age": 12}