from fastapi import FastAPI
from .app_config import APP_TITLE, APP_DESCRIPTION
from .lib.models import InputData, OutputData
from .lib.utils import load_pickle
from .app_config import TRAIN_PATH, TEST_PATH, PREDICT_PATH, MODEL_PATH
from .lib.inference import process_data, predict_age, compute_target, extract_x_y, encode_categorical_cols
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
    data = {
        "Sex": abalones.Sex,
        "Length": abalones.Length,
        "Diameter": abalones.Diameter,
        "Height": abalones.Height,
        "Whole_weight": abalones.Whole_weight,
        "Shucked_weight": abalones.Shucked_weight,
        "Viscera_weight": abalones.Viscera_weight,
        "Shell_weight": abalones.Shell_weight,
    }
    df = pd.DataFrame([data])

    dv = load_pickle("src/web_service/local_objects/dv.pkl")
    model = load_pickle("src/web_service/local_objects/model.pkl")
    df2 = encode_categorical_cols(df)
    x, y, _ = extract_x_y(df2, dv=dv, with_target=False)
    
    predictions = predict_age(x, model)

    return {"Age": predictions}