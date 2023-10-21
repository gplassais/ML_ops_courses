# MODELS
MODEL_VERSION = "0.0.1"
PATH_TO_PREPROCESSOR = f"local_models/dv__v{MODEL_VERSION}.pkl"
PATH_TO_MODEL = f"local_models/model__v{MODEL_VERSION}.pkl"
CATEGORICAL_VARS = ["PULocationID", "DOLocationID", "passenger_count"]


# MISC
APP_TITLE = "AbaloneAgePredictionApp"
APP_DESCRIPTION = (
    "A simple API to predict abalone ages given"
    "different features extracted from a Kaggle dataset."
    "This app aims to simulate a ML industrialization for"
    "our MLOPS class"
)
APP_VERSION = "0.0.1"
