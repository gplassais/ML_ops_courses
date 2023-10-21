# Code with FastAPI (app = FastAPI(...))


from fastapi import FastAPI
from app_config import APP_TITLE, APP_DESCRIPTION
from lib.models import InputData, OutputData

# Other imports

app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION)


@app.get("/")
def home() -> dict:
    return {"health_check": "App up and running!"}


@app.post(
    "/predict",
    response_model=OutputData,
    status_code=201,
)
def predict(payload: InputData) -> dict:
    return {"Age": 0}
