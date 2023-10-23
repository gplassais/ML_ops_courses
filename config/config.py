from pathlib import Path

DATA_PATH = 'data'
TRAIN_PATH = Path(DATA_PATH, 'abalone_train.csv')
TEST_PATH = Path(DATA_PATH, 'abalone_test.csv')
PREDICT_PATH = Path(DATA_PATH, 'abalone_test.csv')

MODEL_PATH = 'src/web_service/local_object'

CATEGORICAL_COLS = ["Sex"]
