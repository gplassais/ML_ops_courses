from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = str(PROJECT_ROOT / 'data')
TRAIN_PATH = DATA_PATH + '/abalone_train.csv'
TEST_PATH = DATA_PATH + '/abalone_test.csv'
PREDICT_PATH = DATA_PATH + '/abalone_predict.csv'

MODEL_PATH = 'src/web_service/local_objects'

CATEGORICAL_COLS = ["Sex"]