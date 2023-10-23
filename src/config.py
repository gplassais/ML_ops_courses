from pathlib import Path

DATA_PATH = '../data'
TRAIN_PATH = Path(DATA_PATH, 'abalone_train.csv')
TEST_PATH = Path(DATA_PATH, 'abalone_test.csv')

CATEGORICAL_COLS = ["Sex"]
