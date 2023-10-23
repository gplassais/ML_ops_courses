# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse
from pathlib import Path
from loguru import logger
from training import train_model
from preprocessing import process_data
from utils import save_pickle
from config.config import TRAIN_PATH


def main(trainset_path: Path = TRAIN_PATH) -> None:
    """Train a model using the data at the given path and save the model (pickle)."""
    # Read data
    logger.info("Processing training data...")
    X_train, y_train, dv = process_data(filepath=trainset_path)
    # (Optional) Pickle encoder if need be
    save_pickle("src/web_service/local_objects" + "/dv.pkl", dv)
    # Train model
    logger.info("Training model...")
    model = train_model(X_train, y_train)
    # Pickle model --> The model should be saved in pkl format the `src/web_service/local_objects` folder
    save_pickle("src/web_service/local_objects" + "/model.pkl", model)
    logger.info("Done !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using the data at the given path.")
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    args = parser.parse_args()
    main(args.trainset_path)
