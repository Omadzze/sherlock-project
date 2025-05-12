#!/usr/bin/env python3
"""
Script for loading data and initializing the Sherlock fine-tuned model.
"""
import argparse
from ast import literal_eval
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

from sherlock.deploy.model import SherlockModel

def main(data_dir, model_id):
    # Log start time
    start = datetime.now()
    print(f"Started at {start}")

    # Load training data
    X_train = pd.read_parquet(f"{data_dir}/processed/train.parquet")
    y_train = pd.read_parquet(f"{data_dir}/raw/train_labels.parquet").values.flatten()
    y_train = np.array([x.lower() for x in y_train])
    print(f"Load data (train) process took {datetime.now() - start} seconds.")

    # Check data types
    print("Distinct types for columns in the DataFrame (should be all float32):")
    print(set(X_train.dtypes))

    # Load validation data
    start = datetime.now()
    print(f"Started at {start}")
    X_validation = pd.read_parquet(f"{data_dir}/processed/validation.parquet")
    y_validation = pd.read_parquet(f"{data_dir}/raw/validation_labels.parquet").values.flatten()
    y_validation = np.array([x.lower() for x in y_validation])
    print(f"Load data (validation) process took {datetime.now() - start} seconds.")

    # Load test data
    start = datetime.now()
    print(f"Started at {start}")
    X_test = pd.read_parquet(f"{data_dir}/processed/test.parquet")
    y_test = pd.read_parquet(f"{data_dir}/raw/test_labels.parquet").values.flatten()
    y_test = np.array([x.lower() for x in y_test])
    print(f"Finished at {datetime.now()}, took {datetime.now() - start} seconds")

    # Initialize and load the fine-tuned model
    model = SherlockModel()
    model.initialize_model_from_json(with_weights=True, model_id=model_id)

    base_model = model.model

    print(base_model.summary(line_length = 120))

    last_layer = base_model.layers[-1]
    print("Classifier layer:", last_layer.name, last_layer.output_shape)

    # Optionally, you can add evaluation code here

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sherlock fine-tuned model pipeline.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../custom_data",
        help="Base directory for processed and raw data folders"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="sherlock",
        help="ID of the fine-tuned Sherlock model"
    )
    args = parser.parse_args()
    main(args.data_dir, args.model_id)
