#!/usr/bin/env python3
"""
This script replicates the functionality of the original Jupyter notebook.
It downloads data, extracts features from raw parquet files, performs
imputation on missing values, and saves the processed data as CSV and parquet files.
Ensure that your environment is correctly set up (see HOW-TO-ENVIRONMENT.md)
and that any required data and models are available (for example, run the
"01-train-paragraph-vector-features" notebook to generate the required paragraph vectors).
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

# Import the necessary functions from the sherlock package.
from sherlock import helpers
from sherlock.functional import extract_features_to_csv
from sherlock.features.paragraph_vectors import initialise_pretrained_model, initialise_nltk
from sherlock.features.preprocessing import (
    extract_features,
    convert_string_lists_to_lists,
    prepare_feature_extraction,
    load_parquet_values,
)
from sherlock.features.word_embeddings import initialise_word_embeddings


def main():
    # Print start timestamp.
    print(f"Started at {datetime.now()}.")

    # Download raw data and prepare feature extraction files.
    print("Downloading data and preparing feature extraction files.")
    helpers.download_data()
    prepare_feature_extraction()

    # Check that the trained paragraph vector file exists.
    par_vec_file = '../sherlock/features/par_vec_trained_400.pkl.docvecs.vectors_docs.npy'
    if not os.path.exists(par_vec_file):
        raise SystemExit(
            "Trained paragraph vectors do not exist,\n"
            "please run the '01-train-paragraph-vector-features' notebook before continuing."
        )

        # Ensure that the processed directory exists.
    processed_dir = '../data/data/processed'
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        print(f"Created directory: {processed_dir}")

    # Generate a timestamp string to use in output filenames.
    timestr = time.strftime("%Y%m%d-%H%M%S")
    X_test_filename_csv = f'../data/data/processed/test_{timestr}.csv'
    X_train_filename_csv = f'../data/data/processed/train_{timestr}.csv'
    X_validation_filename_csv = f'../data/data/processed/validation_{timestr}.csv'

    # Initialize embeddings and other resources.
    # (Re-run prepare_feature_extraction() to ensure all files are present.)
    prepare_feature_extraction()
    initialise_word_embeddings()
    initialise_pretrained_model(400)
    initialise_nltk()

    # -------------------------------------------------
    # Extract features to CSV files.
    # -------------------------------------------------

    # --- TEST SET ---
    print(f"Starting feature extraction for test set at {datetime.now()}.")
    test_values = load_parquet_values("../data/data/raw/test_values.parquet")
    extract_features_to_csv(X_test_filename_csv, test_values)
    test_values = None  # release memory
    print(f"Finished test set at {datetime.now()}.")

    # --- TRAIN SET ---
    print(f"Starting feature extraction for train set at {datetime.now()}.")
    train_values = load_parquet_values("../data/data/raw/train_values.parquet")
    extract_features_to_csv(X_train_filename_csv, train_values)
    train_values = None  # release memory
    print(f"Finished train set at {datetime.now()}.")

    # --- VALIDATION SET ---
    print(f"Starting feature extraction for validation set at {datetime.now()}.")
    validation_values = load_parquet_values("../data/data/raw/val_values.parquet")
    extract_features_to_csv(X_validation_filename_csv, validation_values)
    validation_values = None  # release memory
    print(f"Finished validation set at {datetime.now()}.")

    # -------------------------------------------------
    # Load the processed CSV files as pandas DataFrames.
    # -------------------------------------------------

    print("Loading processed CSV features.")

    start = datetime.now()
    X_test = pd.read_csv(X_test_filename_csv, dtype=np.float32)
    print(f"Load Features (test) process took {datetime.now() - start} seconds.")
    print("Test set preview:")
    print(X_test.head())

    start = datetime.now()
    X_train = pd.read_csv(X_train_filename_csv, dtype=np.float32)
    print(f"Load Features (train) process took {datetime.now() - start} seconds.")
    print("Train set preview:")
    print(X_train.head())

    start = datetime.now()
    X_validation = pd.read_csv(X_validation_filename_csv, dtype=np.float32)
    print(f"Load Features (validation) process took {datetime.now() - start} seconds.")
    print("Validation set preview:")
    print(X_validation.head())

    # -------------------------------------------------
    # Save preliminary backups before imputing missing (NaN) values.
    # -------------------------------------------------
    print("Saving backup copies of the raw processed data (pre-imputation).")
    X_test.to_csv(f'{processed_dir}/test_preimpute_{timestr}.csv', index=False)
    X_train.to_csv(f'{processed_dir}/train_preimpute_{timestr}.csv', index=False)
    X_validation.to_csv(f'{processed_dir}/validation_preimpute_{timestr}.csv', index=False)
    print("Backup copies saved.")

    # -------------------------------------------------
    # Impute missing (NaN) values with the mean of each feature from the training set.
    # -------------------------------------------------

    print("Imputing NaN values with feature means.")
    start = datetime.now()
    # Calculate the mean of each column from the training set.
    train_columns_means = pd.DataFrame(X_train.mean()).transpose()
    print(f"Transpose process took {datetime.now() - start} seconds.")

    start = datetime.now()
    # Replace NaN values in each set with the corresponding training set mean.
    X_train.fillna(train_columns_means.iloc[0], inplace=True)
    X_validation.fillna(train_columns_means.iloc[0], inplace=True)
    X_test.fillna(train_columns_means.iloc[0], inplace=True)
    train_columns_means = None  # clear variable to free memory
    print(f"FillNA process took {datetime.now() - start} seconds.")

    # -------------------------------------------------
    # Save the DataFrames to parquet files.
    # -------------------------------------------------

    print("Saving processed DataFrames to parquet files.")
    start = datetime.now()
    X_train.to_parquet('../data/data/processed/train.parquet', engine='pyarrow', compression='snappy')
    X_validation.to_parquet('../data/data/processed/validation.parquet', engine='pyarrow', compression='snappy')
    X_test.to_parquet('../data/data/processed/test.parquet', engine='pyarrow', compression='snappy')
    print(f"Save parquet process took {datetime.now() - start} seconds.")

    print(f"Completed at {datetime.now()}.")


if __name__ == "__main__":
    main()