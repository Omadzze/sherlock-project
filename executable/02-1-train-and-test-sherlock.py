#!/usr/bin/env python
"""
This script trains and tests the Sherlock model.
Procedure:
- Loads train, validation, and test datasets (assumed to be preprocessed)
- Initializes the model either using pretrained weights or by training from scratch
- Evaluates and analyzes the model predictions
"""

from ast import literal_eval
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, classification_report

from sherlock.deploy.model import SherlockModel
from sherlock import helpers

def main():

    print("Downloading raw data and processed data")
    # Raw data
    helpers.download_data()
    # -------------------------------------------------------------------------
    # Set model identifier: use "retrained_sherlock" when training from scratch.
    # If you wish to use the pretrained model, adjust the code accordingly.
    #model_id = 'retrained_sherlock'

    model_id = "sherlock"

    # -------------------------------------------------------------------------
    # Load training data
    start = datetime.now()
    print(f'Started at {start}')

    X_train = pd.read_parquet('../data/data/processed/train.parquet')
    y_train = pd.read_parquet('../data/data/raw/train_labels.parquet').values.flatten()
    y_train = np.array([x.lower() for x in y_train])

    print(f'Load data (train) process took {datetime.now() - start} seconds.')
    print('Distinct types for columns in the DataFrame (should be all float32):')
    print(set(X_train.dtypes))

    # -------------------------------------------------------------------------
    # Load validation data
    start = datetime.now()
    print(f'\nStarted at {start}')

    X_validation = pd.read_parquet('../data/data/processed/validation.parquet')
    y_validation = pd.read_parquet('../data/data/raw/val_labels.parquet').values.flatten()
    y_validation = np.array([x.lower() for x in y_validation])

    print(f'Load data (validation) process took {datetime.now() - start} seconds.')

    # -------------------------------------------------------------------------
    # Load test data
    start = datetime.now()
    print(f'\nStarted at {start}')

    X_test = pd.read_parquet('../data/data/processed/test.parquet')
    y_test = pd.read_parquet('../data/data/raw/test_labels.parquet').values.flatten()
    y_test = np.array([x.lower() for x in y_test])

    print(f'Finished at {datetime.now()}, took {datetime.now() - start} seconds')

    # -------------------------------------------------------------------------
    # Option 1: Load Sherlock using pretrained weights (commented out)
    #
    # Uncomment the block below if you wish to use the pretrained model.
    #
    start = datetime.now()
    print(f'\nStarted at {start}')
    model = SherlockModel()
    model.initialize_model_from_json(with_weights=True, model_id="sherlock")
    print("Initialized model.")
    print(f"Finished at {datetime.now()}, took {datetime.now() - start} seconds")

    # -------------------------------------------------------------------------
    # Option 2: Train Sherlock from scratch (and save for later use)
    # Set or reassign model_id if needed.
    #model_id = "retrained_sherlock"

    #start = datetime.now()
    #print(f'\nStarted at {start}')

    #model = SherlockModel()
    # Train model with the training and validation data.
    #model.fit(X_train, y_train, X_validation, y_validation, model_id=model_id)

    #print("Trained and saved new model.")
    #print(f"Finished at {datetime.now()}, took {datetime.now() - start} seconds")

    # Store model weights (for later use)
    #model.store_weights(model_id=model_id)

    # -------------------------------------------------------------------------
    # Make predictions on the test set
    predicted_labels = model.predict(X_test, model_id=model_id)
    predicted_labels = np.array([x.lower() for x in predicted_labels])

    print(f'\nprediction count: {len(predicted_labels)}, type: {type(predicted_labels)}')
    size = len(y_test)
    f1 = f1_score(y_test[:size], predicted_labels[:size], average="weighted")
    print("Weighted F1 score:", f1)

    # -------------------------------------------------------------------------
    # Generate a classification report with detailed scores
    # (If using the pretrained model, set model_id = "sherlock" as needed.)
    classes = np.load(f"../model_files/classes_{model_id}.npy", allow_pickle=True)

    report = classification_report(y_test, predicted_labels, output_dict=True)
    class_scores = list(filter(lambda x: isinstance(x, tuple) and isinstance(x[1], dict)
                                         and 'f1-score' in x[1] and x[0] in classes,
                               list(report.items())))
    class_scores = sorted(class_scores, key=lambda item: item[1]['f1-score'], reverse=True)

    # -------------------------------------------------------------------------
    # Top 5 Types
    print("\n\t\tf1-score\tprecision\trecall\t\tsupport")
    for key, value in class_scores[0:5]:
        tabs = '\t' if len(key) >= 8 else '\t\t'
        print(f"{key}{tabs}{value['f1-score']:.3f}\t\t{value['precision']:.3f}\t\t{value['recall']:.3f}\t\t{value['support']}")

    # -------------------------------------------------------------------------
    # Bottom 5 Types
    print("\n\t\tf1-score\tprecision\trecall\t\tsupport")
    for key, value in class_scores[-5:]:
        tabs = '\t' if len(key) >= 8 else '\t\t'
        print(f"{key}{tabs}{value['f1-score']:.3f}\t\t{value['precision']:.3f}\t\t{value['recall']:.3f}\t\t{value['support']}")

    # -------------------------------------------------------------------------
    # Print full classification report
    print("\nFull Classification Report:")
    print(classification_report(y_test, predicted_labels, digits=3))

    # -------------------------------------------------------------------------
    # Review errors: Print specific error examples for label "address"
    size = len(y_test)
    mismatches = []
    for idx, k1 in enumerate(y_test[:size]):
        k2 = predicted_labels[idx]
        if k1 != k2:
            mismatches.append(k1)
            if k1 == "address":
                print(f'[{idx}] expected "address" but predicted "{k2}"')

    f1 = f1_score(y_test[:size], predicted_labels[:size], average="weighted")
    print(f'\nTotal mismatches: {len(mismatches)} (F1 score: {f1})')

    data = Counter(mismatches)
    print("Mismatches by label:", data.most_common())

    # -------------------------------------------------------------------------
    # Review a specific sample: inspect details for the sample at index 1001
    test_samples = pd.read_parquet('../data/data/raw/test_values.parquet')
    idx = 1001
    original = test_samples.iloc[idx]
    converted = original.apply(literal_eval).to_list()
    print(f'\nPredicted "{predicted_labels[idx]}", actual label "{y_test[idx]}". Actual values:')
    print(converted)


if __name__ == "__main__":
    main()