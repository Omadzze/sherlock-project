#!/usr/bin/env python
"""
Train and test Sherlock when ensembled with a RF classifier

This script loads training, validation, and test data, trains a VotingClassifier
using a RandomForestClassifier and an ExtraTreesClassifier, makes predictions with
each individual classifier and with the Sherlock NN model. In the end, it combines
the predictions from the VotingClassifier and Sherlock NN model to form a final
prediction, prints performance metrics, and reviews some errors.
"""

import os
import itertools
from ast import literal_eval
from collections import Counter
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score

# Import the Sherlock model
from sherlock.deploy.model import SherlockModel

# ================================================================
# Set environment variables (if desired)
# For deterministic results, set PYTHONHASHSEED prior to launch.
# In a Jupyter notebook you might use: %env PYTHONHASHSEED
# In this script, you can set it in Python as follows (if needed):
# os.environ['PYTHONHASHSEED'] = '13'
#
# The following IPython reload extension commands are specific to notebooks and have been omitted:
# %load_ext autoreload
# %autoreload 2
# ================================================================

def main():
    # ---------------------------
    # Load training and validation sets
    # ---------------------------
    start = datetime.now()
    print(f'Started at {start}')

    # Load train data
    X_train = pd.read_parquet('../data/data/processed/train.parquet')
    y_train = pd.read_parquet('../data/data/raw/train_labels.parquet').values.flatten()
    y_train = np.array([x.lower() for x in y_train])
    print(f'Load data (train) process took {datetime.now() - start} seconds.')

    # Print distinct types in X_train (should be all float32)
    print('Distinct types for columns in the Dataframe (should be all float32):')
    print(set(X_train.dtypes))

    # Load validation data
    start = datetime.now()
    print(f'\nStarted at {start}')
    X_validation = pd.read_parquet('../data/data/processed/validation.parquet')
    y_validation = pd.read_parquet('../data/data/raw/val_labels.parquet').values.flatten()
    y_validation = np.array([x.lower() for x in y_validation])
    print(f'Load data (validation) process took {datetime.now() - start} seconds.')

    # Concatenate training and validation sets
    X_train = pd.concat([X_train, X_validation], ignore_index=True)
    y_train = np.array([x.lower() for x in itertools.chain(y_train, y_validation)])

    # ---------------------------
    # Train Voting Classifier using RFC and ETC
    # ---------------------------
    # Using n_estimators=100 for each classifier.
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=100, random_state=13, n_jobs=-1)),
            ('et', ExtraTreesClassifier(n_estimators=100, random_state=13, n_jobs=-1))
        ],
        voting='soft'
    )
    start = datetime.now()
    print(f'\nStarted training VotingClassifier at {start}')
    voting_clf.fit(X_train, y_train)
    print(f'Finished training at {datetime.now()}, took {datetime.now() - start} seconds')

    # Make individual (trained) estimators available
    rf_clf = voting_clf.estimators_[0]
    et_clf = voting_clf.estimators_[1]

    # ---------------------------
    # Load test set
    # ---------------------------
    start = datetime.now()
    print(f'\nStarted loading test set at {start}')
    X_test = pd.read_parquet('../data/data/processed/test.parquet')
    y_test = pd.read_parquet('../data/data/raw/test_labels.parquet').values.flatten()
    y_test = np.array([x.lower() for x in y_test])
    print('Trained and saved new model.')
    print(f'Finished loading test set at {datetime.now()}, took {datetime.now() - start} seconds')

    # Optionally print unique labels in validation set
    print("\nUnique labels in validation set:")
    print(np.unique(y_validation))

    # ---------------------------
    # Make predictions
    # ---------------------------
    # Load classes file and ensure classes are sorted
    model_id = 'sherlock'
    classes = np.load(f"../model_files/classes_{model_id}.npy", allow_pickle=True)
    classes = np.array([cls.lower() for cls in classes])
    assert (classes == sorted(classes)).all(), "Classes are not sorted!"

    # Helper functions for converting prediction probabilities into labels and summarizing predictions
    def predicted_labels(y_pred_proba, classes):
        y_pred_int = np.argmax(y_pred_proba, axis=1)
        encoder = LabelEncoder()
        encoder.classes_ = classes
        return encoder.inverse_transform(y_pred_int)

    def prediction_summary(y_test, predicted_labels_arr):
        print(f'prediction count {len(predicted_labels_arr)}, type = {type(predicted_labels_arr)}')
        size = len(y_test)
        print(f'f1 score {f1_score(y_test[:size], predicted_labels_arr[:size], average="weighted")}')

    # Predict using RandomForestClassifier
    predicted_rfc_proba = rf_clf.predict_proba(X_test)
    print("\n--- RFC Prediction Summary ---")
    prediction_summary(y_test, predicted_labels(predicted_rfc_proba, classes))

    # Predict using ExtraTreesClassifier
    predicted_etc_proba = et_clf.predict_proba(X_test)
    print("\n--- ETC Prediction Summary ---")
    prediction_summary(y_test, predicted_labels(predicted_etc_proba, classes))

    # Predict using the Voting Classifier (RFC + ETC)
    predicted_voting_proba = voting_clf.predict_proba(X_test)
    print("\n--- Voting Classifier (RFC + ETC) Prediction Summary ---")
    prediction_summary(y_test, predicted_labels(predicted_voting_proba, classes))

    # Predict using Sherlock Neural Network
    print("\n--- Sherlock NN Prediction ---")
    model = SherlockModel()
    model.initialize_model_from_json(with_weights=True, model_id="sherlock")
    predicted_sherlock_proba = model.predict_proba(X_test)
    prediction_summary(y_test, predicted_labels(predicted_sherlock_proba, classes))

    # Predict Combined: average the probabilities from Voting Classifier and Sherlock NN
    combined = []
    for i in range(len(y_test)):
        nn_probs = predicted_sherlock_proba[i]
        voting_probs = predicted_voting_proba[i]
        x = (nn_probs + voting_probs) / 2  # Average the probabilities
        combined.append(x)

    labels = predicted_labels(np.array(combined), classes)
    print("\n--- Combined Prediction Summary ---")
    prediction_summary(y_test, labels)

    # ---------------------------
    # Generate Classification Report and Score Tables
    # ---------------------------
    report = classification_report(y_test, labels, output_dict=True)
    class_scores = list(filter(lambda x: isinstance(x, tuple) and isinstance(x[1], dict) and
                                         'f1-score' in x[1] and x[0] in classes,
                               list(report.items())))
    class_scores = sorted(class_scores, key=lambda item: item[1]['f1-score'], reverse=True)

    def score_table(class_scores_subset):
        print("\n\t\tf1-score\tprecision\trecall\t\tsupport")
        for key, value in class_scores_subset:
            tabs = '\t' if len(key) >= 8 else '\t\t'
            print(f"{key}{tabs}{value['f1-score']:.3f}\t\t{value['precision']:.3f}\t\t{value['recall']:.3f}\t\t{value['support']}")

    # Top 5 Types
    print("\n--- Top 5 Types ---")
    score_table(class_scores[0:5])

    # Bottom 5 Types
    print("\n--- Bottom 5 Types ---")
    score_table(class_scores[-5:])

    # All Scores (by class)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, labels, digits=3))

    # ---------------------------
    # Review errors
    # ---------------------------
    size = len(y_test)
    mismatches = []
    for idx, expected in enumerate(y_test[:size]):
        predicted = labels[idx]
        if expected != predicted:
            mismatches.append(expected)
    overall_f1 = f1_score(y_test[:size], labels[:size], average="weighted")
    print(f"\nTotal mismatches: {len(mismatches)} (F1 score: {overall_f1})")
    error_counts = Counter(mismatches)
    print("\nError counts (most common mismatches):")
    print(error_counts.most_common())

    # ---------------------------
    # Check a single test sample's details
    # ---------------------------
    test_samples = pd.read_parquet('../data/data/raw/test_values.parquet')
    idx = 541
    original = test_samples.iloc[idx]
    converted = original.apply(literal_eval).to_list()
    print(f'\nPredicted "{labels[idx]}", actual label "{y_test[idx]}". Actual values:\n{converted}')

    print(f'\nCompleted at {datetime.now()}')

if __name__ == "__main__":
    main()