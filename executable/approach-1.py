#!/usr/bin/env python3
"""
Script for loading data, initializing the Sherlock fine-tuned model,
and applying confidence thresholding and class holdout.
"""
import argparse
import time
from ast import literal_eval
from collections import Counter
from datetime import datetime
import random

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from sherlock.deploy.model import LabelEncoder, SherlockModel
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from sherlock.deploy import helpers

# Reproducibility
SEED = 13
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Confidence threshold for predictions
THRESHOLD = 0.5

# Classes to hold out (never seen during training/validation/testing)
HOLDOUT_CLASSES = [
    "contact_setting",
    "occupation",
    "symptoms",
]

# Original label mapping
LABEL_MAP = {
    # Date
    "Vaccination_date": "date",
    "Date_report": "date",
    "Date_onset":  "date",
    "Date_confirmation": "date",
    "Date_of_first_consultation": "date",
    "Date_hospitalisation":  "date",
    "Date_discharge_hospital": "date",
    "Date_admission_ICU":   "date",
    "Date_discharge_ICU":  "date",
    "Date_isolation":  "date",
    "Date_death":  "date",
    "Date_recovered":  "date",
    "Travel_history_entry": "date",
    "Travel_history_start":  "date",
    "Date_entry":  "date",
    "Date_last_modified": "date",

    # ID
    "Contact_ID": "id",
    "ID": "id",

    # Gender
    "Gender": "gender",
    "Sex_at_birth": "gender",
    "Gender_other": "gender",
    "Sex_at_birth_other": "gender",

    # Location
    "Travel_history_location": "location",
    "Location_information": "location",

    # Contact setting
    "Contact_setting": "contact_setting",
    "Contact_setting_other": "contact_setting",

    # Demographic
    "Race": "demographic",
    "Ehtnicity": "demographic",

    # Medical Boolean
    "Healthcare_worker": "medical_boolean",
    "Previous_infection": "medical_boolean",
    "Pregnancy_Status": "medical_boolean",
    "Vaccination":  "medical_boolean",
    "Hospitalised":  "medical_boolean",
    "Intensive_care":  "medical_boolean",
    "Home_monitoring":  "medical_boolean",
    "Isolated": "medical_boolean",
    "Contact_with_case": "medical_boolean",
    "Travel_history": "medical_boolean",
    "Contact_animal": "medical_boolean",

    # Source
    "Source": "source",
    "Source_II": "source",
    "Source_III": "source",
    "Source_IV": "source",
}

LABEL_MAP_LC = {k.lower(): v.lower() for k, v in LABEL_MAP.items()}

def remap_labels(arr, mapping=LABEL_MAP_LC):
    """
    • forces each element to lower-case
    • replaces it if the key exists in `mapping`
    • otherwise leaves it as lower-case original
    """
    return np.array([mapping.get(x.lower(), x.lower()) for x in arr])

def make_inputs(df):
    feature_cols = helpers.categorize_features()
    return [
        df[feature_cols["char"]].values.astype("float32"),
        df[feature_cols["word"]].values.astype("float32"),
        df[feature_cols["par"]].values.astype("float32"),
        df[feature_cols["rest"]].values.astype("float32"),
    ]

def filter_holdout(X_df, y_arr):
    mask = ~np.isin(y_arr, HOLDOUT_CLASSES)
    return X_df[mask], y_arr[mask]


def main(data_dir, model_id):
    # Load data
    X_train = pd.read_parquet(f"{data_dir}/processed/train.parquet")
    y_train = pd.read_parquet(f"{data_dir}/raw/train_labels.parquet").values.flatten()
    y_train = remap_labels(y_train)
    y_train = np.array([x.lower() for x in y_train])

    # Filter out holdout classes from training
    X_train, y_train = filter_holdout(X_train, y_train)

    X_validation = pd.read_parquet(f"{data_dir}/processed/validation.parquet")
    y_validation = pd.read_parquet(f"{data_dir}/raw/validation_labels.parquet").values.flatten()
    y_validation = remap_labels(y_validation)
    y_validation = np.array([x.lower() for x in y_validation])
    X_validation, y_validation = filter_holdout(X_validation, y_validation)

    X_test = pd.read_parquet(f"{data_dir}/processed/test.parquet")
    y_test = pd.read_parquet(f"{data_dir}/raw/test_labels.parquet").values.flatten()
    y_test = remap_labels(y_test)
    y_test = np.array([x.lower() for x in y_test])
    X_test, y_test = filter_holdout(X_test, y_test)

    # Encode labels
    le = LabelEncoder().fit(y_train)
    class_names = le.classes_
    NUM_LABELS = len(class_names)

    # Initialize and load the fine-tuned base model
    wrapper = SherlockModel()
    wrapper.initialize_model_from_json(with_weights=True, model_id=model_id)
    base_model = wrapper.model

    # Build fine-tune head
    penultimate = base_model.get_layer("dense_7").output
    logits = Dense(NUM_LABELS, activation="softmax", name="classifier")(penultimate)
    finetune_model = Model(inputs=base_model.input, outputs=logits, name="sherlock_finetune")

    finetune_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Prepare inputs
    train_inputs = make_inputs(X_train)
    val_inputs = make_inputs(X_validation)
    test_inputs = make_inputs(X_test)

    y_train_int = le.transform(y_train)
    y_val_int = le.transform(y_validation)

    print("Classes:", class_names)      # ['age', 'city', 'pre_existing_condition', ...]
    print("num_classes:", NUM_LABELS)  # e.g. 20
    print("y_train_int shape:", y_train_int.shape)   # (195,)
    print("train_inputs[0].dtype:", train_inputs[0].dtype)  # float32

    # Fine-tuning
    start = time.perf_counter()
    finetune_model.fit(
        train_inputs, y_train_int,
        validation_data=(val_inputs, y_val_int),
        epochs=20,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )
    print(f"Fine-tuning completed in {time.perf_counter() - start:.2f}s")

    # Save model
    finetune_model.save_weights("my_custom_sherlock_head.h5")
    new_id = "sherlock_fine_tuned"
    model_dir = "../model_files/"
    with open(f"{model_dir}/{new_id}_model.json", "w") as f:
        f.write(finetune_model.to_json())
    finetune_model.save_weights(f"{model_dir}/{new_id}_weights.h5")

    # Load weights for inference
    finetune_model.load_weights("my_custom_sherlock_head.h5")

    # Inference
    start_inf = time.perf_counter()
    y_pred_probs = finetune_model.predict(test_inputs, batch_size=256)
    print(f"Inference completed in {time.perf_counter() - start_inf:.2f}s")

    # Apply confidence threshold: filter out low-confidence samples
    y_pred_idx = np.argmax(y_pred_probs, axis=1)
    max_probs = np.max(y_pred_probs, axis=1)
    mask_confident = max_probs >= THRESHOLD
    # Filter true labels and predictions
    y_true_conf = y_test[mask_confident]
    y_pred_conf_idx = y_pred_idx[mask_confident]

    # Evaluate on confident samples only
    print(classification_report(
        le.transform(y_true_conf),
        y_pred_conf_idx,
        target_names=class_names,
        digits=3,
        zero_division=0
    ))
    cm = confusion_matrix(
        le.transform(y_true_conf), y_pred_conf_idx
    )
    print("Confusion matrix shape:", cm.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Sherlock fine-tuned model with threshold and holdout"
    )
    parser.add_argument(
        "--data_dir", type=str, default="../custom_data",
        help="Base directory for processed and raw data"
    )
    parser.add_argument(
        "--model_id", type=str, default="sherlock",
        help="ID of the fine-tuned Sherlock model"
    )
    args = parser.parse_args()
    main(args.data_dir, args.model_id)