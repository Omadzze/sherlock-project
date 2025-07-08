#!/usr/bin/env python3
"""
Script for loading data, initializing the Sherlock fine-tuned model,
and applying confidence thresholding, class holdout, and displaying raw inputs.
"""
import argparse
import os
import time
import random

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from sherlock.deploy.model import LabelEncoder, SherlockModel
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from sherlock.deploy import helpers

# Reproducibility
SEED = 13
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_random_seed(SEED)

# Confidence threshold for predictions
THRESHOLD = 0.5
UNKNOWN_LABEL = 'unknown'

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
    # --- Load and filter training data ---
    X_train = pd.read_parquet(f"{data_dir}/processed/train.parquet")
    y_train = pd.read_parquet(f"{data_dir}/raw/train_labels.parquet").values.flatten()
    y_train = remap_labels(y_train)
    y_train = np.array([x.lower() for x in y_train])
    X_train, y_train = filter_holdout(X_train, y_train)

    # --- Load and filter validation data ---
    X_val = pd.read_parquet(f"{data_dir}/processed/validation.parquet")
    y_val = pd.read_parquet(f"{data_dir}/raw/validation_labels.parquet").values.flatten()
    y_val = remap_labels(y_val)
    y_val = np.array([x.lower() for x in y_val])
    X_val, y_val = filter_holdout(X_val, y_val)

    # --- Load and filter test data ---
    X_test_raw = pd.read_parquet(f"{data_dir}/processed/test.parquet")
    y_test = pd.read_parquet(f"{data_dir}/raw/test_labels.parquet").values.flatten()
    y_test = remap_labels(y_test)
    y_test = np.array([x.lower() for x in y_test])
    #X_test, y_test = filter_holdout(X_test, y_test)

    mask_holdout = ~np.isin(y_test, HOLDOUT_CLASSES)
    X_test, y_test = filter_holdout(X_test_raw, y_test)

    # Encode labels
    le = LabelEncoder().fit(y_train)
    class_names = le.classes_
    num_labels = len(le.classes_)

    # Initialize base model
    wrapper = SherlockModel()
    wrapper.initialize_model_from_json(with_weights=True, model_id=model_id)
    base_model = wrapper.model

    # Build fine-tune head
    penultimate = base_model.get_layer("dense_7").output
    logits = Dense(len(class_names), activation="softmax", name="classifier")(penultimate)
    finetune_model = Model(inputs=base_model.input, outputs=logits)
    finetune_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Prepare inputs
    train_inputs = make_inputs(X_train)
    val_inputs = make_inputs(X_val)
    test_inputs = make_inputs(X_test)
    y_train_int = le.transform(y_train)
    y_val_int = le.transform(y_val)

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
    model_dir = "../model_files/"
    with open(f"{model_dir}/{model_id}_model.json", "w") as f:
        f.write(finetune_model.to_json())
    finetune_model.save_weights(f"{model_dir}/{model_id}_weights.h5")

    # Inference
    start_inf = time.perf_counter()
    y_pred_probs = finetune_model.predict(test_inputs, batch_size=256)
    print(f"Inference completed in {time.perf_counter() - start_inf:.2f}s")

    # Top-3 predictions
    top3_idx = np.argsort(y_pred_probs, axis=1)[:, -3:][:, ::-1]
    y_top3_labels = [[class_names[i] for i in row] for row in top3_idx]
    y_top3_probs = [row[idxs] for row, idxs in zip(y_pred_probs, top3_idx)]

    # Thresholded predictions
    max_probs = np.max(y_pred_probs, axis=1)
    y_pred_idx = np.argmax(y_pred_probs, axis=1)
    y_pred_labels = [class_names[i] if p >= THRESHOLD else UNKNOWN_LABEL
                     for i, p in zip(y_pred_idx, max_probs)]

    # Evaluate confident only
    mask = np.array(y_pred_labels) != UNKNOWN_LABEL
    y_t_conf = le.transform(y_test[mask])
    y_p_conf = y_pred_idx[mask]
    present = np.unique(np.concatenate([y_t_conf, y_p_conf]))
    print(classification_report(
        y_t_conf, y_p_conf,
        labels=present, target_names=class_names[present],
        digits=3, zero_division=0
    ))
    print("Confusion matrix shape:", confusion_matrix(y_t_conf, y_p_conf, labels=present).shape)

    # --- after you have X_test, y_test, and have done:
    #mask_holdout = ~np.isin(y_test, HOLDOUT_CLASSES)

    raw_df = pd.read_parquet(os.path.join(data_dir, "raw", "test_data.parquet"))
    print("Raw test data before holdout: ", raw_df.shape[0])
    # Load raw inputs for display and align with holdout
    raw_df = raw_df.iloc[np.where(mask_holdout)[0]].reset_index(drop=True)
    raw_inputs = raw_df["values"].astype(str).tolist()
    print("Classes:", le.classes_)      # ['age', 'city', 'pre_existing_condition', ...]
    print("num_classes:", num_labels)  # e.g. 20
    print("After holdout test: ", X_test.shape[0])

    # Display inputs vs predictions
    print("\nSample-level Input vs Predicted:")
    for inp, pred, top_lbls, top_prs in zip(raw_inputs, y_pred_labels, y_top3_labels, y_top3_probs):
        print(f"INPUT: {inp}")
        print(f"Predicted: {pred}")
        print("Top-3 predictions:")
        for l, p in zip(top_lbls, top_prs):
            print(f"  {l}: {p:.4f}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Sherlock fine-tuned model with threshold and holdout and raw-input display")
    parser.add_argument("--data_dir", type=str, default="../custom_data")
    parser.add_argument("--model_id", type=str, default="sherlock")
    args = parser.parse_args()
    main(args.data_dir, args.model_id)
