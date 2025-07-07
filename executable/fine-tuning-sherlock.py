#!/usr/bin/env python3
"""
Script for loading data and initializing the Sherlock fine-tuned model.
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

SEED = 13
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_random_seed(SEED)

LABEL_MAP = {
    # Date
    "Vaccination_date": "date",
    "Date_report":"date",
    "Date_onset":  "date",
    "Date_confirmation": "date",
    "Date_of_first_consultation":"date",
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

    #Gender
    "Gender": "gender",
    "Sex_at_birth": "gender",
    "Gender_other": "gender",
    "Sex_at_birth_other": "gender",

    #Location
    "Travel_history_location": "location",
    "Location_information": "location",

    # Contact setting
    "Contact_setting": "contact_setting",
    "Contact_setting_other": "contact_setting",

    # demographic
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


    # Sourec
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

def main(data_dir, model_id):
    # Load training data
    X_train = pd.read_parquet(f"{data_dir}/processed/train.parquet")
    y_train = pd.read_parquet(f"{data_dir}/raw/train_labels.parquet").values.flatten()
    y_train = remap_labels(y_train)
    y_train = np.array([x.lower() for x in y_train])

    # Load validation data
    X_validation = pd.read_parquet(f"{data_dir}/processed/validation.parquet")
    y_validation = pd.read_parquet(f"{data_dir}/raw/validation_labels.parquet").values.flatten()
    y_validation = remap_labels(y_validation)
    y_validation = np.array([x.lower() for x in y_validation])

    # Load test data
    X_test = pd.read_parquet(f"{data_dir}/processed/test.parquet")
    y_test  = pd.read_parquet(f"{data_dir}/raw/test_labels.parquet").values.flatten()
    y_test = remap_labels(y_test)
    y_test = np.array([x.lower() for x in y_test])

    # Encode labels
    le = LabelEncoder().fit(y_train)
    # number of classes in train
    NUM_LABELS = len(le.classes_)

    # Initialize and load the fine-tuned model
    wrapper = SherlockModel()
    wrapper.initialize_model_from_json(with_weights=True, model_id=model_id)
    base_model = wrapper.model

    # Tune last dense layer fc
    penultimate = base_model.get_layer("dense_7").output

    # attach new classifier head
    logits = Dense(NUM_LABELS, activation="softmax", name = "classifier")(penultimate)

    finetune_model = Model(inputs = base_model.input, outputs = logits, name="sherlock_finetune")


    finetune_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",   # or focal loss if you’ve implemented it
        metrics=["accuracy"],
    )

    #finetune_model.summary()


    # process data
    train_inputs = make_inputs(X_train)
    val_inputs = make_inputs(X_validation)
    test_inputs = make_inputs(X_test)

    # keep only labels that exist in both train and test
    mask = np.isin(y_test, le.classes_)

    print("Kept rows:", mask.sum(), " / ", len(mask))

    test_inputs = [arr[mask] for arr in test_inputs]   # slice every branch

    # keep only rows whose label exists in the encoder
    y_test_int = le.transform(y_test[mask])

    # keep only existing rows for the validation data
    mask_valid = np.isin(y_validation, le.classes_)

    val_inputs = [arr[mask_valid] for arr in val_inputs]

    # encode string labels to integers
    y_train_int = le.transform(y_train)
    y_val_int = le.transform(y_validation[mask_valid])


    # assert to check whether data is float32
    for name, arr in zip(["char","word","par","rest"], train_inputs):
        assert arr.dtype == np.float32, f"{name} slice is {arr.dtype}"

    print("Classes:", le.classes_)      # ['age', 'city', 'pre_existing_condition', ...]
    print("num_classes:", NUM_LABELS)  # e.g. 20
    print("y_train_int shape:", y_train_int.shape)   # (195,)
    print("train_inputs[0].dtype:", train_inputs[0].dtype)  # float32

    # Fine-tuning
    train_start = time.perf_counter()
    finetune_model.fit(
        train_inputs, y_train_int,
        validation_data=(val_inputs, y_val_int),
        epochs=20,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )

    train_end = time.perf_counter()
    print(f"Fine-tuning time: {train_end - train_start:.2f}s")
    # saving weights
    finetune_model.save_weights("my_custom_sherlock_head.h5")
    
    new_id = "sherlock_fine_tuned"
    model_dir = "../model_files/"

    # save JSON
    model_json = finetune_model.to_json()
    with open(f"{model_dir}/{new_id}_model.json", "w") as f:
        f.write(model_json)

    # save weights under the same id
    finetune_model.save_weights(f"{model_dir}/{new_id}_weights.h5")


    # Testing the model!

    # load weights
    finetune_model.load_weights("my_custom_sherlock_head.h5")

    infer_start = time.perf_counter()

    # raw logits → predicted class indices
    y_pred_probs = finetune_model.predict(test_inputs, batch_size=256)
    y_pred_int   = y_pred_probs.argmax(axis=1)

    infer_end = time.perf_counter()
    print(f"Inference time: {infer_end - infer_start:.2f}s")

    # overall macro-F1
    weighted_f1 = f1_score(y_test_int, y_pred_int, average="weighted")
    macro_f1 = f1_score(y_test_int, y_pred_int, average="macro")

    print(f"Macro-F1   (unweighted) : {macro_f1:.4f}")
    print(f"Weighted F1 (support-avg): {weighted_f1:.4f}")

    present = np.unique(np.concatenate([y_test_int, y_pred_int]))
    present_names = le.inverse_transform(present)

    # per-class precision / recall / F1
    print(classification_report(
        y_test_int,
        y_pred_int,
        labels=present,
        target_names=present_names,  # human-readable names
        digits=3,
        zero_division = 0
    ))

    # optional confusion matrix
    cm = confusion_matrix(y_test_int, y_pred_int)
    print("Confusion matrix shape:", cm.shape)

    # inverse integers to string
    true_names = le.inverse_transform(y_test_int)
    pred_names = le.inverse_transform(y_pred_int)

    # 1) Print a few sample-level results
    print("\nSample-level True vs. Predicted:")
    for i, (t, p) in enumerate(zip(true_names, pred_names)):
        print(f"  [{i:4d}] True: {t:20s}  Pred: {p}")





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
