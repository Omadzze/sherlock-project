#!/usr/bin/env python3
"""
Script for loading data, initializing the Sherlock fine-tuned model,
and displaying raw inputs without applying a confidence threshold.
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

# Classes to hold out (never seen during training/validation/testing)
#holdout_classes = [
#    "contact_setting",
#    "occupation",
#    "symptoms",
#]

UNKNOWN_LABEL = "unknown"

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
MODEL_ID = "sherlock"


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


def save_metrics(name: str, raw_report, y_true, run_id: int, inference_time, out_dir):

    run_name = f"{name}-run-{run_id}"

    n = len(y_true)

    # 2. flatten into a single dict
    flat = {}
    for label, metrics in raw_report.items():
        if label == "accuracy":
            flat["accuracy"] = metrics
        else:
            for metric_name, val in metrics.items():
                # replace any dashes so your CSV columns are valid identifiers
                clean_metric = metric_name.replace("-", "_")
                flat[f"{label}_{clean_metric}"] = val


    flat["total_entries"] = n
    flat["run_name"]      = run_name
    flat["inference_time"] = f"{inference_time:.2f}s"


    df = pd.DataFrame([flat])
    metrics_csv = os.path.join(out_dir, f"{name}-metrics.csv")

    # only write header if file doesn’t exist
    if not os.path.isfile(metrics_csv):
        df.to_csv(metrics_csv, index=False, float_format="%.4f")
    else:
        df.to_csv(metrics_csv, mode="a", header=False, index=False, float_format="%.4f")



def main(data_dir, holdout_classes, run_id, out_dir):
    # --- Load and label training data ---
    X_train = pd.read_parquet(f"{data_dir}/processed/train.parquet")
    y_train = pd.read_parquet(f"{data_dir}/raw/train_labels.parquet").values.flatten()
    y_train = remap_labels(y_train)
    y_train = np.array([lbl.lower() for lbl in y_train])
    # mark holdout classes as unknown
    y_train = np.array([
        UNKNOWN_LABEL if lbl in holdout_classes else lbl
        for lbl in y_train
    ])

    # --- Load and label validation data ---
    X_val = pd.read_parquet(f"{data_dir}/processed/validation.parquet")
    y_val = pd.read_parquet(f"{data_dir}/raw/validation_labels.parquet").values.flatten()
    y_val = remap_labels(y_val)
    y_val = np.array([lbl.lower() for lbl in y_val])
    y_val = np.array([
        UNKNOWN_LABEL if lbl in holdout_classes else lbl
        for lbl in y_val
    ])

    # --- Load and label test data ---
    X_test = pd.read_parquet(f"{data_dir}/processed/test.parquet")
    y_test_original = pd.read_parquet(f"{data_dir}/raw/test_labels.parquet").values.flatten()
    y_test = remap_labels(y_test_original)
    y_test = np.array([lbl.lower() for lbl in y_test])
    y_test = np.array([
        UNKNOWN_LABEL if lbl in holdout_classes else lbl
        for lbl in y_test
    ])

    # Encode labels (including 'unknown')
    le = LabelEncoder().fit(y_train)
    class_names = le.classes_
    num_labels = len(class_names)

    # Initialize base model
    wrapper = SherlockModel()
    wrapper.initialize_model_from_json(with_weights=True, model_id=MODEL_ID)
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

    # Prepare integer labels
    y_train_int = le.transform(y_train)
    y_val_int   = le.transform(y_val)
    y_test_int  = le.transform(y_test)

    # Prepare inputs
    train_inputs = make_inputs(X_train)
    val_inputs   = make_inputs(X_val)
    test_inputs  = make_inputs(X_test)

    # Fine-tuning
    start = time.perf_counter()
    finetune_model.fit(
        train_inputs, y_train_int,
        validation_data=(val_inputs, y_val_int),
        epochs=20,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )
    print(f"Fine-tuning completed in {time.perf_counter() - start:.2f}s")

    print("Classes:", le.classes_)      # ['age', 'city', 'pre_existing_condition', ...]
    print("num_classes:", num_labels)  # e.g. 20
    print("y_train_int shape:", y_train_int.shape)   # (195,)
    print("train_inputs[0].dtype:", train_inputs[0].dtype)  # float32

    # Save model artifacts
    finetune_model.save_weights("my_custom_sherlock_head.h5")
    model_dir = "../model_files/"
    with open(f"{model_dir}/{MODEL_ID}_model.json", "w") as f:
        f.write(finetune_model.to_json())
    finetune_model.save_weights(f"{model_dir}/{MODEL_ID}_weights.h5")

    # Inference
    start_inf = time.perf_counter()
    y_pred_probs = finetune_model.predict(test_inputs, batch_size=256)
    end_inf = time.perf_counter()

    inference_time = end_inf-start_inf

    print(f"Inference completed in {inference_time:.2f}s")

    # Predicted labels based on highest probability
    y_pred_idx    = np.argmax(y_pred_probs, axis=1)
    y_pred_labels = [class_names[i] for i in y_pred_idx]

    # Evaluation
    print((classification_report(
        y_test_int, y_pred_idx,
        target_names=class_names,
        digits=3, zero_division=0)))

    raw_report = classification_report(
        y_test_int, y_pred_idx,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    save_metrics("Sherlock", raw_report, y_test, run_id,  inference_time, out_dir)

    print("Confusion matrix shape:", confusion_matrix(y_test_int, y_pred_idx).shape)

    # Display raw test inputs vs predictions
    raw_df     = pd.read_parquet(os.path.join(data_dir, "raw", "test_data.parquet"))
    raw_inputs = raw_df["values"].astype(str).tolist()
    print("\nSample-level Input vs Predicted:")
    for inp, pred in zip(raw_inputs, y_pred_labels[:len(raw_inputs)]):
        print(f"INPUT: {inp}")
        print(f"Predicted: {pred}\n")

    save_unknown_parquets(raw_df, y_test, y_test_original,  "../custom_data", "../custom_data/label_generation")



def save_unknown_parquets(raw_df: pd.DataFrame,
                          masked_labels: np.ndarray,
                          original_labels: np.ndarray,
                          data_dir: str,
                          temp_dir: str = None):
    """
    Extracts all examples whose masked_labels == UNKNOWN_LABEL, and writes out
    two parquet files in the same format as 'intest_data.parquet' and
    'train_labels.parquet', preserving the original row index as "__index_level_0__".

    Parameters
    ----------
    raw_df : pd.DataFrame
        The raw data DataFrame (must contain a 'values' column), indexed by the original row IDs.
    masked_labels : np.ndarray
        1-D array of labels after marking holdouts as UNKNOWN_LABEL.
    original_labels : np.ndarray
        1-D array of the true labels before masking, aligned with raw_df.
    data_dir : str
        Base data directory (used only to locate default temp dir).
    temp_dir : str, optional
        Directory to write the parquet files. If None, defaults to os.path.join(data_dir, 'temp').
    """
    if temp_dir is None:
        temp_dir = os.path.join(data_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # Boolean mask for the rows to extract
    mask_unknown = (masked_labels == UNKNOWN_LABEL)

    # --- Prepare and save intest_data.parquet ---
    df_data = raw_df.loc[mask_unknown, ['values']].copy()
    # reset_index to turn the original index into a column
    df_data = df_data.reset_index().rename(columns={'index': '__index_level_0__'})
    test_generation_path = os.path.join(temp_dir, 'test_data_generation.parquet')
    df_data.to_parquet(test_generation_path, index=False)

    # --- Prepare and save train_labels.parquet ---
    # Build a Series of original labels indexed by the same index
    orig_series = pd.Series(original_labels, index=raw_df.index, name='type')
    df_labels = orig_series.loc[mask_unknown].reset_index().rename(
        columns={'index': '__index_level_0__', 'type': 'type'}
    )
    labels_generation_path = os.path.join(temp_dir, 'test_labels_generation.parquet')
    df_labels.to_parquet(labels_generation_path, index=False)

    print(f"Saved {mask_unknown.sum()} unknown examples:")
    print(f"  - data -> {test_generation_path}")
    print(f"  - labels -> {labels_generation_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Sherlock fine-tuned model for semantic label generation task")
    parser.add_argument("--data_dir", type=str, default="../custom_data", help="Data directory, where all your data is located, processed and raw.")
    parser.add_argument("--holdout_classes", nargs="*", default=[],
                   help="list of labels to mask as 'unknown'. E.g [symptoms, location]")
    parser.add_argument("--run_id", type=int, default=1,)
    parser.add_argument("--out_dir", type=str, default="")

    args = parser.parse_args()
    main(args.data_dir, args.holdout_classes, args.run_id, args.out_dir)