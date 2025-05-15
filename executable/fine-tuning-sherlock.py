#!/usr/bin/env python3
"""
Script for loading data and initializing the Sherlock fine-tuned model.
"""
import argparse
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

def num_labels():
    splits = {
        "train": Path("/content/sherlock-project/custom_data/raw/train_labels.parquet"),
        "val":   Path("/content/sherlock-project/custom_data/raw/validation_labels.parquet"),
        "test":  Path("/content/sherlock-project/custom_data/raw/test_labels.parquet"),
    }

    all_labels = set()
    for name, fp in splits.items():
        df = pd.read_parquet(fp)
        vc = df["type"].value_counts()
        print(f"\n== {name.upper()} ({len(df)} rows) ==")
        print(vc.sort_index())
        all_labels |= set(vc.index)

    #print(f"\nTOTAL UNIQUE LABELS ACROSS SPLITS: {len(all_labels)}")
    print(sorted(all_labels))

    #return len(all_labels)

def make_inputs(df):
    feature_cols = helpers.categorize_features()

    return [
        df[feature_cols["char"]].values.astype("float32"),
        df[feature_cols["word"]].values.astype("float32"),
        df[feature_cols["par"]].values.astype("float32"),
        df[feature_cols["rest"]].values.astype("float32"),
    ]

def main(data_dir, model_id):
    # Log start time
    start = datetime.now()
    print(f"Started at {start}")

    # Load training data
    X_train = pd.read_parquet(f"{data_dir}/processed/train.parquet")
    y_train = pd.read_parquet(f"{data_dir}/raw/train_labels.parquet").values.flatten()
    y_train = np.array([x.lower() for x in y_train])

    # Load validation data
    X_validation = pd.read_parquet(f"{data_dir}/processed/validation.parquet")
    y_validation = pd.read_parquet(f"{data_dir}/raw/validation_labels.parquet").values.flatten()
    y_validation = np.array([x.lower() for x in y_validation])

    # Load test data
    X_test = pd.read_parquet(f"{data_dir}/processed/test.parquet")
    y_test = pd.read_parquet(f"{data_dir}/raw/test_labels.parquet").values.flatten()
    y_test = np.array([x.lower() for x in y_test])

    X_test_synthetic = pd.read_parquet(f"{data_dir}/processed/test_synthetic.parquet")
    y_test_synthetic  = pd.read_parquet(f"{data_dir}/raw/test_synthetic_labels.parquet").values.flatten()
    y_test_synthetic = np.array([x.lower() for x in y_test_synthetic])

    # Initialize and load the fine-tuned model
    wrapper = SherlockModel()
    wrapper.initialize_model_from_json(with_weights=True, model_id=model_id)
    base_model = wrapper.model

    # Tune last dense layer fc
    penultimate = base_model.get_layer("dense_7").output

    # Encode labels
    le = LabelEncoder().fit(y_train)
    # number of classes in train
    NUM_LABELS = len(le.classes_)

    # attach new classifier head
    logits = Dense(NUM_LABELS, activation="softmax", name = "classifier")(penultimate)

    finetune_model = Model(inputs = base_model.input, outputs = logits, name="sherlock_finetune")

    # freeze all original layers
    for layer in finetune_model.layers:
        layer.trainable = layer.name in {"classifier"}


    finetune_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",   # or focal loss if you’ve implemented it
        metrics=["accuracy"],
    )

    #finetune_model.summary()


    # process data
    train_inputs = make_inputs(X_train)
    val_inputs = make_inputs(X_validation)
    # original data
    #test_inputs = make_inputs(X_test)
    # synthetic test data
    test_inputs = make_inputs(X_test_synthetic)

    # keep only labels that exist in both train and test
    #original data
    #mask = np.isin(y_test, le.classes_)     # boolean mask the same length as y_test
    #synthetic data
    mask = np.isin(y_test_synthetic, le.classes_)

    print("Kept rows:", mask.sum(), " / ", len(mask))

    test_inputs = [arr[mask] for arr in test_inputs]   # slice every branch

    # keep only rows whose label exists in the encoder
    # original data
    #y_test_int = le.transform(y_test[mask])
    # synthetic data
    y_test_int = le.transform(y_test_synthetic[mask])

    # encode string labels to integers
    y_train_int = le.transform(y_train)
    y_val_int = le.transform(y_validation)


    # assert to check whether data is float32
    for name, arr in zip(["char","word","par","rest"], train_inputs):
        assert arr.dtype == np.float32, f"{name} slice is {arr.dtype}"

    # TODO: Solve problem with num_classes since now it's different
    print("Classes:", le.classes_)      # ['age', 'city', 'pre_existing_condition', ...]
    print("num_classes:", NUM_LABELS)  # e.g. 20
    print("y_train_int shape:", y_train_int.shape)   # (195,)
    print("train_inputs[0].dtype:", train_inputs[0].dtype)  # float32

    # Training

    finetune_model.fit(
        train_inputs, y_train_int,
        validation_data=(val_inputs, y_val_int),
        epochs=80,
        #class_weight=class_weights_dict,   # optional, for imbalance
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)],
    )

    # saving weights
    finetune_model.save_weights("my_custom_sherlock_head.h5")


    # Testing the model!

    # load weights
    finetune_model.load_weights("my_custom_sherlock_head.h5")

    # raw logits → predicted class indices
    y_pred_probs = finetune_model.predict(test_inputs, batch_size=256)
    y_pred_int   = y_pred_probs.argmax(axis=1)

    # overall macro-F1
    macro_f1 = f1_score(y_test_int, y_pred_int, average="macro")
    print(f"Macro-F1: {macro_f1:.4f}")

    all_ids = np.arange(NUM_LABELS)

    # per-class precision / recall / F1
    print(classification_report(
        y_test_int,
        y_pred_int,
        labels=all_ids,
        target_names=le.classes_,  # human-readable names
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
