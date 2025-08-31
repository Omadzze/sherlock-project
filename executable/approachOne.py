#!/usr/bin/env python3
"""
Script for loading data, initializing the Sherlock fine-tuned model,
OPTIONALLY running K-fold cross-validation on train+val (80%),
and displaying raw inputs without applying a confidence threshold.

Toggle cross-validation with the USE_CV constant below (or pass via env var).

- USE_CV=False → original single train/val/test path (unchanged behavior)
- USE_CV=True  → 5-fold Stratified CV on 80% (train+val), then retrain on 80% and evaluate once on frozen 20% test
"""
import argparse
import os
import time
import random
from collections import Counter

import numpy as np
import pandas as pd
from docutils.core import default_description
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from sherlock.deploy.model import LabelEncoder, SherlockModel
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from sherlock.deploy import helpers

from tensorflow.keras.callbacks import CSVLogger


import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
# ---------------------------
# Configuration / Toggles
# ---------------------------

SHERLOCK_MODE = "fine-tune"  # "train-scratch" or "fine-tune"
MODEL_ID = os.environ.get("MODEL_ID", "sherlock")
UNKNOWN_LABEL = "unknown"
SEED = 13

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
# TF2-compatible seeding
try:
    tf.random.set_seed(SEED)
except Exception:
    # fallback for older TF
    try:
        tf.random.set_random_seed(SEED)
    except Exception:
        pass

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


# ---------------------------
# Helpers
# ---------------------------

def remap_labels(arr, mapping=LABEL_MAP_LC):
    return np.array([mapping.get(x.lower(), x.lower()) for x in arr])


def make_inputs(df: pd.DataFrame):
    feature_cols = helpers.categorize_features()
    return [
        df[feature_cols["char"]].values.astype("float32"),
        df[feature_cols["word"]].values.astype("float32"),
        df[feature_cols["par"]].values.astype("float32"),
        df[feature_cols["rest"]].values.astype("float32"),
    ]


def save_metrics(name: str, raw_report, y_true, run_id: int, inference_time, out_dir: str):
    run_name = f"{name}-run-{run_id}"
    n = len(y_true)

    # flatten into a single row
    flat = {}
    for label, metrics in raw_report.items():
        if label == "accuracy":
            flat["accuracy"] = metrics
        else:
            for metric_name, val in metrics.items():
                clean_metric = metric_name.replace("-", "_")
                flat[f"{label}_{clean_metric}"] = val

    flat["total_entries"] = n
    flat["run_name"] = run_name
    flat["inference_time"] = f"{inference_time:.2f}s"

    df = pd.DataFrame([flat])
    metrics_csv = os.path.join(out_dir, f"{name}-metrics.csv") if out_dir else f"{name}-metrics.csv"

    if not os.path.isfile(metrics_csv):
        df.to_csv(metrics_csv, index=False, float_format="%.4f")
    else:
        df.to_csv(metrics_csv, mode="a", header=False, index=False, float_format="%.4f")


# ---------------------------
# CV utilities (fine-tune path)
# ---------------------------

def build_finetune_model(num_classes: int):
    """Create a fresh Sherlock base + softmax head, compiled."""
    w = SherlockModel()
    if SHERLOCK_MODE == "train-scratch":
        raise ValueError("build_finetune_model called in train-scratch mode. Use training_scratch().")
    else:
        w.initialize_model_from_json(with_weights=True, model_id=MODEL_ID)
    base = w.model
    penultimate = base.get_layer("dense_7").output
    logits = Dense(num_classes, activation="softmax", name="classifier")(penultimate)
    m = Model(inputs=base.input, outputs=logits)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


def run_finetune_cv_then_test(X_train: pd.DataFrame, y_train: np.ndarray,
                              X_val: pd.DataFrame, y_val: np.ndarray,
                              X_test: pd.DataFrame, y_test: np.ndarray,
                              run_id: int, out_dir: str):
    """K-fold CV on train+val (80%), then retrain on full 80% and evaluate on frozen 20% test."""
    # 1) Concatenate train+val
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = np.concatenate([y_train, y_val])

    # Encode labels ON trainval (ensures 'unknown' present if used)
    le_cv = LabelEncoder().fit(y_trainval)
    y_trainval_int = le_cv.transform(y_trainval)
    y_test_int = le_cv.transform(y_test)  # will fail only if test has unseen label
    num_classes = len(le_cv.classes_)

    # 2) CV loop
    K = 5
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    cv_macro, cv_weighted, cv_acc, fit_times, infer_times = [], [], [], [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_trainval, y_trainval_int), 1):
        X_tr, X_va = X_trainval.iloc[tr_idx], X_trainval.iloc[va_idx]
        y_tr, y_va = y_trainval_int[tr_idx], y_trainval_int[va_idx]

        tr_inputs = make_inputs(X_tr)
        va_inputs = make_inputs(X_va)

        model = build_finetune_model(num_classes)

        t0 = time.perf_counter()
        model.fit(
            tr_inputs, y_tr,
            validation_data=(va_inputs, y_va),
            epochs=100, batch_size=256, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")],
        )
        ft = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_va_pred = model.predict(va_inputs, batch_size=256, verbose=0).argmax(axis=1)
        it = time.perf_counter() - t1

        cv_macro.append(f1_score(y_va, y_va_pred, average="macro", zero_division=0))
        cv_weighted.append(f1_score(y_va, y_va_pred, average="weighted", zero_division=0))
        cv_acc.append(accuracy_score(y_va, y_va_pred))
        fit_times.append(ft); infer_times.append(it)

        print(f"[Fold {fold}/{K}] macroF1={cv_macro[-1]:.3f} | wF1={cv_weighted[-1]:.3f} | acc={cv_acc[-1]:.3f} "
              f"| fit {ft:.2f}s | infer {it:.2f}s")

    def summarize(a):
        a = np.asarray(a, dtype=float)
        m, s = a.mean(), a.std(ddof=1)
        return f"{m:.3f} ± {s:.3f}"

    print("\n[Sherlock Fine-tune | 5-fold CV on 80%]")
    print("macro-F1    :", summarize(cv_macro))
    print("weighted-F1 :", summarize(cv_weighted))
    print("accuracy    :", summarize(cv_acc))
    print("fit_time(s) :", summarize(fit_times), "| infer_time(s):", summarize(infer_times))

    # 3) Final training on full 80% + evaluation on frozen 20% test
    trainval_inputs = make_inputs(X_trainval)
    test_inputs = make_inputs(X_test)

    final_model = build_finetune_model(num_classes)

    start = time.perf_counter()
    final_model.fit(
        trainval_inputs, y_trainval_int,
        validation_split=0.1,  # small internal val for ES; remove if undesired
        epochs=100, batch_size=256, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")],
    )
    print(f"Fine-tuning (full 80%) completed in {time.perf_counter() - start:.2f}s")

    t_inf = time.perf_counter()
    y_pred_probs = final_model.predict(test_inputs, batch_size=256, verbose=0)
    inference_time = time.perf_counter() - t_inf
    y_pred_idx = y_pred_probs.argmax(axis=1)

    print(f"[Sherlock | Holdout 20% test] Inference: {inference_time:.2f}s for {len(y_test_int)} rows")
    print(classification_report(y_test_int, y_pred_idx, target_names=le_cv.classes_, digits=3, zero_division=0))

    raw_report = classification_report(
        y_test_int, y_pred_idx, target_names=le_cv.classes_, output_dict=True, zero_division=0
    )
    save_metrics("Sherlock", raw_report, y_test, run_id, inference_time, out_dir)

    # Optional: save final artifacts
    final_model.save_weights("my_custom_sherlock_head.h5")
    model_dir = "../model_files/"
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, f"{MODEL_ID}_model.json"), "w") as f:
        f.write(final_model.to_json())
    final_model.save_weights(os.path.join(model_dir, f"{MODEL_ID}_weights.h5"))


# ---------------------------
# Training from scratch (unchanged)
# ---------------------------

def save_unknown_parquets(raw_df: pd.DataFrame,
                          masked_labels: np.ndarray,
                          original_labels: np.ndarray,
                          data_dir: str,
                          temp_dir: str = None):
    if temp_dir is None:
        temp_dir = os.path.join(data_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    mask_unknown = (masked_labels == UNKNOWN_LABEL)

    df_data = raw_df.loc[mask_unknown, ['values']].copy()
    df_data = df_data.reset_index().rename(columns={'index': '__index_level_0__'})
    test_generation_path = os.path.join(temp_dir, 'test_data_generation.parquet')
    df_data.to_parquet(test_generation_path, index=False)

    orig_series = pd.Series(original_labels, index=raw_df.index, name='type')
    df_labels = orig_series.loc[mask_unknown].reset_index().rename(
        columns={'index': '__index_level_0__', 'type': 'type'}
    )
    labels_generation_path = os.path.join(temp_dir, 'test_labels_generation.parquet')
    df_labels.to_parquet(labels_generation_path, index=False)

    print(f"Saved {mask_unknown.sum()} unknown examples:")
    print(f"  - data -> {test_generation_path}")
    print(f"  - labels -> {labels_generation_path}")



def has_val_only_labels(y_tr, y_va):
    return bool(set(y_va) - set(y_tr))

def print_support(tag, y):
    c = Counter(y); print(tag, dict(sorted(c.items(), key=lambda kv: kv[0])))


#def save_results_csv(fold, curves_dir):
#    os.makedirs(curves_dir, exist_ok=True)
#    csv_path = os.path.join(curves_dir, f"cv_fold_{fold}_log.csv")
#    logger = CSVLogger(csv_path, append=False)
#    return logger, csv_path


def run_scratch_cv_then_test(X_train: pd.DataFrame, y_train: np.ndarray,
                             X_val: pd.DataFrame, y_val: np.ndarray,
                             X_test: pd.DataFrame, y_test: np.ndarray,
                             run_id: int, out_dir: str, data_dir: str):
    """K-fold CV for the wrapper.fit(...) scratch flow, then final train+test.
       Uses fresh SherlockModel per fold and string labels (no encoding).
    """

    # 1) Concatenate train+val for CV
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = np.concatenate([y_train, y_val])

    # 2) CV loop on strings
    K = 5
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    cv_macro, cv_weighted, cv_acc, fit_times, infer_times = [], [], [], [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_trainval, y_trainval), 1):
        X_tr, X_va = X_trainval.iloc[tr_idx], X_trainval.iloc[va_idx]
        y_tr, y_va = y_trainval[tr_idx], y_trainval[va_idx]

        # --- Coverage check ---
        if has_val_only_labels(y_tr, y_va):
            print(f"[Fold {fold}] Skipping: val-only labels detected:",
                  set(y_va) - set(y_tr))
            continue  # or replace with a reshuffle logic if you want K folds fixed

        print_support(f"Fold {fold} Train", y_tr)
        print_support(f"Fold {fold} Val", y_va)

        wrapper = SherlockModel()
        wrapper.initialize_model_from_json(with_weights=True, model_id=MODEL_ID)

        #curves_dir = os.path.join(out_dir, "cv_curves")
        #callbacks, csv_path = save_results_csv(fold, curves_dir)

        t0 = time.perf_counter()
        # wrapper.fit expects (X_train, y_train, X_val, y_val)
        wrapper.fit(X_tr, y_tr, X_va, y_va, model_id=f"retrained_sherlock_fold{fold}", fold=fold)
        ft = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_va_pred = wrapper.predict(X_va, model_id=f"retrained_sherlock_fold{fold}")
        y_va_pred = np.array([x.lower() for x in y_va_pred])
        it = time.perf_counter() - t1

        cv_macro.append(f1_score(y_va, y_va_pred, average="macro", zero_division=0))
        cv_weighted.append(f1_score(y_va, y_va_pred, average="weighted", zero_division=0))
        cv_acc.append(accuracy_score(y_va, y_va_pred))
        fit_times.append(ft); infer_times.append(it)

        print(f"[Scratch Fold {fold}/{K}] macroF1={cv_macro[-1]:.3f} | wF1={cv_weighted[-1]:.3f} | acc={cv_acc[-1]:.3f} "
              f"| fit {ft:.2f}s | infer {it:.2f}s")

        for_fold_csv = os.path.join("/home/omadbek/projects/Sherlock/executable", f"cv_fold_{fold}.csv")
        plot_from_csv(for_fold_csv, fold)

    def summarize(a):
        a = np.asarray(a, dtype=float)
        m, s = a.mean(), a.std(ddof=1)
        return f"{m:.3f} ± {s:.3f}"

    print("\n[Sherlock Scratch | 5-fold CV on 80%]")
    print("macro-F1    :", summarize(cv_macro))
    print("weighted-F1 :", summarize(cv_weighted))
    print("accuracy    :", summarize(cv_acc))
    print("fit_time(s) :", summarize(fit_times), "| infer_time(s):", summarize(infer_times))

    # 3) Final training on full 80% (make a tiny internal val for ES) + test
    # Create a small stratified split from trainval for validation (10%)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    tr_idx, va_idx = next(sss.split(X_trainval, y_trainval))
    X_tr, X_va = X_trainval.iloc[tr_idx], X_trainval.iloc[va_idx]
    y_tr, y_va = y_trainval[tr_idx], y_trainval[va_idx]

    wrapper_final = SherlockModel()
    wrapper_final.initialize_model_from_json(with_weights=True, model_id=MODEL_ID)

    print("Training scratch wrapper on full 80% (with 10% internal val)...")
    wrapper_final.fit(X_tr, y_tr, X_va, y_va, model_id="retrained_sherlock_full80", fold=11)

    start_inf = time.perf_counter()
    predicted_labels = wrapper_final.predict(X_test, model_id="retrained_sherlock_full80")
    end_inf = time.perf_counter()
    inference_time = end_inf - start_inf

    predicted_labels = np.array([x.lower() for x in predicted_labels])
    print(f"[Scratch | Holdout 20% test] Inference completed in {inference_time:.2f}s")

    # Report & save
    print(classification_report(y_test, predicted_labels, digits=3))
    raw_report = classification_report(y_test, predicted_labels, output_dict=True, zero_division=0)
    save_metrics("Sherlock-scratch", raw_report, y_test, run_id, inference_time, out_dir)

    print("Confusion matrix shape:", confusion_matrix(y_test, predicted_labels).shape)

    raw_df = pd.read_parquet(os.path.join(data_dir, "raw", "test_data.parquet"))
    raw_inputs = raw_df["values"].astype(str).tolist()
    print("\nSample-level Input vs Predicted:")
    for inp, pred in zip(raw_inputs, predicted_labels[:len(raw_inputs)]):
        print(f"INPUT: {inp}")
        print(f"Predicted: {pred}\n")


def plot_from_csv(csv_path, fold):
    df = pd.read_csv(csv_path)  # columns: epoch, loss, categorical_accuracy, val_loss, val_categorical_accuracy, ...
    epochs = np.arange(1, len(df) + 1)

    # Loss
    if {"loss","val_loss"}.issubset(df.columns):
        fig = plt.figure()
        plt.plot(epochs, df["loss"], label="loss")
        plt.plot(epochs, df["val_loss"], label="val_loss")
        vi = int(df["val_loss"].idxmin())
        plt.scatter(epochs[vi], df["val_loss"].iloc[vi], s=50)
        plt.title(f"Fold {fold} — Loss (best val @ epoch {epochs[vi]})")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True); plt.legend()
        fig.savefig(os.path.join("/home/omadbek/projects/Sherlock/executable", f"cv_fold{fold}_loss.png"),
                    bbox_inches="tight")
        plt.close(fig)

    # Accuracy-like metric
    if {"categorical_accuracy","val_categorical_accuracy"}.issubset(df.columns):
        fig = plt.figure()
        plt.plot(epochs, df["categorical_accuracy"], label="categorical_accuracy")
        plt.plot(epochs, df["val_categorical_accuracy"], label="val_categorical_accuracy")
        vi = int(df["val_categorical_accuracy"].idxmax())
        plt.scatter(epochs[vi], df["val_categorical_accuracy"].iloc[vi], s=50)
        plt.title(f"Fold {fold} — Categorical Accuracy (best val @ epoch {epochs[vi]})")
        plt.xlabel("Epoch"); plt.ylabel("categorical_accuracy"); plt.grid(True); plt.legend()
        fig.savefig(os.path.join("/home/omadbek/projects/Sherlock/executable", f"cv_fold{fold}_categorical_accuracy.png"),
                    bbox_inches="tight")
        plt.close(fig)

def training_scratch(X_train, y_train, X_val, y_val, X_test, y_test, run_id, out_dir, data_dir):
    wrapper = SherlockModel()
    wrapper.initialize_model_from_json(with_weights=True, model_id="sherlock")

    wrapper.fit(X_train, y_train, X_val, y_val, model_id="retrained_sherlock")

    print("Trained and saved new model")
    wrapper.store_weights(model_id="retrained_sherlock")

    start_inf = time.perf_counter()
    predicted_labels = wrapper.predict(X_test, model_id="retrained_sherlock")
    end_inf = time.perf_counter()

    inference_time = end_inf - start_inf
    predicted_labels = np.array([x.lower() for x in predicted_labels])

    print(f"Inference completed in {inference_time:.2f}s")
    size = len(y_test)
    _ = f1_score(y_test[:size], predicted_labels[:size], average="weighted")

    classes = np.load(f"../model_files/classes_retrained_sherlock.npy", allow_pickle=True)

    report = classification_report(y_test, predicted_labels, output_dict=True)

    class_scores = list(
        filter(lambda x: isinstance(x, tuple) and isinstance(x[1], dict) and 'f1-score' in x[1] and x[0] in classes,
               list(report.items())))

    class_scores = sorted(class_scores, key=lambda item: item[1]['f1-score'], reverse=True)

    print(f"\t\tf1-score\tprecision\trecall\t\tsupport")
    for key, value in class_scores[0:5]:
        tabs = '\t' * (1 if len(key) >= 8 else 2)
        print(f"{key}{tabs}{value['f1-score']:.3f}\t\t{value['precision']:.3f}\t\t{value['recall']:.3f}\t\t{value['support']}")

    print(f"\t\tf1-score\tprecision\trecall\t\tsupport")
    for key, value in class_scores[-5:]:
        tabs = '\t' * (1 if len(key) >= 8 else 2)
        print(f"{key}{tabs}{value['f1-score']:.3f}\t\t{value['precision']:.3f}\t\t{value['recall']:.3f}\t\t{value['support']}")

    print(classification_report(y_test, predicted_labels, digits=3))

    raw_report = classification_report(y_test, predicted_labels, output_dict=True, zero_division=0)
    save_metrics("Sherlock-scratch-training", raw_report, y_test, run_id, inference_time, out_dir)

    print("Confusion matrix shape:", confusion_matrix(y_test, predicted_labels).shape)

    raw_df = pd.read_parquet(os.path.join(data_dir, "raw", "test_data.parquet"))
    raw_inputs = raw_df["values"].astype(str).tolist()
    print("\nSample-level Input vs Predicted:")
    for inp, pred in zip(raw_inputs, predicted_labels[:len(raw_inputs)]):
        print(f"INPUT: {inp}")
        print(f"Predicted: {pred}\n")

    size = len(y_test)
    mismatches = []
    for idx, k1 in enumerate(y_test[:size]):
        k2 = predicted_labels[idx]
        if k1 != k2:
            mismatches.append(k1)
            if k1 in ('address'):
                print(f'[{idx}] expected "{k1}" but predicted "{k2}"')

    f1 = f1_score(y_test[:size], predicted_labels[:size], average="weighted")
    print(f'Total mismatches: {len(mismatches)} (F1 score: {f1})')

    data = Counter(mismatches)
    _ = data.most_common()



# ---------------------------
# Main
# ---------------------------

def main(data_dir, holdout_classes, run_id, out_dir, USE_CV):
    # --- Load and label training data ---
    X_train = pd.read_parquet(f"{data_dir}/processed/train.parquet")
    y_train = pd.read_parquet(f"{data_dir}/raw/train_labels.parquet").values.flatten()
    y_train = remap_labels(y_train)
    y_train = np.array([lbl.lower() for lbl in y_train])
    y_train = np.array([UNKNOWN_LABEL if lbl in holdout_classes else lbl for lbl in y_train])

    # --- Load and label validation data ---
    X_val = pd.read_parquet(f"{data_dir}/processed/validation.parquet")
    y_val = pd.read_parquet(f"{data_dir}/raw/validation_labels.parquet").values.flatten()
    y_val = remap_labels(y_val)
    y_val = np.array([lbl.lower() for lbl in y_val])
    y_val = np.array([UNKNOWN_LABEL if lbl in holdout_classes else lbl for lbl in y_val])

    # --- Load and label test data ---
    X_test = pd.read_parquet(f"{data_dir}/processed/test.parquet")
    y_test_original = pd.read_parquet(f"{data_dir}/raw/test_labels.parquet").values.flatten()
    y_test = remap_labels(y_test_original)
    y_test = np.array([lbl.lower() for lbl in y_test])
    y_test = np.array([UNKNOWN_LABEL if lbl in holdout_classes else lbl for lbl in y_test])

    if SHERLOCK_MODE == "train-scratch":
        if USE_CV:
            run_scratch_cv_then_test(X_train, y_train, X_val, y_val, X_test, y_test, run_id, out_dir, data_dir)
        else:
            training_scratch(X_train, y_train, X_val, y_val, X_test, y_test, run_id, out_dir, data_dir)
        return

    # ---------------- Fine-tune path ----------------
    if USE_CV:
        run_finetune_cv_then_test(X_train, y_train, X_val, y_val, X_test, y_test, run_id, out_dir)
    else:
        # Original single-run fine-tune path (unchanged)
        # Encode labels on TRAIN ONLY (matches your original behavior)
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
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Prepare integer labels
        y_train_int = le.transform(y_train)
        y_val_int = le.transform(y_val)
        y_test_int = le.transform(y_test)

        # Prepare inputs
        train_inputs = make_inputs(X_train)
        val_inputs = make_inputs(X_val)
        test_inputs = make_inputs(X_test)

        # Fine-tuning
        start = time.perf_counter()
        finetune_model.fit(
            train_inputs, y_train_int,
            validation_data=(val_inputs, y_val_int),
            epochs=100,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")],
        )
        print(f"Fine-tuning completed in {time.perf_counter() - start:.2f}s")

        print("Classes:", class_names)
        print("num_classes:", num_labels)
        print("y_train_int shape:", y_train_int.shape)
        print("train_inputs[0].dtype:", train_inputs[0].dtype)

        # Save model artifacts
        finetune_model.save_weights("my_custom_sherlock_head.h5")
        model_dir = "../model_files/"
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, f"{MODEL_ID}_model.json"), "w") as f:
            f.write(finetune_model.to_json())
        finetune_model.save_weights(os.path.join(model_dir, f"{MODEL_ID}_weights.h5"))

        # Inference
        start_inf = time.perf_counter()
        y_pred_probs = finetune_model.predict(test_inputs, batch_size=256)
        end_inf = time.perf_counter()
        inference_time = end_inf - start_inf
        print(f"Inference completed in {inference_time:.2f}s")

        # Predicted labels
        y_pred_idx = np.argmax(y_pred_probs, axis=1)
        y_test_int = le.transform(y_test)

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
        save_metrics("Sherlock", raw_report, y_test, run_id, inference_time, out_dir)

        print("Confusion matrix shape:", confusion_matrix(y_test_int, y_pred_idx).shape)

        # Display raw test inputs vs predictions
        raw_df = pd.read_parquet(os.path.join(data_dir, "raw", "test_data.parquet"))
        raw_inputs = raw_df["values"].astype(str).tolist()
        class_names_list = list(class_names)
        print("\nSample-level Input vs Predicted:")
        for inp, pred in zip(raw_inputs, [class_names_list[i] for i in y_pred_idx][:len(raw_inputs)]):
            print(f"INPUT: {inp}")
            print(f"Predicted: {pred}\n")

        save_unknown_parquets(raw_df, y_test, y_test_original, "../custom_data", "../custom_data/label_generation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Sherlock fine-tuned model for semantic label generation task")
    parser.add_argument("--data_dir", type=str, default="../custom_data",
                        help="Data directory, where all your data is located, processed and raw.")
    parser.add_argument("--holdout_classes", nargs="*", default=[],
                        help="list of labels to mask as 'unknown'. E.g [symptoms, location]")
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--cv", type=bool, default=False)

    args = parser.parse_args()
    main(args.data_dir, args.holdout_classes, args.run_id, args.out_dir, args.cv)
