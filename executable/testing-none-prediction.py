import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tensorflow.keras.models import model_from_json
from sherlock.deploy import helpers
import random
import tensorflow as tf

# Reproducibility
SEED = 13
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Define the full set of class names for inference
class_names = [
    'age', 'case_status', 'contact_setting', 'date', 'gender', 'id',
    'location', 'medical_boolean', 'occupation', 'outcome', 'symptoms'
]
NUM_CLASSES = len(class_names)

# Mapping utility for raw labels
LABEL_MAP = {
    # Date
    "Vaccination_date": "date", "Date_report": "date", "Date_onset": "date",
    "Date_confirmation": "date", "Date_of_first_consultation": "date",
    "Date_hospitalisation": "date", "Date_discharge_hospital": "date",
    "Date_admission_ICU": "date", "Date_discharge_ICU": "date",
    "Date_isolation": "date", "Date_death": "date", "Date_recovered": "date",
    "Travel_history_entry": "date", "Travel_history_start": "date",
    "Date_entry": "date", "Date_last_modified": "date",
    # ID
    "Contact_ID": "id", "ID": "id",
    # Gender
    "Gender": "gender", "Sex_at_birth": "gender",
    "Gender_other": "gender", "Sex_at_birth_other": "gender",
    # Location
    "Travel_history_location": "location", "Location_information": "location",
    # Contact setting
    "Contact_setting": "contact_setting", "Contact_setting_other": "contact_setting",
    # Demographic
    "Race": "demographic", "Ehtnicity": "demographic",
    # Medical Boolean
    "Healthcare_worker": "medical_boolean", "Previous_infection": "medical_boolean",
    "Pregnancy_Status": "medical_boolean", "Vaccination": "medical_boolean",
    "Hospitalised": "medical_boolean", "Intensive_care": "medical_boolean",
    "Home_monitoring": "medical_boolean", "Isolated": "medical_boolean",
    "Contact_with_case": "medical_boolean", "Travel_history": "medical_boolean",
    "Contact_animal": "medical_boolean",
    # Source
    "Source": "source", "Source_II": "source",
    "Source_III": "source", "Source_IV": "source",
}
LABEL_MAP_LC = {k.lower(): v.lower() for k, v in LABEL_MAP.items()}
def remap_labels(arr):
    return [LABEL_MAP_LC.get(x.lower(), x.lower()) for x in arr]

# Feature preparation
def make_inputs(df):
    cols = helpers.categorize_features()
    return [
        df[cols['char']].values.astype('float32'),
        df[cols['word']].values.astype('float32'),
        df[cols['par']].values.astype('float32'),
        df[cols['rest']].values.astype('float32'),
    ]

# Paths
data_dir = '../none_data'
model_json_path = '../model_files/sherlock_fine_tuned_model.json'
model_weights_path = '../model_files/sherlock_fine_tuned_weights.h5'

# Load processed features
X_test = pd.read_parquet(os.path.join(data_dir, 'processed', 'test.parquet'))

# Load raw inputs for human-readable records
raw_test_file = os.path.join(data_dir, 'raw', 'test.parquet')
raw_df = pd.read_parquet(raw_test_file)
human_col = 'values'  # adjust if needed
raw_inputs = raw_df[human_col].astype(str).tolist()

# Load and remap test labels
y_test_raw = pd.read_parquet(os.path.join(data_dir, 'test_labels.parquet')).values.flatten()
y_test = [x.lower() for x in remap_labels(y_test_raw)]

# Prepare model inputs
test_inputs = make_inputs(X_test)

# Load model architecture & weights
def load_model(json_path, weights_path):
    with open(json_path, 'r') as f:
        m = model_from_json(f.read())
    m.load_weights(weights_path)
    return m

model = load_model(model_json_path, model_weights_path)

# Inference
y_pred_probs = model.predict(test_inputs, batch_size=256)

# Identify top-1 and top-3
y_pred_idx = np.argmax(y_pred_probs, axis=1)
THRESHOLD = 0.5

y_pred_labels = []
for probs, idx in zip(y_pred_probs, y_pred_idx):
    if np.max(probs) < THRESHOLD or idx < 0 or idx >= NUM_CLASSES:
        y_pred_labels.append('__none__')
    else:
        y_pred_labels.append(class_names[idx])

y_top3_idx = np.argsort(y_pred_probs, axis=1)[:, -3:][:, ::-1]
y_top3_labels = [[class_names[i] if i < NUM_CLASSES else '__none__' for i in row] for row in y_top3_idx]
y_top3_probs = [[probs[i] for i in row] for probs, row in zip(y_pred_probs, y_top3_idx)]

# Metrics: map true labels to indices
y_true_int, y_pred_int = [], []
for true_lbl, pred_lbl in zip(y_test, y_pred_labels):
    if true_lbl in class_names:
        y_true_int.append(class_names.index(true_lbl))
        y_pred_int.append(class_names.index(pred_lbl) if pred_lbl in class_names else -1)

# Compute and display metrics
if y_true_int:
    print(f"Accuracy: {accuracy_score(y_true_int, y_pred_int):.4f}")
    print(f"Macro F1: {f1_score(y_true_int, y_pred_int, average='macro', zero_division=0):.4f}")
    print(f"Weighted F1: {f1_score(y_true_int, y_pred_int, average='weighted', zero_division=0):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true_int, y_pred_int, labels=list(range(NUM_CLASSES)), target_names=class_names, zero_division=0))
else:
    print("No valid samples to evaluate metrics.")

# Sample-level human-readable input vs predictions
print("\nSample-level Input vs Predicted:")
for inp, pred_lbl, top3_lbls, top3_prs in zip(raw_inputs, y_pred_labels, y_top3_labels, y_top3_probs):
    print(f"INPUT: {inp}")
    print(f"Predicted: {pred_lbl}")
    print("Top-3 predictions:")
    for lbl, pr in zip(top3_lbls, top3_prs):
        print(f"  {lbl}: {pr:.4f}")
    print()