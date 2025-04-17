#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import pyarrow as pa

# Import necessary modules from Sherlock
from sherlock import helpers
from sherlock.deploy.model import SherlockModel
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
    # Set a fixed hash seed (replacing the %env magic command)
    os.environ['PYTHONHASHSEED'] = '13'

    # Initialize feature extraction models
    # This step downloads necessary embedding files (only once) and sets up the models.
    prepare_feature_extraction()
    initialise_word_embeddings()
    initialise_pretrained_model(400)
    initialise_nltk()

    df = pd.read_csv(
        "../cholera.csv",
        sep=';',
        skiprows=1,
        dtype=str
    )

    series_of_columns = pd.Series(
        { col: df[col].dropna().tolist() for col in df.columns }
    )

    # Extract features from the data.
    # The features will be saved to a temporary CSV file.
    extract_features("../custom-sherlock.csv", series_of_columns)
    feature_vectors = pd.read_csv("../custom-sherlock.csv", dtype=np.float32)

    # Initialize the Sherlock model and load the weights.
    model = SherlockModel()
    model.initialize_model_from_json(with_weights=True, model_id="sherlock")

    # Predict the semantic type for the column based on its feature vector.
    predicted_labels = model.predict(feature_vectors, "sherlock")
    #print("Predicted semantic types:", predicted_labels)

    results = pd.DataFrame({
        "column": series_of_columns.index,
        "predicted_labels": predicted_labels
    })

    print(results)

if __name__ == "__main__":
    main()