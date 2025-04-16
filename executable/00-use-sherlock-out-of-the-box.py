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

    # Create a pandas Series with sample table column data
    data = pd.Series(
        [
            ["Jane Smith", "Lute Ahorn", "Anna James"],
            ["Amsterdam", "Haarlem", "Zwolle"],
            ["Chabot Street 19", "1200 fifth Avenue", "Binnenkant 22, 1011BH"],
        ],
        name="values"
    )

    # Extract features from the data.
    # The features will be saved to a temporary CSV file.
    extract_features("../temporary.csv", data)
    feature_vectors = pd.read_csv("../temporary.csv", dtype=np.float32)

    # Initialize the Sherlock model and load the weights.
    model = SherlockModel()
    model.initialize_model_from_json(with_weights=True, model_id="sherlock")

    # Predict the semantic type for the column based on its feature vector.
    predicted_labels = model.predict(feature_vectors, "sherlock")
    print("Predicted semantic types:", predicted_labels)

if __name__ == "__main__":
    main()