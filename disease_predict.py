import os
import pickle
import numpy as np

from PIL import Image


# =========================================================
# MODEL LOCATION
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "disease_model.pkl"
)


# =========================================================
# LOAD MODEL
# =========================================================

with open(
    MODEL_FILE,
    "rb"
) as file:

    model_data = pickle.load(file)


model = model_data["model"]

label_encoder = model_data["label_encoder"]

image_size = model_data["image_size"]


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(image):

    # Open uploaded image
    image = Image.open(
        image
    ).convert("RGB")


    # Resize exactly like training
    image = image.resize(
        tuple(image_size)
    )


    # Convert to NumPy
    pixels = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0


    # Pixel features
    pixel_features = pixels.flatten()


    # Color histogram features
    histogram_features = []


    for channel in range(3):

        histogram, _ = np.histogram(

            pixels[:, :, channel],

            bins=16,

            range=(0, 1),

            density=True
        )

        histogram_features.extend(
            histogram
        )


    # Combine exactly like training
    features = np.concatenate(

        [
            pixel_features,

            np.array(
                histogram_features,
                dtype=np.float32
            )
        ]

    )


    return features


# =========================================================
# PREDICTION
# =========================================================

def predict_disease(image):

    features = extract_features(
        image
    )


    # Reshape for sklearn
    features = features.reshape(
        1,
        -1
    )


    # Prediction
    prediction = model.predict(
        features
    )[0]


    # Disease name
    disease_name = (
        label_encoder.inverse_transform(
            [prediction]
        )[0]
    )


    # Confidence
    probabilities = model.predict_proba(
        features
    )[0]


    confidence = float(
        np.max(probabilities) * 100
    )


    return disease_name, confidence