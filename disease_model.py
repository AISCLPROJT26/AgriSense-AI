import os
import pickle

import numpy as np
from PIL import Image
from skimage.feature import hog


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "disease_model.pkl"
)

IMAGE_SIZE = (32, 32)


def _load_model_data():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Disease model not found: {MODEL_FILE}"
        )

    with open(MODEL_FILE, "rb") as file:
        model_data = pickle.load(file)

    if not isinstance(model_data, dict):
        raise TypeError(
            "Disease model file is not a valid model package."
        )

    return model_data


# Load saved model package
model_data = _load_model_data()

model = model_data["model"]
label_encoder = model_data["label_encoder"]
image_size = tuple(model_data["image_size"])


def extract_features(image):
    image = Image.open(image).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32) / 255.0

    gray = np.mean(image_array, axis=2)
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(3, 3),
        block_norm="L2-Hys"
    )

    color_features = []
    for channel in range(3):
        channel_data = image_array[:, :, channel]
        color_features.extend([
            np.mean(channel_data),
            np.std(channel_data),
            np.min(channel_data),
            np.max(channel_data)
        ])

    histogram_features = []
    for channel in range(3):
        histogram, _ = np.histogram(
            image_array[:, :, channel],
            bins=64,
            range=(0, 1),
            density=True
        )
        histogram_features.extend(histogram)

    features = np.concatenate([
        hog_features,
        np.asarray(color_features, dtype=np.float32),
        np.asarray(histogram_features, dtype=np.float32)
    ])

    return features.astype(np.float32)


def predict_disease(image):
    features = extract_features(image).reshape(1, -1)

    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is not None and features.shape[1] != expected_features:
        raise ValueError(
            "Disease model feature mismatch: "
            f"expected {expected_features}, got {features.shape[1]}"
        )

    prediction = model.predict(features)[0]
    disease_name = label_encoder.inverse_transform([prediction])[0]

    probabilities = model.predict_proba(features)[0]
    confidence = float(np.max(probabilities) * 100)

    return disease_name, confidence