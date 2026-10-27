import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# Load crop dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "crop_data.csv")

data = pd.read_csv(DATA_PATH)


# Features used by the AI
FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]


# Training data
X = data[FEATURES]
y = data["label"]


# Create AI model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train the model
model.fit(X, y)


def recommend_crop(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall
):
    """
    Predict the most suitable crop.
    """

    input_data = pd.DataFrame([[
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    ]], columns=FEATURES)

    prediction = model.predict(input_data)

    return prediction[0]