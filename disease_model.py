import joblib
import numpy as np
from PIL import Image


MODEL_PATH = "models/disease_model.pkl"


# Load saved model package
model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
label_encoder = model_data["label_encoder"]
image_size = model_data["image_size"]


def predict_disease(image):

    # Open image
    img = Image.open(image).convert("RGB")

    # Resize to the size used during training
    img = img.resize(tuple(image_size))

    # Convert to numpy array
    img_array = np.array(img)

    # Normalize pixel values
    img_array = img_array / 255.0

    # Flatten image
    img_array = img_array.reshape(1, -1)

    # Prediction
    prediction = model.predict(img_array)[0]

    # Convert encoded label to disease name
    disease_name = label_encoder.inverse_transform(
        [prediction]
    )[0]

    # Confidence
    probabilities = model.predict_proba(img_array)[0]

    confidence = float(np.max(probabilities) * 100)

    return disease_name, confidence