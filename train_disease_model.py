import os
import pickle

import numpy as np

from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog


# =========================================================
# SETTINGS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "plant_dataset"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "disease_model.pkl"
)

MAX_IMAGES_PER_CLASS = 150
IMAGE_SIZE = (32, 32)


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(image):

    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    gray = np.mean(
        image_array,
        axis=2
    )

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


# =========================================================
# START
# =========================================================

print()
print("=" * 60)
print(" AgriSense AI Disease Detection Trainer")
print("=" * 60)
print()

if not os.path.exists(DATASET_DIR):

    raise FileNotFoundError(
        f"Dataset folder not found:\n{DATASET_DIR}"
    )

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# =========================================================
# LOAD DATASET
# =========================================================

X = []
y = []
class_counts = {}

print("Loading disease images...")
print()

for class_name in sorted(os.listdir(DATASET_DIR)):

    class_path = os.path.join(
        DATASET_DIR,
        class_name
    )

    if not os.path.isdir(class_path):
        continue

    print(f"Loading: {class_name}")

    image_count = 0

    for filename in sorted(os.listdir(class_path)):

        if image_count >= MAX_IMAGES_PER_CLASS:
            break

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):
            continue

        image_path = os.path.join(class_path, filename)

        try:

            with Image.open(image_path) as image:
                features = extract_features(image)

        except Exception as exc:

            print(f"Skipped {filename}: {exc}")
            continue

        X.append(features)
        y.append(class_name)
        image_count += 1

    class_counts[class_name] = image_count
    print(f"  Images used: {image_count}")

if len(X) == 0:
    raise ValueError("No images were found in the dataset folder.")

print()
print("=" * 60)
print(" DATASET SUMMARY")
print("=" * 60)
print()

for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

print(f"\nTotal images: {len(X)}")


# =========================================================
# CONVERT DATA
# =========================================================

X = np.asarray(
    X,
    dtype=np.float32
)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\nNumber of classes: {len(label_encoder.classes_)}")
print("\nClasses:")
for index, class_name in enumerate(label_encoder.classes_):
    print(f"  {index}: {class_name}")

print(f"\nFeatures per image: {X.shape[1]}")


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print()
print("=" * 60)
print(" TRAINING MODEL")
print("=" * 60)
print(f"\nTraining images: {len(X_train)}")
print(f"Testing images: {len(X_test)}")


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

model = LogisticRegression(
    max_iter=1500,
    solver="lbfgs",
    random_state=42
)

print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print()
print("=" * 60)
print(" MODEL RESULTS")
print("=" * 60)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(
    y_test,
    predictions,
    target_names=label_encoder.classes_,
    zero_division=0
))


# =========================================================
# SAVE MODEL
# =========================================================

model_data = {
    "model": model,
    "label_encoder": label_encoder,
    "image_size": IMAGE_SIZE,
    "feature_type": "HOG_plus_color_statistics_plus_histogram",
    "feature_count": X.shape[1]
}

with open(MODEL_FILE, "wb") as file:
    pickle.dump(model_data, file)

print()
print("=" * 60)
print(" MODEL SAVED SUCCESSFULLY")
print("=" * 60)
print(f"\nSaved to:\n{MODEL_FILE}")
print()
print("AgriSense AI disease model is ready! 🌱🤖")
