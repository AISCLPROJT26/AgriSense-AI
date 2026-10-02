import os
import pickle
import numpy as np

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# =========================================================
# SETTINGS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

def extract_features(image_path):

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        image = image.resize(
            IMAGE_SIZE
        )

        pixels = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        # -------------------------------------------------
        # Pixel features
        # -------------------------------------------------

        pixel_features = pixels.flatten()


        # -------------------------------------------------
        # Color histogram features
        # -------------------------------------------------

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


        # -------------------------------------------------
        # Combine everything
        # -------------------------------------------------

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


    except Exception as e:

        print(
            f"Could not read {image_path}: {e}"
        )

        return None


# =========================================================
# START
# =========================================================

print()
print("===================================")
print(" AgriSense AI Disease Model")
print("===================================")
print()


# =========================================================
# CHECK DATASET
# =========================================================

if not os.path.exists(DATASET_DIR):

    print(
        f"ERROR: Dataset folder not found:"
    )

    print(
        DATASET_DIR
    )

    input("\nPress Enter to exit...")
    exit()


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


print("Loading images...\n")


for class_name in sorted(
    os.listdir(DATASET_DIR)
):

    class_path = os.path.join(
        DATASET_DIR,
        class_name
    )

    if not os.path.isdir(class_path):
        continue


    print(
        f"Loading: {class_name}"
    )


    image_count = 0


    for filename in os.listdir(
        class_path
    ):

        if image_count >= MAX_IMAGES_PER_CLASS:
            break


        if not filename.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            )
        ):
            continue


        image_path = os.path.join(
            class_path,
            filename
        )


        features = extract_features(
            image_path
        )


        if features is not None:

            X.append(features)
            y.append(class_name)

            image_count += 1


    class_counts[class_name] = image_count


    print(
        f"  Images used: {image_count}"
    )


# =========================================================
# DATASET SUMMARY
# =========================================================

if len(X) == 0:

    print(
        "\nERROR: No images were found."
    )

    input("\nPress Enter to exit...")
    exit()


print()
print("===================================")
print(" Dataset Summary")
print("===================================")
print()


for class_name, count in class_counts.items():

    print(
        f"{class_name}: {count}"
    )


print(
    f"\nTotal images: {len(X)}"
)


# =========================================================
# CONVERT DATA
# =========================================================

X = np.asarray(
    X,
    dtype=np.float32
)


label_encoder = LabelEncoder()


y_encoded = label_encoder.fit_transform(
    y
)


print(
    f"\nNumber of classes: "
    f"{len(label_encoder.classes_)}"
)


print(
    f"Features per image: "
    f"{X.shape[1]}"
)


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
print("Training model...")

print(
    f"Training images: {len(X_train)}"
)

print(
    f"Testing images: {len(X_test)}"
)


# =========================================================
# TRAIN LOGISTIC REGRESSION
# =========================================================

model = LogisticRegression(

    max_iter=1500,

    solver="lbfgs"
)


model.fit(
    X_train,
    y_train
)


# =========================================================
# TEST MODEL
# =========================================================

predictions = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    predictions
)


print()
print("===================================")
print(" MODEL RESULTS")
print("===================================")

print(
    f"\nAccuracy: "
    f"{accuracy * 100:.2f}%"
)


print(
    "\nClassification Report:\n"
)


print(
    classification_report(

        y_test,

        predictions,

        target_names=label_encoder.classes_,

        zero_division=0
    )
)


# =========================================================
# SAVE MODEL PACKAGE
# =========================================================

model_data = {

    "model": model,

    "label_encoder": label_encoder,

    "image_size": IMAGE_SIZE,

    "feature_type": "pixels_plus_color_histogram",

    "feature_count": X.shape[1]
}


with open(
    MODEL_FILE,
    "wb"
) as file:

    pickle.dump(
        model_data,
        file
    )


# =========================================================
# FINISHED
# =========================================================

print()
print("===================================")
print(" MODEL SAVED")
print("===================================")

print(
    f"\nSaved to:"
)

print(
    MODEL_FILE
)

print()
print(
    "AgriSense AI disease model is ready! 🌱🤖"
)

input(
    "\nPress Enter to exit..."
)