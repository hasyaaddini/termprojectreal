import os
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def extract_features(img_path):
    img = Image.open(img_path).resize((128, 128))
    histogram = img.histogram()
    return np.array(histogram) / sum(histogram)

# --- Load dataset ---
image_folder = "sample_images"
X = []
y = []

for label in os.listdir(image_folder):
    label_folder = os.path.join(image_folder, label)
    if not os.path.isdir(label_folder):
        continue

    for img_name in os.listdir(label_folder):
        if img_name.endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(label_folder, img_name)
            features = extract_features(path)
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

# --- Train model ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = SVC(kernel="linear", probability=True)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# --- Save model ---
joblib.dump(clf, "model.pkl")
print("Model saved as model.pkl")

