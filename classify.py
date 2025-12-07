import numpy as np
from PIL import Image
import joblib

def extract_features(img_path):
    img = Image.open(img_path).resize((128, 128))
    histogram = img.histogram()
    return np.array(histogram) / sum(histogram)

def classify_image(img_path):
    model = joblib.load("model.pkl")
    features = extract_features(img_path)
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0].max()
    return prediction, probability
