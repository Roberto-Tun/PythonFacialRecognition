import pickle
import numpy as np

model = pickle.load(open("models/svm_model.pkl", "rb"))
encoder = pickle.load(open("models/label_encoder.pkl", "rb"))

def predict_face(embedding):
    embedding = embedding.reshape(1, -1)
    pred = model.predict(embedding)[0]
    prob = model.predict_proba(embedding)[0][pred]
    label = encoder.inverse_transform([pred])[0]
    return label, prob