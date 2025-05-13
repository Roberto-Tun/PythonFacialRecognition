import os
import cv2
import numpy as np
import pickle
import time
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

mtcnn = MTCNN()
embedder = FaceNet()

def extract_face(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = mtcnn.detect_faces(img)
    if not detections:
        return None
    x, y, w, h = detections[0]['box']
    x, y = abs(x), abs(y)
    face = img[y:y+h, x:x+w]
    return cv2.resize(face, (160, 160))

def get_embedding(face):
    face = face.astype('float32')
    face = np.expand_dims(face, axis=0)
    return embedder.embeddings(face)[0]

def train_and_save(data_dir="data/rostros"):
    start_time = time.time()
    embeddings = []
    labels = []
    known_faces = []

    for file in os.listdir(data_dir):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        name = os.path.splitext(file)[0].replace("_", " ")
        path = os.path.join(data_dir, file)
        face = extract_face(path)
        if face is None:
            print(f"Skipping: {file}")
            continue
        emb = get_embedding(face)
        embeddings.append(emb)
        labels.append(name)
        known_faces.append({ "name": name, "embedding": emb })

    X = np.asarray(embeddings)
    y = LabelEncoder().fit_transform(labels)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    model = SVC(kernel="linear", probability=True)
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    pickle.dump(model, open("models/svm_model.pkl", "wb"))
    pickle.dump(encoder, open("models/label_encoder.pkl", "wb"))
    pickle.dump(known_faces, open("models/known_faces.pkl", "wb"))
    print("Time taken to train the model: ", time.time() - start_time, "seconds")
    print("✅ Model, encoder, and known_faces saved.")
