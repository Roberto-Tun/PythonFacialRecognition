from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
import numpy as np

from app.core.face_detector import detect_faces
from app.core.embedder import get_embedding
from app.core.svm_model import model, encoder
from app.utils.image_utils import read_image

# Load known embeddings for similarity calculation
import pickle
with open("models/known_faces.pkl", "rb") as f:
    known_faces = pickle.load(f)  # List of dicts: { "name": ..., "embedding": ... }

router = APIRouter()

def calculate_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

@router.post("/identify/")
async def identify_faces(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        contents = await file.read()
        img = read_image(contents)
        face_images = detect_faces(img)

        if not face_images:
            results.append({
                "filename": file.filename,
                "error": "No faces found"
            })
            continue

        for face in face_images:
            query_emb = get_embedding(face).reshape(1, -1)

            pred = model.predict(query_emb)[0]
            prob = model.predict_proba(query_emb)[0][pred]
            label = encoder.inverse_transform([pred])[0]

            # Cosine similarity against known embeddings
            similarity = 0
            for known in known_faces:
                if known["name"] == label:
                    similarity = calculate_similarity(query_emb.flatten(), known["embedding"])
                    break

            results.append({
                "filename": file.filename,
                "match": label,
                "confidence": f"{round(prob * 100, 2)}%",
                "similarity": f"{round(similarity * 100, 2)}%"
            })

    return {"results": results}


@router.post("/embeddings/")
async def generate_embeddings(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        contents = await file.read()
        img = read_image(contents)
        face_images = detect_faces(img)

        if not face_images:
            results.append({
                "filename": file.filename,
                "error": "No faces found"
            })
            continue

        for idx, face in enumerate(face_images):
            embedding = get_embedding(face)
            results.append({
                "filename": file.filename,
                "face_index": idx,
                "embedding": embedding.tolist()
            })

    return {"results": results}