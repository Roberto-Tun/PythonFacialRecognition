from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List
import numpy as np
from datetime import datetime
from app.core.face_detector import detect_faces
from app.core.embedder import get_embedding
from app.utils.image_utils import read_image
from app.core.firebase import db, bucket
from google.cloud import storage  # NEW
import uuid


router = APIRouter()

def calculate_similarity(emb1, emb2):
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


# Upload image to Firebase Storage
def upload_to_firebase_storage(bucket_name, filename, file_data, content_type="image/jpeg"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"rostros/{filename}")
    blob.upload_from_string(file_data, content_type=content_type)
    blob.make_public()
    return blob.public_url


@router.post("/register_missing_person/")
async def register_missing_person(
    PersonaID: str = Form(...),
    Nombre: str = Form(...),
    Edad: str = Form(...),
    Genero: str = Form(...),
    Nacionalidad: str = Form(...),
    Fecha_denuncia: str = Form(...),
    Fecha_hechos: str = Form(...),
    Lugar_hechos: str = Form(...),
    Complexion: str = Form(...),
    Tez: str = Form(...),
    Cabello: str = Form(...),
    Ojos: str = Form(...),
    Estatura: str = Form(...),
    Peso: str = Form(...),
    Encontrado: str = Form(...),
    Senas_particulares: str = Form(...),
    files: List[UploadFile] = File(...)
):
    try:
        person_data = {
            "PersonaID": PersonaID,
            "Nombre": Nombre,
            "Edad": Edad,
            "Género": Genero,
            "Nacionalidad": Nacionalidad,
            "Fecha de la denuncia": Fecha_denuncia,
            "Fecha de los hechos": Fecha_hechos,
            "Lugar de los hechos": Lugar_hechos,
            "Complexión": Complexion,
            "Tez": Tez,
            "Cabello": Cabello,
            "Ojos": Ojos,
            "Estatura": Estatura,
            "Peso": Peso,
            "Encontrado": Encontrado,
            "Señas particulares": Senas_particulares,
            "fecha_subida": datetime.utcnow()
        }

        db.collection("personas_desaparecidas").document(PersonaID).set(person_data)

        for file in files:
            contents = await file.read()
            img = read_image(contents)
            faces = detect_faces(img)

            if not faces:
                continue

            # Subir imagen a Storage
            filename = f"{PersonaID}_{uuid.uuid4().hex}.jpg"
            blob = bucket.blob(f"rostros/{filename}")
            blob.upload_from_string(contents, content_type=file.content_type)
            blob.make_public()
            image_url = blob.public_url

            for face in faces:
                embedding = get_embedding(face)
                db.collection("Vectores").add({
                    "id_persona_desaparecida": PersonaID,
                    "vector": embedding.tolist(),
                    "nombre_imagen": file.filename,
                    "ruta_storage": image_url,
                    "fecha_subida": datetime.utcnow()
                })

        return {"status": "success", "id": PersonaID}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
@router.post("/identify/")
async def identify_faces(files: List[UploadFile] = File(...)):
    try:
        known_faces = []
        vector_docs = []

        for doc in db.collection("Vectores").stream():
            data = doc.to_dict()
            vector_raw = data.get("vector", [])

            if (
                isinstance(vector_raw, list)
                and len(vector_raw) == 512
                and all(isinstance(x, (float, int)) for x in vector_raw)
            ):
                known_faces.append({
                    "persona_id": data.get("id_persona_desaparecida"),
                    "embedding": np.array(vector_raw, dtype=np.float32).flatten(),
                    "image_url": data.get("ruta_storage")
                })

        if not known_faces:
            return JSONResponse(status_code=404, content={"error": "No embeddings found in database"})

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
                query_emb = get_embedding(face).flatten()
                best_match = {"persona_id": None, "similarity": 0, "image_url": None}

                for known in known_faces:
                    sim = calculate_similarity(query_emb, known["embedding"])
                    if sim > best_match["similarity"]:
                        best_match = {
                            "persona_id": known["persona_id"],
                            "similarity": sim,
                            "image_url": known["image_url"]
                        }

                person_info = {}
                if best_match["persona_id"]:
                    ref = db.collection("personas_desaparecidas").document(best_match["persona_id"])
                    doc = ref.get()
                    if doc.exists:
                        person_info = doc.to_dict()
                        person_info["PersonaID"] = doc.id

                results.append({
                    "filename": file.filename,
                    "match_info": person_info,
                    "similarity": f"{round(best_match['similarity'] * 100, 2)}%",
                    "image_url": best_match["image_url"]
                })

        return {"results": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Firestore error: {str(e)}"})