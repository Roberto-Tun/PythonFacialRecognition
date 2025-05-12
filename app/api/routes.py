from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
import numpy as np
from datetime import datetime
from app.core.face_detector import detect_faces
from app.core.embedder import get_embedding
from app.utils.image_utils import read_image
from app.core.firebase import db

router = APIRouter()

def calculate_similarity(emb1, emb2):
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


from fastapi import Form, File, UploadFile
from typing import List
from datetime import datetime
from fastapi.responses import JSONResponse
import json
from app.core.firebase import db
from app.core.face_detector import detect_faces
from app.core.embedder import get_embedding
from app.utils.image_utils import read_image

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

        doc_ref = db.collection("personas_desaparecidas").document(PersonaID)
        doc_ref.set(person_data)

        for file in files:
            contents = await file.read()
            img = read_image(contents)
            faces = detect_faces(img)

            if not faces:
                continue

            for face in faces:
                embedding = get_embedding(face)
                db.collection("Vectores").add({
                    "id_persona_desaparecida": PersonaID,
                    "vector": embedding.tolist(),
                    "nombre_imagen": file.filename,
                    "ruta_storage": None,
                    "fecha_subida": datetime.utcnow()
                })

        return {"status": "success", "id": PersonaID}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})



@router.post("/identify/")
async def identify_faces(files: List[UploadFile] = File(...)):
    try:
        known_faces = []
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
                    "embedding": np.array(vector_raw, dtype=np.float32).flatten()
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
                best_match = {"persona_id": None, "similarity": 0}

                for known in known_faces:
                    sim = calculate_similarity(query_emb, known["embedding"])
                    if sim > best_match["similarity"]:
                        best_match = {
                            "persona_id": known["persona_id"],
                            "similarity": sim
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
                    "similarity": f"{round(best_match['similarity'] * 100, 2)}%"
                })

        return {"results": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Firestore error: {str(e)}"})
