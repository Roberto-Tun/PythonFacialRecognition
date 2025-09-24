from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Any, Dict
import numpy as np
from datetime import datetime
from app.core.face_detector import detect_faces
from app.core.embedder import get_embedding
from app.utils.image_utils import read_image
from app.core.firebase import db, bucket
from google.cloud import storage, firestore
import uuid

router = APIRouter()

# --- Helpers -----------------------------------------------------------------


def calculate_similarity(emb1, emb2) -> float:
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    denom = (np.linalg.norm(emb1) * np.linalg.norm(emb2)) + 1e-12
    return float(np.dot(emb1, emb2) / denom)


def sanitize_id(doc_id: str) -> str:
    return doc_id.replace("/", "_").strip()


def encode_firestore_value(v: Any) -> Any:
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, firestore.DocumentReference):
        return v.path
    return v


def sanitize_firestore_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = sanitize_firestore_dict(v)
        elif isinstance(v, list):
            out[k] = [
                sanitize_firestore_dict(x)
                if isinstance(x, dict)
                else encode_firestore_value(x)
                for x in v
            ]
        else:
            out[k] = encode_firestore_value(v)
    return out


# Upload image to Firebase Storage
def upload_to_firebase_storage(
    bucket_name, filename, file_data, content_type="image/jpeg"
):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"rostros/{filename}")
    blob.upload_from_string(file_data, content_type=content_type)
    blob.make_public()
    return blob.public_url


# --- Routes ------------------------------------------------------------------


@router.post("/register_missing_person/")
async def register_missing_person(
    id: str = Form(...),
    nombre: str = Form(...),
    apellido_paterno: str = Form(...),
    apellido_materno: str = Form(...),
    genero: str = Form(...),
    edad: str = Form(...),
    fecha_de_la_denuncia: str = Form(...),
    fecha_de_los_hechos: str = Form(...),
    lugar_de_los_hechos: str = Form(...),
    nacionalidad: str = Form(...),
    ojos: str = Form(...),
    cabello: str = Form(...),
    complexion: str = Form(...),
    estatura: str = Form(...),
    peso: str = Form(...),
    tez: str = Form(...),
    senas_particulares: str = Form(...),
    encontrado: str = Form(...),
    files: List[UploadFile] = File(...),
):
    try:
        id = sanitize_id(id)

        person_data = {
            "id": id,
            "nombre": nombre,
            "apellido_paterno": apellido_paterno,
            "apellido_materno": apellido_materno,
            "género": genero,
            "edad": edad,
            "fecha_de_la_denuncia": fecha_de_la_denuncia,
            "fecha_de_los_hechos": fecha_de_los_hechos,
            "lugar_de_los_hechos": lugar_de_los_hechos,
            "nacionalidad": nacionalidad,
            "ojos": ojos,
            "cabello": cabello,
            "complexion": complexion,
            "estatura": estatura,
            "peso": peso,
            "tez": tez,
            "señas_particulares": senas_particulares,
            "encontrado": encontrado,
            "fecha_registro": datetime.utcnow(),
        }

        db.collection("PersonasDesaparecidas").document(id).set(person_data)

        for file in files:
            contents = await file.read()
            img = read_image(contents)
            faces = detect_faces(img)
            if not faces:
                continue

            # Subir imagen a Storage
            filename = f"{id}_{uuid.uuid4().hex}.jpg"
            blob = bucket.blob(f"rostros/{filename}")
            blob.upload_from_string(contents, content_type=file.content_type)
            blob.make_public()
            image_url = blob.public_url

            for face in faces:
                embedding = get_embedding(face)
                db.collection("Vectores").add(
                    {
                        # keep as plain ID (string) for compatibility
                        "id_persona_desaparecida": id,
                        "vector": embedding.tolist(),
                        "nombre_imagen": file.filename,
                        "ruta_storage": image_url,
                        "fecha_subida": datetime.utcnow(),
                    }
                )

        return {"status": "success", "id": id}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/identify/")
async def identify_faces(files: List[UploadFile] = File(...)):
    try:
        known_faces = []

        # Load all known embeddings
        for doc in db.collection("Vectores").stream():
            data = doc.to_dict()
            vector_raw = data.get("vector", [])

            if (
                isinstance(vector_raw, list)
                and len(vector_raw) == 512
                and all(isinstance(x, (float, int)) for x in vector_raw)
            ):
                # Puede venir como referencia o como string
                person_ref_or_id = data.get("id_persona_desaparecida")

                if isinstance(person_ref_or_id, firestore.DocumentReference):
                    persona_ref = person_ref_or_id
                    person_id = None
                else:
                    persona_ref = None
                    person_id = person_ref_or_id  # string o None

                known_faces.append(
                    {
                        "persona_ref": persona_ref,
                        "person_id": person_id,
                        "embedding": np.array(vector_raw, dtype=np.float32).flatten(),
                        "image_url": data.get("ruta_storage"),
                    }
                )

        if not known_faces:
            return JSONResponse(
                status_code=404, content={"error": "No embeddings found in database"}
            )

        results = []

        for file in files:
            contents = await file.read()
            img = read_image(contents)
            face_images = detect_faces(img)

            if not face_images:
                results.append({"filename": file.filename, "error": "No faces found"})
                continue

            for face in face_images:
                query_emb = get_embedding(face).flatten()
                best_match = {
                    "persona_ref": None,
                    "person_id": None,
                    "similarity": 0.0,
                    "image_url": None,
                }

                for known in known_faces:
                    sim = calculate_similarity(query_emb, known["embedding"])
                    if sim > best_match["similarity"]:
                        best_match = {
                            "persona_ref": known["persona_ref"],
                            "person_id": known["person_id"],
                            "similarity": sim,
                            "image_url": known["image_url"],
                        }

                # Obtener info de la persona
                person_info = {}
                if best_match["persona_ref"] is not None:
                    doc_persona = best_match["persona_ref"].get()
                    if doc_persona.exists:
                        person_info = sanitize_firestore_dict(doc_persona.to_dict())
                        person_info["id"] = doc_persona.id
                elif best_match["person_id"]:
                    ref = db.collection("PersonasDesaparecidas").document(
                        str(best_match["person_id"])
                    )
                    doc_persona = ref.get()
                    if doc_persona.exists:
                        person_info = sanitize_firestore_dict(doc_persona.to_dict())
                        person_info["id"] = doc_persona.id

                results.append(
                    {
                        "filename": file.filename,
                        "match_info": person_info,  # already sanitized (JSON safe)
                        "similarity": f"{round(best_match['similarity'] * 100, 2)}%",
                        "image_url": best_match["image_url"] or "",
                    }
                )

        # Entire response is JSON serializable now
        return {"results": results}

    except Exception as e:
        # keep error visible
        return JSONResponse(
            status_code=500, content={"error": f"Firestore error: {str(e)}"}
        )
