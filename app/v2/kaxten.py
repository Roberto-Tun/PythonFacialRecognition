"""Índice `desaparecidos`: los datos de KAXTEN **en su formato de la v1**.

KAXTEN está en producción y su app y la v1 (`/identify/`) leen `Vectores` de la
base `(default)` así:

    id_persona_desaparecida: DocumentReference a PersonasDesaparecidas/<id>
                             (en documentos viejos, el id en texto)
    vector:                  512 números
    nombre_imagen, ruta_storage (la app muestra esa foto), fecha_subida

Este adaptador presenta esos documentos como el índice `desaparecidos` sin
migrarlos, y escribe las altas nuevas en ese mismo formato, para que la app de
KAXTEN y la v1 las sigan viendo. Ver docs/v2/PLAN_V2.md, fase E.

Reglas propias de este índice:
- Solo coincide una persona con `visibilidad` pública (T-05). El campo que falta
  cuenta como público; un valor raro se oculta (mismo criterio que la app).
- Una alta exige que la persona ya exista: la v2 no crea personas.
"""

import hashlib
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Set

from app.v2.config import INDICE_KAXTEN, MODELO
from app.v2.repositorio import Vector

COLECCION_VECTORES = "Vectores"
COLECCION_PERSONAS = "PersonasDesaparecidas"
CARPETA_FOTOS = "rostros"
VISIBILIDAD_PUBLICA = 1


def sujeto_de(valor) -> str:
    """El id de la persona, venga como referencia o como texto (documentos viejos)."""
    if valor is None:
        return ""
    id_ = getattr(valor, "id", None)
    if id_:
        return str(id_)
    texto = str(valor)
    return texto.rsplit("/", 1)[-1]


def vector_desde_legacy(doc_id: str, d: dict) -> Vector:
    fecha = d.get("fecha_subida")
    return Vector(
        id=doc_id,
        indice=INDICE_KAXTEN,
        sujeto_id=sujeto_de(d.get("id_persona_desaparecida")),
        grupo="",
        vector=list(d.get("vector") or []),
        # Los de la v1 no traen `modelo`: son del mismo FaceNet del servidor.
        modelo=str(d.get("modelo") or MODELO),
        fecha=fecha.isoformat() if hasattr(fecha, "isoformat") else str(fecha or ""),
        ruta_storage=str(d.get("ruta_storage") or ""),
    )


def documento_legacy(v: Vector, referencia_persona) -> dict:
    return {
        "id_persona_desaparecida": referencia_persona,
        "vector": [float(x) for x in v.vector],
        "nombre_imagen": v.ruta_storage.rsplit("/", 1)[-1] if v.ruta_storage else "",
        "ruta_storage": v.ruta_storage,
        "fecha_subida": datetime.now(timezone.utc),
        # Campos nuevos: la v1 y la app los ignoran.
        "modelo": v.modelo,
        "origen": "v2",
    }


def es_visible(datos_persona: Optional[dict]) -> bool:
    if datos_persona is None:
        return False
    valor = datos_persona.get("visibilidad", VISIBILIDAD_PUBLICA)
    try:
        return int(valor) == VISIBILIDAD_PUBLICA
    except (TypeError, ValueError):
        return False


def nombre_foto(sujeto_id: str, contenido: bytes) -> str:
    return f"{sujeto_id}_{hashlib.sha256(contenido).hexdigest()[:16]}.jpg"


class RepositorioKaxten:
    """Firestore `(default)` + el bucket compartido. Solo el índice `desaparecidos`."""

    def __init__(self, cliente_firestore, bucket) -> None:
        self._db = cliente_firestore
        self._vectores = cliente_firestore.collection(COLECCION_VECTORES)
        self._personas = cliente_firestore.collection(COLECCION_PERSONAS)
        self._bucket = bucket

    @staticmethod
    def _exigir(indice: str) -> None:
        if indice != INDICE_KAXTEN:
            raise ValueError(f"El repositorio de KAXTEN solo tiene el índice '{INDICE_KAXTEN}'.")

    def indices_existentes(self) -> List[str]:
        return [INDICE_KAXTEN]

    def datos(self, indice: str, ids: List[str]) -> Dict[str, dict]:
        """Los datos de la persona, como los devolvía `/identify/` en `match_info`."""
        from app.v2.datos import sanear

        self._exigir(indice)
        if not ids:
            return {}
        docs = self._db.get_all([self._personas.document(i) for i in ids])
        return {d.id: sanear(d.to_dict() or {}) for d in docs if d.exists}

    def leer_indice(self, indice: str) -> List[Vector]:
        self._exigir(indice)
        # La v1 también lee toda la colección; aquí al menos se hace una vez por
        # refresco y no en cada búsqueda.
        return [vector_desde_legacy(d.id, d.to_dict() or {}) for d in self._vectores.stream()]

    def guardar(self, vectores: List[Vector]) -> None:
        for i in range(0, len(vectores), 400):
            lote = self._db.batch()
            for v in vectores[i : i + 400]:
                self._exigir(v.indice)
                lote.set(self._vectores.document(v.id), documento_legacy(v, self._personas.document(v.sujeto_id)))
            lote.commit()

    def borrar_sujeto(self, indice: str, sujeto_id: str) -> int:
        from google.cloud.firestore_v1.base_query import FieldFilter

        self._exigir(indice)
        ref = self._personas.document(sujeto_id)
        docs = {d.id: d for d in self._vectores.where(filter=FieldFilter("id_persona_desaparecida", "==", ref)).stream()}
        # Documentos viejos con el id en texto.
        for d in self._vectores.where(filter=FieldFilter("id_persona_desaparecida", "==", sujeto_id)).stream():
            docs[d.id] = d
        lista = list(docs.values())
        for i in range(0, len(lista), 400):
            lote = self._db.batch()
            for d in lista[i : i + 400]:
                lote.delete(d.reference)
            lote.commit()
        return len(lista)

    def existe(self, sujeto_id: str) -> bool:
        return self._personas.document(sujeto_id).get().exists

    def visibles(self, sujetos: Iterable[str]) -> Set[str]:
        ids = [s for s in dict.fromkeys(sujetos) if s]
        if not ids:
            return set()
        docs = self._db.get_all([self._personas.document(s) for s in ids])
        return {d.id for d in docs if d.exists and es_visible(d.to_dict())}

    def subir_foto(self, sujeto_id: str, contenido: bytes) -> str:
        """Como la v1: `rostros/<archivo>` pública, que es la que pinta la app."""
        blob = self._bucket.blob(f"{CARPETA_FOTOS}/{nombre_foto(sujeto_id, contenido)}")
        blob.upload_from_string(contenido, content_type="image/jpeg")
        blob.make_public()
        return blob.public_url
