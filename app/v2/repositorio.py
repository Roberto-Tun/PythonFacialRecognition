"""Dónde viven los vectores: `base/Vectores`, con el mismo esquema que ya
escriben los teléfonos (`kaxten_base/lib/services/vectores_local_cache_service.dart`).

Documento:
    id:        <indice>_<sujeto>_<n>      (los escribe el teléfono)
               <indice>_<sujeto>_s<hash>  (los escribe el servidor)
    indice:    "miembros", "registros"…
    sujeto_id: id del socio, del Registro…
    grupo:     subconjunto dentro del índice (p. ej. el negocio); "" si no hay
    vector:    512 números
    modelo:    "facenet_512" (servidor) o "facenet_512_fp16" (teléfono)
    fecha:     ISO 8601 en texto (así lo lee la app)

Cambiar estos nombres rompe el reconocimiento sin red de la app.
"""

import hashlib
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Protocol


@dataclass(frozen=True)
class Vector:
    id: str
    indice: str
    sujeto_id: str
    grupo: str
    vector: List[float]
    modelo: str
    fecha: str

    def documento(self) -> dict:
        return {
            "indice": self.indice,
            "sujeto_id": self.sujeto_id,
            "grupo": self.grupo,
            "vector": [float(x) for x in self.vector],
            "modelo": self.modelo,
            "fecha": self.fecha,
        }

    @staticmethod
    def desde_documento(doc_id: str, d: dict) -> "Vector":
        return Vector(
            id=doc_id,
            indice=str(d.get("indice") or ""),
            sujeto_id=str(d.get("sujeto_id") or ""),
            grupo=str(d.get("grupo") or ""),
            vector=list(d.get("vector") or []),
            modelo=str(d.get("modelo") or ""),
            fecha=str(d.get("fecha") or ""),
        )


def ahora_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def id_servidor(indice: str, sujeto_id: str, vector: Iterable[float]) -> str:
    """Determinista: dar de alta la misma cara dos veces no duplica el documento
    (un reintento de red tampoco)."""
    crudo = b"".join(struct.pack("<f", float(x)) for x in vector)
    huella = hashlib.sha256(crudo).hexdigest()[:12]
    return f"{indice}_{sujeto_id}_s{huella}"


class Repositorio(Protocol):
    def leer_indice(self, indice: str) -> List[Vector]: ...
    def guardar(self, vectores: List[Vector]) -> None: ...
    def borrar_sujeto(self, indice: str, sujeto_id: str) -> int: ...


class RepositorioMemoria:
    """Para pruebas y para correr sin Firestore."""

    def __init__(self) -> None:
        self.docs: Dict[str, dict] = {}
        self.lecturas = 0

    def leer_indice(self, indice: str) -> List[Vector]:
        self.lecturas += 1
        return [Vector.desde_documento(i, d) for i, d in self.docs.items() if d.get("indice") == indice]

    def guardar(self, vectores: List[Vector]) -> None:
        for v in vectores:
            self.docs[v.id] = v.documento()

    def borrar_sujeto(self, indice: str, sujeto_id: str) -> int:
        ids = [i for i, d in self.docs.items() if d.get("indice") == indice and d.get("sujeto_id") == sujeto_id]
        for i in ids:
            del self.docs[i]
        return len(ids)


class RepositorioFirestore:
    """Firestore, base `base`. Solo consultas de igualdad: no necesitan índices
    compuestos."""

    def __init__(self, cliente_firestore, coleccion: str) -> None:
        self._col = cliente_firestore.collection(coleccion)
        self._db = cliente_firestore

    def _por_indice(self, indice: str):
        from google.cloud.firestore_v1.base_query import FieldFilter

        return self._col.where(filter=FieldFilter("indice", "==", indice))

    def leer_indice(self, indice: str) -> List[Vector]:
        return [Vector.desde_documento(d.id, d.to_dict() or {}) for d in self._por_indice(indice).stream()]

    def guardar(self, vectores: List[Vector]) -> None:
        # Un lote admite 500 escrituras; un alta trae unas cuantas.
        for i in range(0, len(vectores), 400):
            lote = self._db.batch()
            for v in vectores[i : i + 400]:
                lote.set(self._col.document(v.id), v.documento())
            lote.commit()

    def borrar_sujeto(self, indice: str, sujeto_id: str) -> int:
        from google.cloud.firestore_v1.base_query import FieldFilter

        docs = list(self._por_indice(indice).where(filter=FieldFilter("sujeto_id", "==", sujeto_id)).stream())
        for i in range(0, len(docs), 400):
            lote = self._db.batch()
            for d in docs[i : i + 400]:
                lote.delete(d.reference)
            lote.commit()
        return len(docs)
