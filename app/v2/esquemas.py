"""Entradas y salidas en JSON de la v2.

Los valores por defecto y los ejemplos importan: la página `/docs` arma el cuerpo
de ejemplo con ellos. Sin ellos ponía `0` en umbral y límite (devolvía a
cualquiera) y `"string"` en grupo y modelo (no encontraba nada o fallaba).
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

LIMITE_DEFECTO = 10
UMBRAL_DEFECTO = 0.5
MODELO_DEFECTO = "facenet_512"

_EJEMPLO_VECTOR = "[[0.012, -0.034, …512 números…]]"


class AltaPorVectores(BaseModel):
    modelo: str = Field(MODELO_DEFECTO, description="facenet_512 (servidor) o facenet_512_fp16 (teléfono)")
    vectores: List[List[float]] = Field(..., min_length=1, max_length=10, description=f"Ejemplo: {_EJEMPLO_VECTOR}")
    grupo: str = Field("", description="Subconjunto dentro del índice (p. ej. el negocio). Vacío si no aplica.")

    model_config = {"json_schema_extra": {"examples": [{"modelo": MODELO_DEFECTO, "vectores": [[0.0]], "grupo": ""}]}}


class BusquedaPorVector(BaseModel):
    modelo: str = Field(MODELO_DEFECTO, description="facenet_512 (servidor) o facenet_512_fp16 (teléfono)")
    vectores: List[List[float]] = Field(..., min_length=1, max_length=5, description=f"Ejemplo: {_EJEMPLO_VECTOR}")
    limite: int = Field(LIMITE_DEFECTO, ge=1, description="Máximo de personas en la respuesta")
    umbral: float = Field(UMBRAL_DEFECTO, ge=0, le=1, description="Similitud mínima (0 a 1)")
    grupo: Optional[str] = Field(None, description="Solo ese grupo. Vacío o ausente: todos")

    model_config = {
        "json_schema_extra": {
            "examples": [{"modelo": MODELO_DEFECTO, "vectores": [[0.0]], "limite": LIMITE_DEFECTO, "umbral": UMBRAL_DEFECTO}]
        }
    }


class CoincidenciaSalida(BaseModel):
    indice: str
    sujeto_id: str
    grupo: str
    similitud: float = Field(..., description="Coseno entre 0 y 1")
    id_vector: str
    foto_url: Optional[str] = Field(None, description="Foto pública del sujeto, si la hay (Registros y KAXTEN; los socios no guardan foto)")
    datos: Optional[Dict[str, Any]] = Field(None, description="Documento del sujeto (socio, Registro o persona de KAXTEN)")


class ResultadoBusquedaSalida(BaseModel):
    indice: str = Field(..., description="El índice buscado, o varios separados por coma si no se indicó")
    coincidencias: List[CoincidenciaSalida]
    comparados: int
    umbral: float
    limite: int
    sin_rostro: List[int] = []
    tiempos_ms: Dict[str, float]


class AltaSalida(BaseModel):
    indice: str
    sujeto_id: str
    guardados: List[str]
    sin_rostro: List[int] = []
    caras_ignoradas: int = 0
