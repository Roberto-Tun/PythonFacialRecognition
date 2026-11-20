"""Entradas y salidas en JSON de la v2."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AltaPorVectores(BaseModel):
    modelo: str = Field(..., description="facenet_512 o facenet_512_fp16")
    vectores: List[List[float]] = Field(..., min_length=1, max_length=10)
    grupo: str = ""


class BusquedaPorVector(BaseModel):
    modelo: str
    vectores: List[List[float]] = Field(..., min_length=1, max_length=5)
    limite: Optional[int] = None
    umbral: Optional[float] = None
    grupo: Optional[str] = None


class CoincidenciaSalida(BaseModel):
    sujeto_id: str
    grupo: str
    similitud: float = Field(..., description="Coseno entre 0 y 1")
    id_vector: str


class ResultadoBusquedaSalida(BaseModel):
    indice: str
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
