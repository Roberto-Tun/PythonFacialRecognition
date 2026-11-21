"""Índices en memoria y la búsqueda por coseno.

La v1 leía **toda** la colección `Vectores` en cada búsqueda. Aquí cada índice se
carga una vez en una matriz normalizada y se refresca cada `refresco_s` segundos
(los teléfonos escriben directo a Firestore, así que el servidor no se entera de
sus altas al instante) o en cuanto la v2 escribe en él.
"""

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from app.v2.config import DIMENSION, MODELOS_COMPATIBLES
from app.v2.repositorio import Repositorio, Vector


@dataclass
class _Cargado:
    momento: float
    ids: List[str]
    sujetos: List[str]
    grupos: List[str]
    rutas: List[str]
    matriz: np.ndarray  # (n, 512), filas de norma 1


@dataclass(frozen=True)
class Coincidencia:
    sujeto_id: str
    grupo: str
    similitud: float
    id_vector: str
    ruta_storage: str = ""


def normalizar(v: Sequence[float]) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32).reshape(-1)
    norma = float(np.linalg.norm(arr))
    return arr / norma if norma > 0 else arr


def vector_valido(v: Sequence[float]) -> bool:
    if len(v) != DIMENSION:
        return False
    try:
        return all(math.isfinite(float(x)) for x in v) and any(float(x) != 0.0 for x in v)
    except (TypeError, ValueError):
        return False


class Indices:
    def __init__(self, repo: Repositorio, refresco_s: float, reloj: Callable[[], float] = time.monotonic):
        self._repo = repo
        self._refresco = refresco_s
        self._reloj = reloj
        self._cache: Dict[str, _Cargado] = {}
        self._candado = threading.Lock()

    def invalidar(self, indice: str) -> None:
        with self._candado:
            self._cache.pop(indice, None)

    def _obtener(self, indice: str) -> _Cargado:
        with self._candado:
            c = self._cache.get(indice)
            if c is not None and self._reloj() - c.momento < self._refresco:
                return c
        vectores = [
            v for v in self._repo.leer_indice(indice) if v.modelo in MODELOS_COMPATIBLES and vector_valido(v.vector)
        ]
        matriz = (
            np.vstack([normalizar(v.vector) for v in vectores])
            if vectores
            else np.zeros((0, DIMENSION), dtype=np.float32)
        )
        c = _Cargado(
            momento=self._reloj(),
            ids=[v.id for v in vectores],
            sujetos=[v.sujeto_id for v in vectores],
            grupos=[v.grupo for v in vectores],
            rutas=[v.ruta_storage for v in vectores],
            matriz=matriz,
        )
        with self._candado:
            self._cache[indice] = c
        return c

    def tamanio(self, indice: str) -> int:
        return len(self._obtener(indice).ids)

    def buscar(
        self,
        indice: str,
        consultas: List[Sequence[float]],
        limite: int,
        umbral: float,
        grupo: Optional[str] = None,
    ) -> "ResultadoBusqueda":
        c = self._obtener(indice)
        filas = np.arange(len(c.ids))
        if grupo is not None:
            filas = np.array([i for i in filas if c.grupos[i] == grupo], dtype=int)
        if len(filas) == 0 or not consultas:
            return ResultadoBusqueda([], len(filas))

        q = np.vstack([normalizar(v) for v in consultas])  # (k, 512)
        # Similitud de cada vector guardado con la mejor de las fotos de consulta.
        sims = (c.matriz[filas] @ q.T).max(axis=1)

        # Una fila por sujeto: su vector más parecido. Un socio con 3 fotos no
        # debe ocupar 3 lugares del resultado.
        mejores: Dict[str, Coincidencia] = {}
        for pos, sim in zip(filas, sims):
            s = float(sim)
            if s < umbral:
                continue
            sujeto = c.sujetos[pos]
            previo = mejores.get(sujeto)
            if previo is None or s > previo.similitud:
                mejores[sujeto] = Coincidencia(sujeto, c.grupos[pos], s, c.ids[pos], c.rutas[pos])
        orden = sorted(mejores.values(), key=lambda m: m.similitud, reverse=True)
        return ResultadoBusqueda(orden[:limite], len(filas))


@dataclass(frozen=True)
class ResultadoBusqueda:
    coincidencias: List[Coincidencia]
    comparados: int


__all__ = ["Indices", "Coincidencia", "ResultadoBusqueda", "normalizar", "vector_valido", "Vector"]
