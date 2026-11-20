"""Foto → cara → vector, con las correcciones de T-04.

Mismo modelo y misma preparación que la v1 (`app/core/`), para que los vectores
sigan siendo comparables con los de KAXTEN y con los del teléfono:
MTCNN → recorte → 160×160 → `keras_facenet.FaceNet().embeddings` (512).

TensorFlow se importa la primera vez que llega una foto, no al arrancar: las
búsquedas por vector (lo que usa la app) no lo necesitan, y las pruebas corren
sin él.
"""

import io
import threading
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

import numpy as np

from app.v2.errores import ErrorV2

LADO_MAXIMO = 1024
TAMANIO_CARA = 160


@dataclass(frozen=True)
class Cara:
    vector: List[float]
    caja: Tuple[int, int, int, int]  # x, y, ancho, alto, en la foto ya reducida
    confianza: float
    caras_detectadas: int


class MotorVision(Protocol):
    def cara_principal(self, contenido: bytes) -> Optional[Cara]: ...


def leer_imagen(contenido: bytes) -> np.ndarray:
    """Bytes → RGB, ya girada según el EXIF y con el lado mayor ≤ 1024 px.

    Sin el giro, una foto vertical del teléfono llega acostada y MTCNN no ve la
    cara. Reducirla acelera la detección sin perder nada útil para 160×160.
    """
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        img = Image.open(io.BytesIO(contenido))
        img = ImageOps.exif_transpose(img)
    except (UnidentifiedImageError, OSError):
        raise ErrorV2(400, "imagen_invalida", "El archivo no es una imagen que se pueda leer.") from None
    img = img.convert("RGB")
    img.thumbnail((LADO_MAXIMO, LADO_MAXIMO))
    return np.asarray(img)


def recortar(img: np.ndarray, caja: Tuple[int, int, int, int]) -> np.ndarray:
    """Recorte con los bordes dentro de la foto. La v1 usaba `abs(x)`, que con una
    caja que se sale por la izquierda (x = -12) recortaba en x = 12: otra cara."""
    import cv2

    x, y, w, h = caja
    alto, ancho = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(ancho, x + w), min(alto, y + h)
    if x1 <= x0 or y1 <= y0:
        raise ErrorV2(422, "sin_rostro", "La cara detectada queda fuera de la foto.")
    return cv2.resize(img[y0:y1, x0:x1], (TAMANIO_CARA, TAMANIO_CARA))


class MotorFaceNet:
    """MTCNN + FaceNet, cargados una sola vez y compartidos entre peticiones."""

    def __init__(self) -> None:
        self._detector = None
        self._facenet = None
        self._carga = threading.Lock()
        # TensorFlow no es seguro con varias inferencias a la vez sobre el mismo
        # modelo en todas las versiones; una a la vez es predecible.
        self._inferencia = threading.Lock()

    def _cargar(self) -> None:
        if self._facenet is not None:
            return
        with self._carga:
            if self._facenet is None:
                from keras_facenet import FaceNet
                from mtcnn import MTCNN

                self._detector = MTCNN()
                self._facenet = FaceNet()

    def cara_principal(self, contenido: bytes) -> Optional[Cara]:
        img = leer_imagen(contenido)
        self._cargar()
        with self._inferencia:
            detecciones = self._detector.detect_faces(img) or []
            if not detecciones:
                return None
            # La cara más grande es la persona retratada; las demás (gente al
            # fondo) se ignoran y se informa cuántas hubo.
            mejor = max(detecciones, key=lambda d: d["box"][2] * d["box"][3])
            cara = recortar(img, tuple(int(v) for v in mejor["box"]))
            emb = self._facenet.embeddings(np.expand_dims(cara.astype("float32"), axis=0))[0]
        return Cara(
            vector=[float(x) for x in emb],
            caja=tuple(int(v) for v in mejor["box"]),
            confianza=float(mejor.get("confidence", 0.0)),
            caras_detectadas=len(detecciones),
        )
