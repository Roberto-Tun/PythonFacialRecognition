"""Inicialización única de Firebase, sin llave dentro de la imagen.

Orden de preferencia:
1. Si `config.FIREBASE_KEY` apunta a un archivo que existe, se usa ese archivo.
   Es como Saul prueba la v1 en su máquina, y como estaba la imagen de producción.
2. Si no, las credenciales del entorno (ADC): la cuenta de servicio en Cloud Run,
   `gcloud auth application-default login` en local, o un archivo montado por
   volumen y apuntado con `GOOGLE_APPLICATION_CREDENTIALS`.
"""

import os

import firebase_admin
from firebase_admin import credentials

BUCKET = "kaxten-2090b.firebasestorage.app"


def _ruta_llave_local():
    try:
        from config import FIREBASE_KEY  # type: ignore
    except Exception:
        return None
    if FIREBASE_KEY and os.path.isfile(FIREBASE_KEY):
        return FIREBASE_KEY
    return None


def inicializar_firebase() -> firebase_admin.App:
    """Devuelve la app de Firebase, creándola la primera vez."""
    try:
        return firebase_admin.get_app()
    except ValueError:
        pass

    ruta = _ruta_llave_local()
    if ruta:
        # La librería de Storage lee esta variable por su cuenta.
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", ruta)
        cred = credentials.Certificate(ruta)
    else:
        cred = credentials.ApplicationDefault()
    return firebase_admin.initialize_app(cred, {"storageBucket": BUCKET})
