from firebase_admin import firestore, storage

from app.core.credenciales import inicializar_firebase

# v1 (KAXTEN): base `(default)` y el bucket compartido. Mismo comportamiento que
# antes; solo cambia de dónde salen las credenciales (ver credenciales.py).
default_app = inicializar_firebase()

db = firestore.client()
bucket = storage.bucket()
