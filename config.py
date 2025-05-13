import os
import json

FIREBASE_KEY_PATH = "firebase_key.json"

# Escribe el contenido a un archivo local si no existe
if not os.path.exists(FIREBASE_KEY_PATH):
    firebase_key_content = os.getenv("FIREBASE_KEY")
    if firebase_key_content:
        with open(FIREBASE_KEY_PATH, "w") as f:
            f.write(firebase_key_content)

FIREBASE_KEY = FIREBASE_KEY_PATH
