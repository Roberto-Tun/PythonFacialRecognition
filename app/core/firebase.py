from firebase_admin import credentials, initialize_app, firestore, storage
from config import FIREBASE_KEY
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = FIREBASE_KEY

cred = credentials.Certificate(FIREBASE_KEY)
default_app = initialize_app(cred, {
    'storageBucket': 'kaxten-2090b.firebasestorage.app'  # ¡IMPORTANTE!
})

db = firestore.client()
bucket = storage.bucket()
