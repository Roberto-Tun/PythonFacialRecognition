import firebase_admin
from firebase_admin import credentials, firestore
from config import FIREBASE_KEY

cred = credentials.Certificate(FIREBASE_KEY)
firebase_admin.initialize_app(cred)
db = firestore.client()
