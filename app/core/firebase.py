import firebase_admin
from firebase_admin import credentials, firestore
from config import API_KEY

cred = credentials.Certificate(API_KEY)
firebase_admin.initialize_app(cred)
db = firestore.client()
