import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("facerecognitionapp-a1ca7-firebase-adminsdk-fbsvc-4282f64066.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
