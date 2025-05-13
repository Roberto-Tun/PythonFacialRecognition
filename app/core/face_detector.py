import cv2
from mtcnn import MTCNN

mtcnn = MTCNN()

def detect_faces(img):
    faces = []
    detections = mtcnn.detect_faces(img)
    for det in detections:
        x, y, w, h = det['box']
        x, y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        faces.append(face)
    return faces