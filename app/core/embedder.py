import numpy as np
from keras_facenet import FaceNet

embedder = FaceNet()

def get_embedding(face):
    face = face.astype('float32')
    face = np.expand_dims(face, axis=0)
    return embedder.embeddings(face)[0]