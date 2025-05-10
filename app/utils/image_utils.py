import numpy as np
import cv2

def read_image(file_bytes):
    img = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img