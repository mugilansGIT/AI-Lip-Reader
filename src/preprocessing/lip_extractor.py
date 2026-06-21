import cv2
import numpy as np

def preprocess_lips(lip_frame):
    gray = cv2.cvtColor(lip_frame, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, (100, 50))

    normalized = resized / 255.0

    return normalized