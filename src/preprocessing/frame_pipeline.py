import os
import numpy as np

from src.utils.video_utils import read_video
from src.preprocessing.face_detector import LipDetector
from src.preprocessing.lip_extractor import preprocess_lips

detector = LipDetector()

def process_video(video_path, output_path):
    frames = read_video(video_path)

    processed = []

    for frame in frames:
        lips = detector.extract_lips(frame)

        if lips is None:
            continue

        lips = preprocess_lips(lips)

        processed.append(lips)

    processed = np.array(processed)

    np.save(output_path, processed)

    print("Saved:", output_path)