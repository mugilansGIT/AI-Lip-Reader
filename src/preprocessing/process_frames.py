import numpy as np

from src.preprocessing.face_detector import LipDetector
from src.preprocessing.lip_extractor import preprocess_lips

detector = LipDetector()


def process_frames(frames):
    """
    Process webcam frames into the format
    expected by the LipNet model.

    Parameters
    ----------
    frames : numpy.ndarray
        Shape: (T, H, W, 3)

    Returns
    -------
    numpy.ndarray
        Shape: (T, 50, 100)
    """

    processed = []

    for frame in frames:

        lips = detector.extract_lips(frame)

        if lips is None:
            continue

        lips = preprocess_lips(lips)

        processed.append(lips)

    return np.array(processed)