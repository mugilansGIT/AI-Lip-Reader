import tempfile
import numpy as np

from src.preprocessing.frame_pipeline import process_video
from src.inference.predict_core import predict_array


def predict_video(video_path):

    # Create temporary npy file
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".npy",
        delete=False
    )

    npy_path = temp_file.name

    # Preprocess video
    process_video(
        video_path,
        npy_path
    )

    # Load processed frames
    data = np.load(npy_path)

    # Predict
    prediction = predict_array(data)

    return prediction