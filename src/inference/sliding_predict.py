import tempfile
import numpy as np

from src.preprocessing.frame_pipeline import process_video
from src.inference.predict_core import predict_array


def sliding_predict(
    video_path,
    window_size=75,
    stride=25,
    fps=25
):

    # -------------------------
    # preprocess video
    # -------------------------
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".npy",
        delete=False
    )

    npy_path = temp_file.name

    process_video(
        video_path,
        npy_path
    )

    print(f"Saved processed file: {npy_path}")

    # -------------------------
    # load processed data
    # -------------------------
    data = np.load(npy_path)

    total_frames = data.shape[0]

    print("Total frames:", total_frames)

    subtitles = []

    # -------------------------
    # handle short videos
    # -------------------------
    if total_frames < window_size:

        window_size = total_frames

        print(
            f"Adjusted window size to {window_size}"
        )

    # -------------------------
    # sliding window prediction
    # -------------------------
    for start in range(
        0,
        max(1, total_frames - window_size + 1),
        stride
    ):

        end = start + window_size

        clip = data[start:end]

        if len(clip) == 0:
            continue

        prediction = predict_array(clip)

        start_time = start / fps
        end_time = end / fps

        subtitles.append({
            "start": start_time,
            "end": end_time,
            "text": prediction
        })

        print(
            f"[{start_time:.2f}s - "
            f"{end_time:.2f}s] "
            f"{prediction}"
        )

    return subtitles