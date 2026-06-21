import tempfile
import numpy as np
import torch

from src.model.architecture import LipNet
from src.inference.decoder import greedy_decode
from src.preprocessing.frame_pipeline import process_video


# =========================
# DEVICE
# =========================
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", DEVICE)


# =========================
# LOAD MODEL
# =========================
model = LipNet().to(DEVICE)

model.load_state_dict(
    torch.load(
        "models/lip_reader/best_model.pth",
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded.")


# =========================
# SLIDING WINDOW PREDICTION
# =========================
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
    # load processed lips
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
            f"Adjusted window size "
            f"to {window_size}"
        )

    # -------------------------
    # sliding windows
    # -------------------------
    for start in range(
        0,
        max(1, total_frames - window_size + 1),
        stride
    ):

        end = start + window_size

        clip = data[start:end]

        # Skip empty clips
        if clip.shape[0] == 0:
            continue

        # -------------------------
        # tensor conversion
        # -------------------------
        x = torch.tensor(
            clip,
            dtype=torch.float32
        )

        # shape:
        # (1, T, 1, H, W)
        x = x.unsqueeze(0).unsqueeze(2)

        x = x.to(DEVICE)

        # -------------------------
        # model inference
        # -------------------------
        with torch.no_grad():

            output = model(x)

            output = output[0]

            prediction = greedy_decode(output)

        # -------------------------
        # timestamps
        # -------------------------
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