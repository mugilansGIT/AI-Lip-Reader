import torch
import tempfile

from src.model.architecture import LipNet
from src.inference.decoder import greedy_decode

from src.preprocessing.frame_pipeline import process_video


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# Load model
model = LipNet().to(DEVICE)

model.load_state_dict(
    torch.load(
        "models/lip_reader/best_model.pth",
        map_location=DEVICE
    )
)

model.eval()


def predict_video(video_path):

    # Temporary npy output
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".npy",
        delete=False
    )

    output_path = temp_file.name

    # Preprocess video
    process_video(
        video_path,
        output_path
    )

    # Load processed lips
    import numpy as np

    data = np.load(output_path)

    x = torch.tensor(
        data,
        dtype=torch.float32
    )

    x = x.unsqueeze(0).unsqueeze(2)

    x = x.to(DEVICE)

    with torch.no_grad():

        output = model(x)

        output = output[0]

        text = greedy_decode(output)

    return text


if __name__ == "__main__":

    video_path = "data/raw_videos/s1_processed/video/mpg_6000/bbaf2n.mpg"

    prediction = predict_video(video_path)

    print("\nPrediction:")
    print(prediction)