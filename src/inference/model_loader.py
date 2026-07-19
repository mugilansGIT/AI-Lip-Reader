import torch

from src.model.architecture import LipNet

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", DEVICE)

_model = None


def get_model():
    global _model

    if _model is None:

        print("Loading LipNet model...")

        _model = LipNet().to(DEVICE)

        _model.load_state_dict(
            torch.load(
                "models/lip_reader/best_model.pth",
                map_location=DEVICE
            )
        )

        _model.eval()

        print("Model loaded.")

    return _model


def get_device():
    return DEVICE