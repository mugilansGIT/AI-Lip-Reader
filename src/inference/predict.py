import torch
import numpy as np

from src.model.architecture import LipNet
from src.inference.decoder import greedy_decode


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


model = LipNet().to(DEVICE)

model.load_state_dict(
    torch.load(
        "models/lip_reader/lipnet.pth",
        map_location=DEVICE
    )
)

model.eval()


# Load sample
data = np.load(
    "data/processed/s1_processed/bbaf2n.npy"
)

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

    print("\nPrediction:")
    print(text)