import torch

from src.inference.decoder import greedy_decode
from src.inference.model_loader import (
    get_model,
    get_device
)


def predict_array(data):
    """
    Predict text from a processed numpy array.

    Parameters
    ----------
    data : numpy.ndarray
        Shape: (T, 50, 100)

    Returns
    -------
    str
        Predicted sentence.
    """

    model = get_model()
    device = get_device()

    x = torch.tensor(
        data,
        dtype=torch.float32
    )

    # (T,50,100)
    # ->
    # (1,T,1,50,100)

    x = x.unsqueeze(0).unsqueeze(2)

    x = x.to(device)

    with torch.no_grad():

        output = model(x)

        output = output[0]

        text = greedy_decode(output)

    return text