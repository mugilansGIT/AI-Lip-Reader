import torch

from src.utils.text_utils import idx_to_char


def greedy_decode(output):

    # output shape:
    # (T, C)

    pred = torch.argmax(output, dim=1)

    pred = pred.cpu().numpy().tolist()

    decoded = []

    previous = None

    for p in pred:

        # Skip blanks
        if p == 0:
            previous = p
            continue

        # Remove duplicates
        if p == previous:
            continue

        decoded.append(p)

        previous = p

    text = "".join([
        idx_to_char[i]
        for i in decoded
        if i in idx_to_char
    ])

    return text