import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):

    lips, labels = zip(*batch)

    input_lengths = torch.tensor(
        [x.shape[0] for x in lips],
        dtype=torch.long
    )

    label_lengths = torch.tensor(
        [len(x) for x in labels],
        dtype=torch.long
    )

    lips = pad_sequence(
        lips,
        batch_first=True
    )

    labels = torch.cat(labels)

    return (
        lips,
        labels,
        input_lengths,
        label_lengths
    )