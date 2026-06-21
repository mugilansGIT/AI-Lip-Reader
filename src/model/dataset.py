import os
import glob
import torch
import numpy as np

from torch.utils.data import Dataset

from src.utils.text_utils import (
    parse_alignment,
    text_to_int
)


class GRIDDataset(Dataset):

    def __init__(self, processed_root):

        self.samples = []

        npy_files = glob.glob(
            os.path.join(
                processed_root,
                "**",
                "*.npy"
            ),
            recursive=True
        )

        print("Found npy files:", len(npy_files))

        for npy_path in npy_files:

            speaker_folder = os.path.basename(
                os.path.dirname(npy_path)
            )

            filename = os.path.basename(npy_path)

            base = filename.replace(".npy", "")

            align_path = os.path.join(
                "data",
                "raw_videos",
                speaker_folder,
                "align",
                base + ".align"
            )

            if os.path.exists(align_path):

                self.samples.append(
                    (npy_path, align_path)
                )

        print("Loaded samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        npy_path, align_path = self.samples[idx]

        lips = np.load(npy_path)

        text = parse_alignment(align_path)

        label = text_to_int(text)

        lips = torch.tensor(
            lips,
            dtype=torch.float32
        )

        # (T, 1, H, W)
        lips = lips.unsqueeze(1)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return lips, label