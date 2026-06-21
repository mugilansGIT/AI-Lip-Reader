import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from src.model.dataset import GRIDDataset
from src.model.collate import collate_fn
from src.model.architecture import LipNet

# =========================
# DEVICE
# =========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# =========================
# DATASET
# =========================

dataset = GRIDDataset("data/processed")

# =========================
# TRAIN ON 10000 SAMPLES
# =========================

subset = dataset

print("Training samples:", len(subset))

# =========================
# DATALOADER
# =========================

loader = DataLoader(
    subset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=True
)

# =========================
# MODEL
# =========================

model = LipNet().to(DEVICE)

# =========================
# LOAD EXISTING MODEL
# =========================

checkpoint = "models/lip_reader/best_model.pth"

if os.path.exists(checkpoint):

    print("Loading existing model...")

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=DEVICE
        )
    )

    print("Existing model loaded.")

else:
    print("No checkpoint found. Training from scratch.")

# =========================
# LOSS
# =========================

criterion = nn.CTCLoss(
    blank=0,
    zero_infinity=True
)

# =========================
# OPTIMIZER
# =========================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

# =========================
# CHECKPOINT DIRECTORY
# =========================

os.makedirs(
    "models/lip_reader",
    exist_ok=True
)

# =========================
# TRAINING CONFIG
# =========================

EPOCHS = 10

best_loss = float("inf")

# =========================
# TRAINING LOOP
# =========================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch_idx, batch in enumerate(loader):

        lips, labels, input_lengths, label_lengths = batch

        lips = lips.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(lips)

        outputs = outputs.permute(1, 0, 2)

        loss = criterion(
            outputs,
            labels,
            input_lengths,
            label_lengths
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:

            print(
                f"Epoch [{epoch+1}/{EPOCHS}] | "
                f"Batch [{batch_idx}/{len(loader)}] | "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / len(loader)

    print("=" * 50)
    print(f"Epoch {epoch+1} Completed")
    print(f"Average Loss: {avg_loss:.4f}")
    print("=" * 50)

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            "models/lip_reader/best_model.pth"
        )

        print("Best model saved.")

# =========================
# SAVE FINAL MODEL
# =========================

torch.save(
    model.state_dict(),
    "models/lip_reader/final_model.pth"
)

print("Training completed.")
print("Final model saved.")