import torch
import torch.nn as nn


class LipNet(nn.Module):

    def __init__(self, vocab_size=29):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv3d(
                1,
                32,
                kernel_size=(3, 5, 5),
                padding=(1, 2, 2)
            ),

            nn.ReLU(),

            nn.MaxPool3d(
                (1, 2, 2)
            ),

            nn.Conv3d(
                32,
                64,
                kernel_size=(3, 5, 5),
                padding=(1, 2, 2)
            ),

            nn.ReLU(),

            nn.MaxPool3d(
                (1, 2, 2)
            )
        )

        self.gru = nn.GRU(
            input_size=64 * 12 * 25,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(
            512,
            vocab_size
        )

    def forward(self, x):

        # x shape:
        # (B, T, 1, 50, 100)

        x = x.permute(0, 2, 1, 3, 4)

        x = self.conv(x)

        B, C, T, H, W = x.shape

        x = x.permute(0, 2, 1, 3, 4)

        x = x.reshape(
            B,
            T,
            C * H * W
        )

        x, _ = self.gru(x)

        x = self.fc(x)

        return x.log_softmax(2)