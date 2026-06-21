import torch

from src.model.architecture import LipNet

model = LipNet()

x = torch.randn(
    2,
    75,
    1,
    50,
    100
)

out = model(x)

print(out.shape)