import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        x = x.view(x.shape[0], -1)
        x = self.model(x)
        return x


if __name__ == "__main__":
    x = torch.randn(4, 1, 28, 28)
    model = Discriminator()
    y = model(x)
    print(y.shape)
