import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        """
        x: noise tensor with shape (batch_size, latent_dim)
        returns: generated image tensor with shape (batch_size, 1*28*28)
        """
        x = self.model(x)
        x = x.view(x.shape[0], 1, 28, 28)
        return x


if __name__ == "__main__":
    noise = torch.normal(size=(4,100), mean=0.0, std=1.0)
    generator = Generator(latent_dim=100)
    out = generator(noise)
    print(out.shape)
