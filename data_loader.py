import torch
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST

training_transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])


class mnist_data_loader(Dataset):
    def __init__(self, real_images, laten_dim):
        self.real_images = real_images
        self.laten_dim = laten_dim

    def __len__(self):
        return len(self.real_images)

    def __getitem__(self, index):
        # real_image
        real_image, label = self.real_images[index]

        # label_real == 1
        label_real = torch.ones(1)

        # fake_image
        laten_noise = torch.normal(mean=0, std=1, size=(self.laten_dim,))

        # label_fake == 0
        label_fake = torch.zeros(1)
        return real_image, label_real, laten_noise, label_fake


def get_data_loader(batch_size, laten_dim):
    train_mnist = MNIST(root="./data", train=True, transform=training_transform, download=True)
    val_mnist = MNIST(root="./data", train=False, transform=training_transform, download=True)

    train_dataset = mnist_data_loader(train_mnist, laten_dim)
    val_dataset = mnist_data_loader(val_mnist, laten_dim)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_data_loader(batch_size=128, laten_dim=100)
    for i, data in enumerate(train_loader):
        real_image, label_real, laten_noise, label_fake = data
        print(label_real)
        break
