import torch


def _accuracy(prediction, target):
    prediction = prediction.detach().cpu()
    target = target.detach().cpu()
    predicted_labels = (prediction >= 0.5).type(torch.float)
    correct = (predicted_labels == target).type(torch.float)
    return correct.sum().item() / correct.shape[0]


def train_discriminator(
    discriminator,
    generator,
    optimizer,
    criterion,
    real_data,
    real_labels,
    fake_data,
    fake_labels,
):
    """
    Trains the discriminator on real and fake data.
    real_data and fake_data are batches of real and fake images respectively.
    real_labels and fake_labels are the corresponding labels for the images.
    Returns the loss and accuracy for real and fake data respectively.
    real_data: (64, channels, height, width)
    real_labels: (64, 1)
    fake_data: (64, laten_dim)
    fake_labels: (64, 1)
    """
    optimizer.zero_grad()

    half_size = real_data.size(0) // 2
    real_data_half = real_data[:half_size, :, :, :]
    real_labels_half = real_labels[:half_size, :]
    fake_data_half = fake_data[:half_size, :]
    fake_labels_half = fake_labels[:half_size, :]

    # generate fake images
    fake_images_half = generator(fake_data_half)

    # mix data
    mixed_data = torch.cat((real_data_half, fake_images_half), dim=0)
    mixed_labels = torch.cat((real_labels_half, fake_labels_half), dim=0)

    # shuffle data
    indices = torch.randperm(mixed_data.size(0))
    mixed_data = mixed_data[indices]
    mixed_labels = mixed_labels[indices]

    # forward pass
    output_mixed = discriminator(mixed_data)

    # compute loss and accuracy
    loss = criterion(output_mixed, mixed_labels)
    acc = _accuracy(output_mixed, mixed_labels)
    # update weights
    loss.backward()
    optimizer.step()

    return loss.item(), acc

def train_generator(
    generator,
    discriminator,
    optimizer,
    criterion,
    real_data,
    real_labels,
    fake_data,
    fake_labels,
):
    optimizer.zero_grad()

    fake_images = generator(fake_data)
    output_fake = discriminator(fake_images)
    loss = criterion(output_fake, real_labels)
    acc = _accuracy(output_fake, real_labels)

    loss.backward()
    optimizer.step()

    return loss.item(), acc


def validate(generator, discriminator, criterion, noise):
    fake_images = generator(noise)
    output_fake = discriminator(fake_images)
    loss = criterion(output_fake, torch.ones(output_fake.size()))
    return loss.item()

def generate_images(generator, noise):
    with torch.no_grad():
        fake_images = generator(noise)
    return fake_images