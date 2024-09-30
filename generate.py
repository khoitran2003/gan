from tasks import generate_images
from models.generator import Generator
import torch
import matplotlib.pyplot as plt


def main(num_images):
    """
    Generate images using the generator model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the generator model
    latent_dim = 100
    generator = Generator(latent_dim=latent_dim).to(device)
    generator.load_state_dict(
        torch.load("results/checkpoints/gen_best.pt", map_location=device)[
            "generator_state_dict"
        ]
    )

    # Generate noise with the same number as num_images
    noise = torch.randn(num_images, latent_dim).to(device)

    # Generate images using the generator
    results = generate_images(generator, noise)

    # Scale the images from [-1, 1] to [0, 1] for visualization
    results = (results + 1) / 2

    # If the images are grayscale (1 channel), remove the channel for plotting
    if results.shape[1] == 1:
        results = results.squeeze(1)  # Remove the channel dimension for grayscale images

    # Plot the generated images
    fig, axes = plt.subplots(1, num_images, figsize=(10, 10))
    for i in range(num_images):
        axes[i].imshow(results[i].cpu().detach().numpy(), cmap='gray')
        axes[i].axis("off")
    plt.show()


if __name__ == "__main__":
    num_images = 10  # Số lượng ảnh bạn muốn generate
    main(num_images)
