import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import argparse
import shutil
import os

from models.discriminator import Discriminator
from models.generator import Generator
from data_loader import get_data_loader
from tasks import train_discriminator, train_generator, validate


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--latent_dim", type=int, default=100, help="latent dimension")
    parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs")
    parser.add_argument(
        "--k", type=int, default=3, help="frequency of train discriminator"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./results/training_logs",
        help="directory to save logs",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="./results/checkpoints",
        help="path to latest checkpoint (default: None)",
    )
    return parser.parse_args()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_val_loss = 10e6

    fixed_noise = torch.normal(
        mean=0, std=1, size=(args.batch_size, args.latent_dim)
    ).to(device)
    # Load data
    train_loader, val_loader = get_data_loader(args.batch_size, args.latent_dim)

    # check log
    if os.path.exists(os.path.join(args.log_dir, "experiment")):
        shutil.rmtree(os.path.join(args.log_dir, "experiment"))
        writer = SummaryWriter(os.path.join(args.log_dir, "experiment"))
    else:
        os.makedirs(os.path.join(args.log_dir, "experiment"))

    # check model
    generator = Generator(latent_dim=args.latent_dim).to(device)
    discriminator = Discriminator().to(device)

    # check optimizer
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # loss function
    criterion = nn.BCELoss()

    # check checkpoints
    gen_last = os.path.join(args.resume, "gen_last.pt")
    gen_best = os.path.join(args.resume, "gen_best.pt")
    dis_last = os.path.join(args.resume, "dis_last.pt")
    dis_best = os.path.join(args.resume, "dis_best.pt")
    try:
        # load generator
        gen_checkpoint = torch.load(gen_last, map_location=device)
        generator.load_state_dict(gen_checkpoint["generator_state_dict"])
        optimizer_G.load_state_dict(gen_checkpoint["optimizer_G_state_dict"])
        loss_G = gen_checkpoint["loss_G"]

        # load discriminator
        dis_checkpoint = torch.load(dis_last, map_location=device)
        discriminator.load_state_dict(dis_checkpoint["discriminator_state_dict"])
        optimizer_D.load_state_dict(dis_checkpoint["optimizer_D_state_dict"])
        loss_D = dis_checkpoint["loss_D"]

        start_epoch = gen_checkpoint["epoch"] + 1
        print("Generator loaded from last checkpoint")
    except FileNotFoundError:
        start_epoch = 0
        print("Generator initialized from scratch")

    # trainining loop
    for epoch in range(start_epoch, args.num_epochs):
        generator.train()
        discriminator.train()
        train_progress_bar = tqdm(train_loader, colour="red")
        for i, (data) in enumerate(train_progress_bar):
            real_images, real_labels, fake_images, fake_labels = data
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)
            fake_images = fake_images.to(device)
            fake_labels = fake_labels.to(device)
            if i % args.k == 0:
                # train discriminator
                loss_D, acc_D = train_discriminator(
                    discriminator,
                    generator,
                    optimizer_D,
                    criterion,
                    real_images,
                    real_labels,
                    fake_images,
                    fake_labels,
                )
                writer.add_scalar("Discriminator/real_loss", loss_D, i + epoch * len(train_loader))
                writer.add_scalar("Discriminator/real_acc", acc_D, i + epoch * len(train_loader))
            # train generator
            loss_G, acc_G = train_generator(
                generator, discriminator, optimizer_G, criterion, real_images, real_labels, fake_images, fake_labels
            )
            writer.add_scalar("Generator/loss", loss_G, i + epoch * len(train_loader))
            writer.add_scalar("Generator/acc", acc_G, i + epoch * len(train_loader))
            train_progress_bar.set_description(f"Epoch [{epoch+1}/{args.num_epochs}] | Loss D: {loss_D:.4f}, Loss G: {loss_G:.4f}, Acc D: {acc_D:.2f}, Acc G: {acc_G:.2f}")
            batch_done = i + epoch * len(train_loader)
            if batch_done % 400 == 0:
                save_image(
                    generator(fake_images),
                    os.path.join("results/generated_images", f"fake_images_{batch_done}.jpg"),
                    nrow=10,
                    normalize=True,
                )
        # validate
        generator.eval()
        discriminator.eval()
        val_loss = validate(generator, discriminator, criterion, fixed_noise)
        writer.add_scalar("Validation/loss", val_loss, epoch)
        print(f"Epoch [{epoch+1}/{args.num_epochs}] | Validation Loss: {val_loss:.4f}")

        # save last and best checkpoints
        torch.save(
            {
                "epoch": epoch,
                "generator_state_dict": generator.state_dict(),
                "optimizer_G_state_dict": optimizer_G.state_dict(),
                "loss_G": loss_G,
            },
            gen_last,
        )

        torch.save(
            {
                "epoch": epoch,
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_D_state_dict": optimizer_D.state_dict(),
                "loss_D": loss_D,
            },
            dis_last,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "generator_state_dict": generator.state_dict(),
                    "optimizer_G_state_dict": optimizer_G.state_dict(),
                    "loss_G": loss_G,
                },
                gen_best,
            )

            torch.save(
                {
                    "epoch": epoch,
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_D_state_dict": optimizer_D.state_dict(),
                    "loss_D": loss_D,
                },
                dis_best,
            )


if __name__ == "__main__":
    args = get_args()
    main(args)
