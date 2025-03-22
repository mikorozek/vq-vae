import torch
from torch.utils.data import DataLoader, random_split
import wandb
from model import VQVAE
from datasets import VCTKFixed
from train import train_vqvae


dataset = VCTKFixed(root="/home/mrozek/VCTK-processed", download=False, sample_rate=48000, seconds=4)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

wandb.login()

vqvae_model = VQVAE(in_channels=1, hidden_dim=128, codebook_size=1024, embedding_dim=512, commitment_cost=0.25)

model, train_losses, recon_losses, vq_losses, val_losses = train_vqvae(
    model=vqvae_model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    learning_rate=1e-3,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_path='./vqvae_models',
    project_name="vqvae-vctk",
    run_name="vctk_experiment",
    sample_rate=48000
)
