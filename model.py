import torch
import torch.nn as nn
import torch.nn.functional as F

from vector_quantizer import VectorQuantizer

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, embedding_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, embedding_dim, kernel_size=4, stride=2, padding=1),
        )
    
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, out_channels):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(embedding_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, out_channels, kernel_size=4, stride=2, padding=1)
        )
    
    def forward(self, x):
        return self.decoder(x)

class VQVAE(nn.Module):
    def __init__(self, in_channels, hidden_dim, codebook_size, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()

        self.encoder = Encoder(in_channels, hidden_dim, embedding_dim)
        self.vq = VectorQuantizer(embedding_dim, codebook_size, commitment_cost)
        self.decoder = Decoder(embedding_dim, hidden_dim, in_channels)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, indices = self.vq(z)
        x_recon = self.decoder(quantized)

        recon_loss = F.mse_loss(x_recon, x)


        loss = recon_loss + vq_loss

        return x_recon, loss, vq_loss, recon_loss, indices
