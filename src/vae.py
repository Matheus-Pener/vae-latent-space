import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32, dropout=0.2):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3_mu = nn.Linear(64, latent_dim)
        self.fc3_logvar = nn.Linear(64, latent_dim)

        # Decoder
        self.fc4 = nn.Linear(latent_dim, 64)
        self.fc5 = nn.Linear(64, 128)
        self.fc6 = nn.Linear(128, input_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(F.relu(self.fc2(h)))
        mu = self.fc3_mu(h)
        logvar = self.fc3_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dropout(F.relu(self.fc4(z)))
        h = self.dropout(F.relu(self.fc5(h)))
        return self.fc6(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# 🔹 Adicione esta função para evitar o erro no `train.py`
def vae_loss(recon_x, x, mu, logvar):
    """ Função de perda do VAE: Erro de reconstrução + KL Divergence """
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div
