import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.dataset import load_dataset
from src.vae import VAE, vae_loss

def preprocess_data(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return torch.tensor(df_scaled, dtype=torch.float32)

def train_vae_boston():
    dataset_name = "boston_housing"
    print(f"\n🔹 Treinando VAE no dataset: {dataset_name}")

    # Carregar e preprocessar os dados
    df = load_dataset(dataset_name)
    X = preprocess_data(df)

    # Hiperparâmetros específicos do Boston Housing
    LATENT_DIM = 32
    DROPOUT = 0.2
    LEARNING_RATE = 0.00001

    # Criar DataLoader para treino
    train_loader = data.DataLoader(X, batch_size=32, shuffle=True)

    # Definir modelo e otimizador
    input_dim = X.shape[1]
    model = VAE(input_dim, latent_dim=LATENT_DIM, dropout=DROPOUT)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Treinamento
    model.train()
    for epoch in range(2500):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch)
            loss = vae_loss(recon_x, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 50 == 0:
            print(f" Epoch [{epoch+1}/3000], Loss: {total_loss:.2f}")

    # Salvar modelo treinado
    model_path = f"models/vae_{dataset_name}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n Modelo salvo em: {model_path}")

if __name__ == "__main__":
    train_vae_boston()
