import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.dataset import load_dataset
from src.vae import VAE

LATENT_DIM = 32 # Ajustamos para uma melhor representação latente

def load_trained_vae(dataset_name, input_dim):
    """ Carrega o modelo treinado """
    model = VAE(input_dim, latent_dim=LATENT_DIM)
    model_path = f"models/vae_{dataset_name}.pt"

    # Verifica se o modelo existe antes de tentar carregar
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo {model_path} não encontrado. Treine primeiro com train_boston.py!")

    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def plot_latent_space(dataset_name):
    """ Gera e plota a projeção do espaço latente do VAE """
    print(f"\n Analisando espaço latente para: {dataset_name}")

    # Carregar dataset (mantendo todas as colunas)
    df = load_dataset(dataset_name)

    # Usa todas as colunas para X (incluindo a última, para manter 14 colunas)
    X = df.iloc[:, :]  # Pegamos todas as colunas

    # Para regressão, normalizamos y para evitar uma única cor no gráfico
    y = (df.iloc[:, -1] - df.iloc[:, -1].min()) / (df.iloc[:, -1].max() - df.iloc[:, -1].min())

    print(f"Shape de X: {X.shape}, Valores únicos de y: {np.unique(y)}")

    # Normalização Min-Max das features
    X = (X - X.min()) / (X.max() - X.min())
    X = torch.tensor(X.values, dtype=torch.float32)

    # Carregar modelo treinado
    model = load_trained_vae(dataset_name, input_dim=X.shape[1])

    # Passar dados pelo encoder do VAE
    with torch.no_grad():
        mu, _ = model.encode(X)

    mu = mu.numpy()  # Converter para NumPy

    # Se o espaço latente for maior que 2D, aplicar PCA para reduzir para 2D
    if mu.shape[1] > 2:
        pca = PCA(n_components=2)
        mu = pca.fit_transform(mu)
        print(f" Variância explicada pelos 2 primeiros componentes: {sum(pca.explained_variance_ratio_):.2f}")

    # Plotar os pontos latentes coloridos pelos rótulos
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(mu[:, 0], mu[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.xlabel("Dimensão 1 do Latente")
    plt.ylabel("Dimensão 2 do Latente")
    plt.title(f"Projeção 2D do Espaço Latente - {dataset_name}")
    plt.colorbar(scatter, label="Classe/Rótulo Real")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_latent_space("boston_housing")
