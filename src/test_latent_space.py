import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from src.dataset import load_dataset
from src.vae import VAE

LATENT_DIM = 32  

def load_trained_vae(dataset_name, input_dim):
    """ Carrega o modelo treinado """
    model = VAE(input_dim, latent_dim=LATENT_DIM)
    model_path = f"models/vae_{dataset_name}.pt"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo {model_path} não encontrado. Treine primeiro com train.py!")

    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def analyze_latent_space(dataset_name):
    """ Realiza análises quantitativas do espaço latente """
    print(f"\n🔹 Testando espaço latente para: {dataset_name}")

    # Carregar dataset
    df = load_dataset(dataset_name)
    X = df
    y = None

    if dataset_name == "wine":
        y = pd.read_csv("data/wine.csv").iloc[:, 0]  # Assumindo que a 1ª coluna seja o rótulo
    elif dataset_name == "boston_housing":
        y = pd.read_csv("data/boston_housing.csv").iloc[:, -1]  # Última coluna como rótulo
    
    if y is not None:
        y = y.values

    # Normalização Min-Max
    X = (X - X.min()) / (X.max() - X.min())
    X = torch.tensor(X.values, dtype=torch.float32)

    # Carregar modelo treinado
    model = load_trained_vae(dataset_name, input_dim=X.shape[1])

    # Passar dados pelo encoder do VAE
    with torch.no_grad():
        mu, _ = model.encode(X)

    mu = mu.numpy()

    # Aplicar PCA para reduzir para 2D
    pca = PCA(n_components=2)
    mu_2d = pca.fit_transform(mu)
    var_explicada = sum(pca.explained_variance_ratio_)

    # Calcular o coeficiente de silhueta para avaliar clusters
    silhouette = silhouette_score(mu, y) if len(np.unique(y)) > 1 else "Não aplicável (regressão)"

    print(f"Variância explicada pelos 2 primeiros componentes: {var_explicada:.2f}")
    print(f"Coeficiente de Silhueta: {silhouette}")

if __name__ == "__main__":
    analyze_latent_space("wine")
    analyze_latent_space("boston_housing")
