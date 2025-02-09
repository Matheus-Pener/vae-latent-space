import pandas as pd
import os
from sklearn.datasets import fetch_openml

# Definição dos caminhos dos arquivos
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')

# Certificar-se de que a pasta existe
os.makedirs(DATA_PATH, exist_ok=True)

def load_dataset(name):
    """
    Carrega um dos datasets disponíveis.

    Args:
        name (str): Nome do dataset ('wine' ou 'boston_housing').

    Returns:
        pd.DataFrame: DataFrame com os dados carregados.
    """
    if name == "wine":
        file_path = os.path.join(DATA_PATH, "wine.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Arquivo {file_path} não encontrado!")
        return pd.read_csv(file_path)

    elif name == "boston_housing":
        file_path = os.path.join(DATA_PATH, "boston_housing.csv")
        if not os.path.exists(file_path):
            print("Baixando o dataset Boston Housing...")
            boston = fetch_openml(name="boston", version=1, as_frame=True)
            df = boston.data
            df["TARGET"] = boston.target  # Adicionamos a variável alvo
            df.to_csv(file_path, index=False)
            print(f"Dataset salvo em {file_path}")
        return pd.read_csv(file_path)

    else:
        raise ValueError(f"Dataset {name} não é suportado!")

# Teste de carregamento
if __name__ == "__main__":
    try:
        wine_df = load_dataset("wine")
        boston_df = load_dataset("boston_housing")
        print("Wine dataset carregado com sucesso!", wine_df.shape)
        print("Boston Housing dataset carregado com sucesso!", boston_df.shape)
    except Exception as e:
        print("Erro ao carregar os datasets:", e)
