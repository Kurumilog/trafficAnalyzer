import numpy as np
import pandas as pd
from config import FEATURES

def load_csv(path):
    """
    Loads CSV, cleans data (removes inf/nan).
    Returns X (matrix n x 10) and y (labels 0/1).
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    feats = [f for f in FEATURES if f in df.columns]
    X = df[feats].values.astype(float)
    y = (df['Label'] != 'BENIGN').astype(int).values
    print(f"Loaded: {path.split('/')[-1]}")
    print(f"  Rows: {len(X)} | Features: {X.shape[1]}")
    print(f"  BENIGN: {(y==0).sum()} | Attacks: {(y==1).sum()}\n")
    return X, y

class ZScaler:
    """
    Standard Z-score scaler: x' = (x - mu) / sigma
    """
    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, X):
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        print("ZScaler — Parameters:")
        print(f"  μ (first 3): {self.mu[:3].round(2)}")
        print(f"  σ (first 3): {self.sigma[:3].round(2)}\n")

    def transform(self, X):
        return (X - self.mu) / self.sigma
