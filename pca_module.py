import numpy as np

def fit_pca(X_scaled, K):
    """
    Computes the covariance matrix and finds eigenvectors.
    Returns W (D x K) projection matrix and all eigenvalues.
    """
    # C = (X^T · X) / (n - 1) — covariance matrix
    C = (X_scaled.T @ X_scaled) / (len(X_scaled) - 1)
    
    # Solve eigenvalue equation: C · v = λ · v
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # Sort by descending variance (λ)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    return eigenvectors[:, :K], eigenvalues

def calculate_reconstruction_error(X, W):
    """
    RE(x) = ||x - x·W·W^T||²
    Projects data into K-dimensional subspace and back.
    Large RE indicates a vector that does not fit the normal subspace.
    """
    X_recon = (X @ W) @ W.T
    return np.sum((X - X_recon) ** 2, axis=1)
