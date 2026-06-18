from data_utils import load_csv, ZScaler
from pca_module import fit_pca, calculate_reconstruction_error
from training import iterative_cleaning, tune_threshold
from metrics import calculate_metrics, print_diagnostics
from plotting import plot_results
import numpy as np

def run_pipeline():
    # 1. Load Data
    X_train, y_train = load_csv('archive/Friday-WorkingHours-Morning.pcap_ISCX.csv')
    X_test, y_test = load_csv('archive/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

    # Ensure training set only contains normal traffic
    if y_train.sum() > 0:
        X_train = X_train[y_train == 0]

    # 2. Scaling
    scaler = ZScaler()
    scaler.fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # 3. Training (Iterative cleaning and threshold tuning)
    K = 3
    W, X_clean = iterative_cleaning(X_train_sc, K=K)
    
    re_train_clean = calculate_reconstruction_error(X_clean, W)
    re_test = calculate_reconstruction_error(X_test_sc, W)

    threshold, best_mult, val_f1 = tune_threshold(re_train_clean, re_test, y_test)
    
    # 4. Prediction and Metrics
    predicted = (re_test > threshold).astype(int)
    precision, recall, f1 = calculate_metrics(y_test, predicted)

    print("=== DETECTOR RESULTS ===")
    print(f"  Threshold: {threshold:.4f}  (multiplier {best_mult:.1f} × log_MAD)")
    print(f"  K = {K} principal components")
    print(f"  Precision:  {precision:.3f}")
    print(f"  Recall:     {recall:.3f}")
    print(f"  F1:         {f1:.3f}\n")

    # 5. Diagnostics and Visualization
    print_diagnostics(re_test, y_test, predicted)
    plot_results(re_test, y_test, threshold, W, X_test_sc, f1, K)

if __name__ == "__main__":
    run_pipeline()
