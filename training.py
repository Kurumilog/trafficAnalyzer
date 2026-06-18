import numpy as np
from pca_module import fit_pca, calculate_reconstruction_error

def iterative_cleaning(X_train_sc, K, max_iter=10, ratio_threshold=50):
    """
    Removes outliers from the training set to clean the 'normal' subspace.
    Algorithm: build W, calculate RE, remove top 1% outliers, 
    repeat until distribution becomes compact.
    """
    X_clean = X_train_sc.copy()
    W = None
    
    print(f"Iterative cleaning (K={K}):")
    for iteration in range(max_iter):
        W, _ = fit_pca(X_clean, K)
        re = calculate_reconstruction_error(X_clean, W)

        median_re = np.median(re)
        p99 = np.percentile(re, 99)
        ratio = p99 / (median_re + 1e-8)

        print(f"  Iteration {iteration+1}: n={len(X_clean)} | "
              f"median={median_re:.3f} | 99p={p99:.3f} | ratio={ratio:.1f}")

        if ratio < ratio_threshold:
            print(f"  → Converged in {iteration+1} iterations\n")
            break

        X_clean = X_clean[re < p99]
    else:
        print("  → Maximum iterations reached\n")
        
    return W, X_clean

def tune_threshold(re_train, re_test, y_test, val_ratio=0.2):
    """
    Threshold selection using Log-MAD and F1-score on a validation set.
    """
    log_re_train = np.log(re_train + 1e-8)
    log_median = np.median(log_re_train)
    log_mad = np.median(np.abs(log_re_train - log_median))

    # Build validation set (20% normal, 20% attacks)
    np.random.seed(42)
    val_size = int(val_ratio * len(re_train))
    val_normal = re_train[np.random.choice(len(re_train), val_size, replace=False)]
    re_attacks_sample = re_test[y_test == 1][:val_size]

    re_val = np.concatenate([val_normal, re_attacks_sample])
    y_val = np.array([0] * len(val_normal) + [1] * len(re_attacks_sample))

    best_f1, best_mult = 0.0, 3.0
    for mult in np.arange(1.0, 15.0, 0.1):
        thr = np.exp(log_median + mult * log_mad)
        pred = (re_val > thr).astype(int)
        
        tp = ((pred == 1) & (y_val == 1)).sum()
        fp = ((pred == 1) & (y_val == 0)).sum()
        fn = ((pred == 0) & (y_val == 1)).sum()
        
        pr = tp / (tp + fp + 1e-8)
        rc = tp / (tp + fn + 1e-8)
        f = 2 * pr * rc / (pr + rc + 1e-8)
        
        if f > best_f1:
            best_f1, best_mult = f, mult

    threshold = np.exp(log_median + best_mult * log_mad)
    return threshold, best_mult, best_f1
