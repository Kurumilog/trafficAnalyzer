import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    Computes Precision, Recall, and F1-score.
    """
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return precision, recall, f1

def print_diagnostics(re_test, y_test, predicted):
    """
    Analyzes missed attacks and false positives.
    """
    re_caught = re_test[(y_test == 1) & (predicted == 1)]
    re_missed = re_test[(y_test == 1) & (predicted == 0)]
    re_fp = re_test[(y_test == 0) & (predicted == 1)]

    print("Detection Diagnostics:")
    print(f"  Caught attacks:   {len(re_caught):6d} | mean RE = {re_caught.mean():.4f}")
    print(f"  Missed attacks:   {len(re_missed):6d} | mean RE = {re_missed.mean():.4f}")
    print(f"  False alarms:     {len(re_fp):6d} | mean RE = {re_fp.mean():.4f}")
    print(f"  Normal traffic — mean RE: {re_test[y_test==0].mean():.4f}")
    print(f"  Normal traffic — 95p RE:  {np.percentile(re_test[y_test==0], 95):.4f}")
    print(f"  Normal traffic — max RE:  {re_test[y_test==0].max():.4f}\n")
