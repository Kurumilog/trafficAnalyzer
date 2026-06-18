# Mathematical & Logical Foundations of PCA Anomaly Detector

This document describes the mathematics and algorithms used by the PCA-based network anomaly detector.

---

## 1. Features And Labels

The detector operates on the following 10 flow-level features:

- Flow Duration
- Total Fwd Packets
- Total Backward Packets
- Flow Bytes/s
- Flow Packets/s
- Fwd Packet Length Mean
- Bwd Packet Length Mean
- Packet Length Std
- SYN Flag Count
- RST Flag Count

Labels are converted to a binary target:

$$
y_i = 
\begin{cases}
0, & \text{if Label} = \text{BENIGN} \\
1, & \text{otherwise}
\end{cases}
$$

The current PCA dimensionality is $K = 3$.

---

## 2. Data Cleaning And Z-Score Normalization

Before learning, the loader removes rows with missing or infinite values. PCA is then applied to standardized features so that high-magnitude columns do not dominate the covariance matrix.

For each feature $j$, the training mean and standard deviation are computed on the normal training subset:

$$
\mu_j = \frac{1}{n} \sum_{i=1}^{n} x_{ij}
$$

$$
\sigma_j = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_{ij} - \mu_j)^2}
$$

The normalized value is:

$$
x'_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j + \epsilon}
$$

where $\epsilon = 10^{-8}$ prevents division by zero.

In matrix form, if $X \in \mathbb{R}^{n \times D}$ is the raw feature matrix, then the standardized matrix is $X' \in \mathbb{R}^{n \times D}$.

---

## 3. PCA Projection

PCA finds an orthonormal basis that maximizes variance in the projected space.

### Covariance Matrix

The covariance matrix of the standardized data is:

$$
C = \frac{1}{n-1} (X')^T X'
$$

### Eigenvalue Decomposition

We solve:

$$
C v_i = \lambda_i v_i
$$

where $\lambda_i$ is the variance explained by the $i$-th component and $v_i$ is the corresponding eigenvector.

The eigenvectors are sorted in descending order of $\lambda_i$ and the top $K$ vectors form the projection matrix:

$$
W = [v_1, v_2, \dots, v_K] \in \mathbb{R}^{D \times K}
$$

For this project, $K=3$ by default.

---

## 4. Anomaly Score: Reconstruction Error

The detector assumes normal flows lie close to a low-dimensional PCA subspace. A sample that cannot be reconstructed well from that subspace gets a high anomaly score.

### Projection And Reconstruction

$$
z = x' W
$$

$$
\hat{x} = z W^T = x' W W^T
$$

### Reconstruction Error

$$
RE(x) = \|x' - \hat{x}\|_2^2 = \sum_{j=1}^{D} (x'_j - \hat{x}_j)^2
$$

Large $RE(x)$ means the sample does not fit the learned normal subspace well.

For a batch $\{x_i\}_{i=1}^{n}$, the detector computes the vector of reconstruction errors:

$$
\mathbf{RE} = [RE(x_1), RE(x_2), \dots, RE(x_n)]
$$

---

## 5. Iterative Cleaning Of The Normal Subspace

The training set is assumed to be mostly normal, but it may still contain noisy samples. The code therefore refines the normal subspace iteratively.

At iteration $t$:

$$
W_t, \mathbf{RE}_t = \text{PCA}(X_t)
$$

$$
m_t = \text{median}(\mathbf{RE}_t)
$$

$$
p_{99,t} = \text{percentile}_{99}(\mathbf{RE}_t)
$$

$$
r_t = \frac{p_{99,t}}{m_t + \epsilon}
$$

If $r_t < 50$, the process stops. Otherwise, the top 1% highest-RE samples are removed:

$$
X_{t+1} = \{x \in X_t : RE(x) < p_{99,t}\}
$$

The loop runs for at most 10 iterations.

---

## 6. Robust Threshold Selection: Log-MAD

Reconstruction errors are heavy-tailed, so the threshold is computed in log space.

### Log Transform

$$
\ell_i = \log(RE_i + \epsilon)
$$

### Median And MAD

$$
M = \text{median}(\ell)
$$

$$
\mathrm{MAD} = \text{median}(|\ell_i - M|)
$$

### Candidate Threshold

$$
T(m) = \exp(M + m \cdot \text{MAD})
$$

The implementation searches multipliers $m$ from $1.0$ to $14.9$ with step $0.1$ and selects the value that maximizes the validation $F_1$ score.

The validation set is built from:

- a sample of normal reconstruction errors from the cleaned normal set
- an equally sized sample of attack reconstruction errors from the anomaly set

Final prediction rule:

$$
\hat{y}_i = \mathbb{1}[RE(x_i) > T]
$$

---

## 7. Evaluation Metrics

The detector reports standard binary classification metrics.

Let $TP$, $FP$, and $FN$ be true positives, false positives, and false negatives.

$$
\mathrm{Precision} = \frac{TP}{TP + FP + \epsilon}
$$

$$
\mathrm{Recall} = \frac{TP}{TP + FN + \epsilon}
$$

$$
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall} + \epsilon}
$$

The diagnostics also print:

- mean RE of caught attacks
- mean RE of missed attacks
- mean RE of false alarms
- mean, 95th percentile, and maximum RE for normal traffic

---

## 8. Interpretation

The learned model can be interpreted through three quantities:

1. The projection matrix $W$, which defines the normal subspace.
2. The reconstruction error $RE(x)$, which measures how far a sample lies from that subspace.
3. The threshold $T$, which converts a continuous anomaly score into a binary decision.

If a sample has small reconstruction error, it is likely consistent with the learned normal behavior. If the error exceeds the threshold, the sample is classified as anomalous.
