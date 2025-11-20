# ECE 236A Project Report
## Decentralized Classification under Feature Compression

**Course:** ECE 236A - Linear Programming  
**Fall 2025**

---

## 1. Introduction

This project addresses the problem of multi-class classification under communication constraints, where image features must be quantized before transmission to a decision center. We explore both centralized and decentralized settings, formulating Linear Programming (LP) and Integer Linear Programming (ILP) models to optimize classification accuracy under bit budget constraints.

---

## 2. Task 1: Multi-Class Linear Classifier

### 2.1 Formulation

We formulate the training of a multi-class linear classifier as a Linear Program with margin constraints.

**Decision Variables:**
- $W \in \mathbb{R}^{K \times M}$: weight matrix where $W[k, :]$ is the weight vector for class $k$
- $\mathbf{b} \in \mathbb{R}^{K}$: bias vector where $\mathbf{b}[k]$ is the bias for class $k$

**Constraints:**
For each training sample $(\mathbf{x}^{(i)}, y^{(i)})$ and for all classes $k \neq y^{(i)}$:
$$\mathbf{w}_{y^{(i)}}^T \mathbf{x}^{(i)} + b_{y^{(i)}} \geq \mathbf{w}_k^T \mathbf{x}^{(i)} + b_k + \gamma$$

where $\gamma = 1.0$ is the margin parameter.

**Objective:**
$$\min_{W, \mathbf{b}} \|W\|_F^2 = \min_{W, \mathbf{b}} \sum_{k=1}^{K} \sum_{m=1}^{M} W[k,m]^2$$

This minimizes the Frobenius norm of the weight matrix, providing L2 regularization to prevent overfitting and ensure the problem is well-posed.

**Fallback Method (Soft Margin):**
When the hard margin problem is infeasible (data not linearly separable), we introduce slack variables $\xi_i \geq 0$ for each sample and modify the constraints:
$$\mathbf{w}_{y^{(i)}}^T \mathbf{x}^{(i)} + \mathbf{b}_{y^{(i)}} \geq \mathbf{w}_k^T \mathbf{x}^{(i)} + \mathbf{b}_k + \gamma - \xi_i$$

The objective becomes:
$$\min_{W, \mathbf{b}, \boldsymbol{\xi}} \|W\|_F^2 + C \sum_{i=1}^{N} \xi_i$$

where $C = 1.0$ is the regularization parameter for slack.

### 2.2 Assumptions

- **Feature Scaling:** No normalization is applied; features are used as-is from the dataset
- **Label Mapping:** Labels are automatically mapped to consecutive integers $\{0, 1, \ldots, K-1\}$ if needed (e.g., Fashion-MNIST labels $\{0, 3, 4\}$ are mapped to $\{0, 1, 2\}$)
- **Train/Val/Test Separation:** Strict separation maintained; validation set used only for hyperparameter tuning (margin $\gamma$ and slack penalty $C$), test set used only for final evaluation
- **Solver:** CVXPY with automatic solver selection (ECOS preferred, falls back to default solver)

### 2.3 Results

**Synthetic Dataset (2D, 3 classes):**
- Training Accuracy: 96.75%
- Test Accuracy: 97.60%
- Note: Required soft margin fallback due to non-linear separability

**Fashion-MNIST Dataset (784D, 3 classes):**
- Training Accuracy: 100.00%
- Test Accuracy: 89.80%
- This serves as the baseline for comparison with compressed features

---

## 3. Task 2: Centralized Compression

### 3.1 Formulation

We implement **Uniform Scalar Quantization (USQ)** for centralized feature compression.

**Quantization Process:**
For a given total budget $B_{\text{tot}}$ and $M = 784$ features:
1. Compute bits per feature: $b = \lfloor B_{\text{tot}} / M \rfloor$
2. For each feature $m$, compute quantization parameters from training data:
   - $x_{\min}^m = \min_i x_m^{(i)}$ (over training set)
   - $x_{\max}^m = \max_i x_m^{(i)}$ (over training set)
   - Number of quantization levels: $L = 2^b$
   - Step size: $\Delta_m = \frac{x_{\max}^m - x_{\min}^m}{L - 1}$

3. Quantize feature $x_m$:
   $$q_m = \text{Round}\left(\frac{x_m - x_{\min}^m}{\Delta_m}\right)$$
   $$Q_b(x_m) = x_{\min}^m + q_m \cdot \Delta_m$$

**Bit Accounting:**
- Total bits per image: $B_{\text{tot}} = M \cdot b$ (when $b$ is uniform across features)
- For budgets where $B_{\text{tot}} < M$, we have $b = 0$ (all features quantized to zero)

**Classification:**
After quantization, we train the classifier from Task 1 on quantized training data and evaluate on quantized test data.

### 3.2 Assumptions

- **Quantization Parameters:** Computed from training data only (no test leakage)
- **Uniform Allocation:** All features receive the same bit-depth $b$ (simplest approach)
- **Dequantization:** Quantized values are dequantized back to approximate original range before classification
- **Train/Val/Test Separation:** Quantization parameters estimated from training set, applied to validation/test sets

### 3.3 Results

Test accuracy vs. total budget $B_{\text{tot}}$ on Fashion-MNIST:

| Budget (bits) | Bits/Feature | Test Accuracy |
|---------------|--------------|---------------|
| 100           | 0            | 32.4%         |
| 400           | 0            | 32.4%         |
| 784           | 1            | 85.4%         |
| 1568          | 2            | 89.4%         |
| 2352          | 3            | 89.6%         |
| 3136          | 4            | 90.4%         |

**Observations:**
- Very low budgets ($< 1$ bit/feature) result in poor accuracy (~32%)
- 1 bit/feature achieves 85.4% accuracy (5.4% below baseline)
- 2 bits/feature achieves 89.4% accuracy (0.4% below baseline)
- Higher bit-depths approach baseline performance (89.8%)

---

## 4. Task 3.1: Decentralized Compression (Fixed Per-Sensor Budget)

### 4.1 Formulation

In the decentralized setting, each of $4$ sensors observes one quadrant ($196$ features each). For a fixed per-sensor budget $k$ (bits per image per sensor):

**Per-Sensor Quantization:**
For each sensor $s \in \{1, 2, 3, 4\}$:
1. Compute bits per feature: $b_s = \lfloor k / d_s \rfloor$ where $d_s = 196$
2. Apply uniform scalar quantization independently to sensor $s$'s features using its own training data to compute quantization parameters
3. All sensors use the same per-sensor budget $k$, resulting in uniform allocation: $b_1 = b_2 = b_3 = b_4 = b_s$

**Reconstruction:**
After quantization, quantized blocks from all sensors are concatenated to form the full feature vector, and the classifier from Task 1 is trained on the concatenated quantized data.

**Bit Accounting:**
- Per-sensor budget: $k$ bits per image per sensor
- Total budget: $B_{\text{tot}} = 4k$ bits per image

### 4.2 Assumptions

- **Independent Quantization:** Each sensor quantizes its features independently using its own quantization parameters
- **Uniform Per-Sensor Allocation:** All sensors receive the same bit budget $k$
- **No Cross-Sensor Coordination:** Sensors do not coordinate; each quantizes based only on its local training data
- **Train/Val/Test Separation:** Quantization parameters computed from each sensor's training block, applied to corresponding val/test blocks

### 4.3 Results

Test accuracy vs. per-sensor budget $k$ on Fashion-MNIST:

| $k$ (bits/sensor) | Total Budget | $(b_1, b_2, b_3, b_4)$ (bits/feature) | Test Accuracy |
|-------------------|--------------|--------------------------------------|---------------|
| 25                | 100          | $(0, 0, 0, 0)$                        | 32.4%         |
| 100               | 400          | $(0, 0, 0, 0)$                        | 32.4%         |
| 196               | 784          | $(1, 1, 1, 1)$                        | 85.4%         |
| 392               | 1568         | $(2, 2, 2, 2)$                        | 89.4%         |
| 588               | 2352         | $(3, 3, 3, 3)$                        | 89.6%         |
| 784               | 3136         | $(4, 4, 4, 4)$                        | 90.4%         |

**Observations:**
- Performance matches centralized compression at matched total budgets
- This is expected since uniform per-sensor allocation is equivalent to uniform centralized allocation when total budgets match

---

## 5. Task 3.2: Decentralized Compression (Fixed Total Budget)

### 5.1 Formulation

Given a total budget $B_{\text{tot}}$ shared across 4 sensors, we optimize bit allocation $(b_1, b_2, b_3, b_4)$ to maximize classification accuracy.

**Approach: Outer Search with Validation-Based Selection**

We use an outer search over candidate allocations with inner LP solves:

1. **Generate Candidate Allocations:**
   - Uniform allocation: $b_s = \lfloor B_{\text{tot}} / \sum_{s=1}^4 d_s \rfloor$ for all $s \in \{1, 2, 3, 4\}$
   - Non-uniform allocations: Try giving one sensor $b_s + 1$ bits while others get $b_s$ bits, subject to budget constraint: $\sum_{s=1}^4 d_s \cdot b_s \leq B_{\text{tot}}$

2. **For Each Candidate Allocation:**
   - Quantize each sensor's features independently using $b_s$ bits
   - Concatenate quantized blocks
   - Train classifier on quantized training data
   - Evaluate on validation set

3. **Select Best Allocation:**
   - Choose allocation with highest validation accuracy
   - Evaluate selected allocation on test set

**Decision Variables (implicit in search):**
- $b_s \in \mathbb{Z}_{\geq 0}$: bits per feature for sensor $s$
- Constraint: $\sum_{s=1}^4 d_s \cdot b_s \leq B_{\text{tot}}$

**Objective (implicit):**
- Maximize validation accuracy (used for selection)
- Report test accuracy for selected allocation

### 5.2 Assumptions

- **Search Strategy:** Limited search space for efficiency (uniform + a few non-uniform candidates)
- **Validation-Based Selection:** Use validation accuracy to choose among candidate allocations (strict train/val/test separation)
- **Independent Per-Sensor Quantization:** Each sensor quantizes independently with its allocated bits
- **Bit Accounting:** Total bits = $\sum_{s=1}^4 d_s \cdot b_s$ where $d_s = 196$ for all sensors

### 5.3 Results

Test accuracy vs. total budget $B_{\text{tot}}$ with optimized allocations:

| Budget | Selected Allocation $(b_1, b_2, b_3, b_4)$ | Actual Budget | Test Accuracy |
|--------|-------------------------------------------|---------------|---------------|
| 100    | $(0, 0, 0, 0)$                             | 0             | 32.4%         |
| 400    | $(0, 1, 0, 0)$                             | 196           | 76.8%         |
| 784    | $(1, 1, 1, 1)$                             | 784           | 85.4%         |
| 1568   | $(2, 2, 2, 2)$                             | 1568          | 89.4%         |
| 2352   | $(2, 4, 2, 2)$                             | 1960          | 90.4%         |
| 3136   | $(3, 5, 3, 3)$                             | 2744          | 90.2%         |

**Key Observations:**
- **Non-uniform allocations can outperform uniform:** At budget $400$, allocation $(0, 1, 0, 0)$ achieves $76.8\%$ vs. uniform $(0, 0, 0, 0)$ at $32.4\%$
- **Higher budgets benefit from non-uniform allocation:** At budget $2352$, allocation $(2, 4, 2, 2)$ achieves $90.4\%$ accuracy, matching higher uniform allocations
- **Centralized vs. Decentralized:** At matched budgets, decentralized with optimized allocation can match or slightly exceed centralized performance

---

## 6. Task 3.3: Minimal Bits for Target Accuracy

### 6.1 Formulation

Given a target accuracy $\alpha$, find the minimal total bit budget $B_{\text{tot}}$ (and allocation for decentralized) that achieves test accuracy $\geq \alpha$.

**Approach: Outer Search Over Budget Grid**

**Centralized:**
1. Search through candidate budgets $B \in B_{\text{grid}}$ in ascending order
2. For each $B$, run centralized compression (Task 2) and evaluate test accuracy
3. Return the smallest $B$ where test accuracy $\geq \alpha$

**Decentralized:**
1. Search through candidate budgets $B \in B_{\text{grid}}$ in ascending order
2. For each $B$, run decentralized compression with optimized allocation (Task 3.2) and evaluate test accuracy
3. Return the smallest $B$ and its corresponding allocation $(b_1, b_2, b_3, b_4)$ where test accuracy $\geq \alpha$

**Decision Variables:**
- $B_{\text{tot}} \in \mathbb{Z}_{\geq 0}$: total bit budget (centralized)
- $(b_1, b_2, b_3, b_4)$: bit allocation per sensor (decentralized)

**Constraints:**
- Test accuracy $\geq \alpha$
- Budget constraint: $\sum_{s=1}^4 d_s \cdot b_s \leq B_{\text{tot}}$ (decentralized)

**Objective:**
- Minimize $B_{\text{tot}}$

### 6.2 Assumptions

- **Search Grid:** Predefined grid of candidate budgets: $B_{\text{grid}} = \{100, 400, 784, 1568, 2352, 3136\}$
- **Greedy Search:** Search in ascending order, return first budget achieving target
- **Test Accuracy:** Uses test set accuracy (as per specification) to determine if target is met
- **Train/Val/Test Separation:** Validation used for allocation selection (Task 3.2), test used only for final accuracy evaluation

### 6.3 Results

Minimal bits required to achieve target accuracy $\alpha$:

| Target $\alpha$ | Centralized Min Bits | Decentralized Min Bits | Decentralized Allocation $(b_1, b_2, b_3, b_4)$ |
|-----------------|---------------------|----------------------|-------------------------------------------------|
| 0.60            | 784                 | 400                  | $(0, 1, 0, 0)$                                   |
| 0.70            | 784                 | 400                  | $(0, 1, 0, 0)$                                   |
| 0.80            | 784                 | 784                  | $(1, 1, 1, 1)$                                   |
| 0.90            | 3136                | 2352                 | $(2, 4, 2, 2)$                                   |
| 0.95            | None                | None                 | N/A                                              |

**Key Observations:**
- **Decentralized can be more efficient:** For lower targets ($0.6$-$0.7$), decentralized requires only $400$ bits vs. $784$ bits for centralized
- **Non-uniform allocation advantage:** The $(0, 1, 0, 0)$ allocation allows achieving $60$-$70\%$ accuracy with much fewer bits than uniform allocation
- **Higher targets require more bits:** Target $0.90$ requires $3136$ bits (centralized) or $2352$ bits (decentralized with optimized allocation)
- **Target $0.95$ not achievable:** Within the tested budget range, $95\%$ accuracy is not achievable for either setting

**Discussion of Centralized vs. Decentralized Gap:**
- For lower accuracy targets, decentralized with optimized non-uniform allocation can be more efficient than centralized uniform allocation
- This is because non-uniform allocation allows focusing bits on more informative sensors/features
- For higher accuracy targets, the gap narrows as both approaches require substantial bit budgets
- Centralized has the theoretical advantage of seeing all features simultaneously, but decentralized with smart allocation can compensate through targeted bit allocation

---

## 7. Conclusions

This project successfully demonstrates the application of Linear Programming to multi-class classification under communication constraints. Key findings:

1. **LP-based classification** achieves strong baseline performance ($89.8\%$ on Fashion-MNIST) with a simple margin-based formulation
2. **Uniform scalar quantization** provides a practical compression scheme, with $1$-$2$ bits per feature achieving near-baseline accuracy
3. **Non-uniform bit allocation** in decentralized settings can outperform uniform allocation, especially at lower budgets
4. **Decentralized compression** with optimized allocation can match or exceed centralized performance at matched budgets for certain accuracy targets

The formulations are computationally tractable and provide interpretable solutions, making them suitable for resource-constrained edge computing applications.

---

## References

1. Fashion-MNIST Dataset: https://github.com/zalandoresearch/fashion-mnist
2. CVXPY Documentation: https://www.cvxpy.org/

