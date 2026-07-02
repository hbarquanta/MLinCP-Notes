# Chapter 1 — Introduction & Core ML Concepts

Machine learning (ML) is the discipline of constructing algorithms that learn patterns from data, rather than relying on explicitly programmed rules. In computational physics and chemistry, ML provides a route to building fast, accurate surrogate models for expensive quantum-mechanical calculations.

## 1.1 Learning Paradigms

Consider an unseen process (the data-generating process) that connects an input space $\mathcal{X}$ to a target space $\mathcal{Y}$, described by some function $f: \mathcal{X} \to \mathcal{Y}$.

Statistical samples of $\mathcal{X}$ are called **features** (also: attributes, input variables): $x_i \in \mathcal{X}$ are vectors of dimension $D$ containing reals, integers, or other values. Samples of $\mathcal{Y}$ are called **labels** (also: classes, targets): $y_i \in \mathcal{Y}$ can be scalars, vectors, or tensors. The index $i$ runs over individual examples.

A **dataset** $T$ can be labeled, $T = \{(x_i, y_i)\}$, or unlabeled, $T = \{x_i\}$.

**Supervised learning** requires a labeled dataset $T = \{(x_i, y_i)\}_{i=1}^n$ of $n$ examples. The goal is to find a model $\hat{f}: \mathcal{X} \to \mathcal{Y}$ that approximates $f$, i.e. $\hat{f}(x_i) = \hat{y}_i \approx y_i$, and generalizes well to new, unseen inputs.

**Unsupervised learning** works on unlabeled data $T = \{x_i\}_{i=1}^n$. The goal is to describe or understand the structure of the data — typical tasks include dimensionality reduction, clustering, outlier detection, and generative modelling (covered in L3 and L9).

The fundamental requirement in both cases is **generalization**: the model must perform well on data it has not seen during training, not merely memorize the training set. A model that achieves this is said to generalize.

## 1.2 Risk Minimization and Loss Functions

The quality of a model $\hat{f}$ is measured by a **loss function** $\ell(\hat{f}(x), y)$ that quantifies the discrepancy between the prediction $\hat{f}(x)$ and the true label $y$.

The **expected risk** is the expectation of the loss over the true (unknown) data distribution $\mathcal{P}(\mathcal{X}, \mathcal{Y})$:

$$R(\hat{f}) = \langle \ell(\hat{f}(\mathcal{X}), \mathcal{Y}) \rangle = \int \ell\!\left(\hat{f}(\mathcal{X}), \mathcal{Y}\right) \mathrm{d}\mathcal{P}(\mathcal{X}, \mathcal{Y})$$

Since $\mathcal{P}$ is unknown, the expected risk is approximated by the **empirical risk** — the mean loss over the finite training set $T$:

$$R_\mathrm{emp}(\hat{f}) = \frac{1}{n} \sum_{i=1}^{n} \ell\!\left(\hat{f}(x_i),\, y_i\right)$$

where $n = |T|$ is the number of training examples. The ML training process is the process of minimizing $R_\mathrm{emp}$.

**Common loss functions for regression** (continuous $y_i$):

L1 loss (least absolute deviations, more robust to outliers):

$$\ell(\hat{f}(x_i), y_i) = \sum_i \left|\hat{f}(x_i) - y_i\right|$$

L2 loss (least squares, standard for regression):

$$\ell(\hat{f}(x_i), y_i) = \sum_i \left(\hat{f}(x_i) - y_i\right)^2$$

The L2 loss is easier to optimize (smooth gradient) but more sensitive to outliers than L1.

## 1.3 Gradient-Based Optimization

Model parameters are optimized iteratively via gradient descent. For a linear model $z = w \cdot x + b$ with weight vector $w$ and scalar bias $b$, the parameter update at step $t$ is:

$$w_t \leftarrow w_{t-1} - \alpha \cdot \nabla_{w_{t-1}} \ell(w, y, z)$$

where $\alpha > 0$ is the **learning rate**. The gradient $\nabla_w \ell$ is propagated backwards through the computational graph using automatic differentiation.

**Stochastic Gradient Descent (SGD)** approximates the full gradient using a random **mini-batch** $B \ll M$ of training examples, where $M$ is the total training set size. The full dataset is divided into $N = M / n_\text{batch}$ mini-batches; one pass over all mini-batches is called an **epoch**.

**Adam** (Adaptive Moment Estimation, Kingma & Ba, 2015) maintains per-parameter adaptive learning rates by tracking the first moment (mean) and second moment (uncentered variance) of the gradients. At step $t$, with gradient $g_t = \nabla_{w_{t-1}} \ell$:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad \text{(first moment)}$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \qquad \text{(second moment)}$$

Bias-corrected estimates: $\hat{m}_t = m_t / (1 - \beta_1^t)$, $\hat{v}_t = v_t / (1 - \beta_2^t)$.
Parameter update:

$$w_{n+1} \leftarrow w_n - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \varepsilon)$$

Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$.

## 1.4 Overfitting, Regularization, and Model Validation

**Underfitting (high bias)** occurs when the model is too simple to capture the structure of the data — it achieves high loss on both training and test data.

**Overfitting (high variance)** occurs when the model is too complex and fits noise in the training data, achieving low training loss but high test loss. Overfitting yields an increased error on unseen data by approximating a simple functional relationship with an overly complex function.

**Regularization** discourages overfitting by adding a complexity penalty to the loss:

$$\ell_\text{reg}(\hat{f}) = \ell(\hat{f}) + \lambda\, R(\hat{f})$$

where $\lambda > 0$ controls the regularization strength. Common choices: $R = \|w\|_2^2$ (L2/Ridge, penalizes large weights) and $R = \|w\|_1$ (L1/Lasso, promotes sparsity).

## 1.5 Data Splitting and Cross-Validation

The dataset is partitioned into three strictly non-overlapping subsets:

- **Training set** (typically 60–80%): the model sees this data directly; parameters are updated by gradient descent on training batches.
- **Validation set** (typically 10–20%): used for hyperparameter tuning and early stopping; the model is indirectly influenced — do not over-optimize on it.
- **Test set** (typically 10–20%): touched exactly once at the very end; provides an honest estimate of generalization to unseen data.

**$K$-fold cross-validation** is used when data is scarce. The training data is divided into $K$ equal folds; the model is trained $K$ times, each time holding out a different fold for validation. The $K$ validation losses are averaged for a more reliable estimate of generalization. Leave-one-out CV (LOO-CV) is the special case $K = n$.

**Data augmentation** artificially expands the training set by applying label-preserving transformations. For atomistic data: 3D rotations and reflections, noise on atomic positions, permutation of equivalent atoms.

---

