# Chapter 5 — Neural Networks, Trees & Regularization

This chapter covers two families of universal approximators — tree-based models and feed-forward neural networks — followed by the optimization and regularization strategies used to train them.

## 5.1 Decision Stumps and Decision Trees

The simplest nonlinear model is a **decision stump**: a single step function along one input coordinate. For a one-dimensional input $x \in \mathbb{R}$, a stump has three parameters — a bias $w_0$, a step height $w_1$, and a split point $s_1$ — and produces

$$f(x;\,w_0, w_1, s_1) = w_0 + w_1 \,\mathbf{1}\!\left\{x > s_1\right\},$$

where $\mathbf{1}\{x > s_1\}$ is 1 if $x > s_1$ and 0 otherwise. For an $N$-dimensional input $\mathbf{x} \in \mathbb{R}^N$ the stump still cuts along a single coordinate: $\mathbf{1}\{x_i > s_1\}$ for some chosen feature index $i$.

A **decision tree** of depth $D$ is a hierarchy of $2^D - 1$ stumps arranged so that the output of each stump routes a data point to the left or right child stump, terminating in $2^D$ leaf nodes (each holding a constant prediction value). The total number of free parameters grows exponentially with depth: each internal node contributes a split index, a split threshold $s$, and the two leaf values $v_L$ and $v_R$.

**Training a tree of depth $D$** proceeds recursively. For each candidate split — choosing feature index $i$ and threshold $s$ — partition the $P$ training points into two groups $\mathcal{P}_L = \{p : x_i^{(p)} \leq s\}$ and $\mathcal{P}_R = \{p : x_i^{(p)} > s\}$. The optimal constant predictions are the group means,

$$v_L = \frac{1}{|\mathcal{P}_L|}\sum_{p \in \mathcal{P}_L} y^{(p)}, \qquad v_R = \frac{1}{|\mathcal{P}_R|}\sum_{p \in \mathcal{P}_R} y^{(p)},$$

and the split is chosen to minimize the least-squares error across both groups. The procedure is repeated recursively within each child node until depth $D$ is reached. Trees are high-variance, low-bias models: a sufficiently deep tree memorizes the training data.

## 5.2 Boosted Trees

**Boosting** converts a sequence of weak learners (shallow trees) into a strong model by fitting each new tree to the residuals left by the current ensemble. Let $\hat{h}_b(\mathbf{x}) = \sum_{b'=1}^b f_{b'}(\mathbf{x}, \omega_{b'})$ denote the ensemble after $b$ rounds, where $\omega_b = (s, v_L, v_R)_b$ collects the parameters of tree $b$. At round $b+1$, define the residuals

$$r_{b}^{(p)} = y^{(p)} - \hat{h}_b(\mathbf{x}^{(p)}), \qquad p = 1, \ldots, P,$$

and train tree $f_{b+1}$ to minimize the sum of squared residuals $\sum_p (r_b^{(p)} - f_{b+1}(\mathbf{x}^{(p)}))^2$. The ensemble is then updated: $\hat{h}_{b+1} = \hat{h}_b + f_{b+1}$. This continues for $B$ rounds, yielding

$$\hat{h}_B(\mathbf{x}) = \sum_{b=1}^B f_b(\mathbf{x}, \omega_b).$$

Each new tree corrects the mistakes of all previous trees. Boosting reduces bias by iterative refinement; combined with early stopping, it is the dominant approach for structured tabular data (implemented in XGBoost, LightGBM, etc.).

## 5.3 Bagging and Random Forests

**Bagging** (bootstrap aggregating) is an ensemble method that trains many models in parallel on different bootstrap samples of the training set (random draws with replacement) and averages their predictions. Unlike boosting, bagging reduces variance rather than bias, and the base models are independent so training is embarrassingly parallel.

A **random forest** is bagged decision trees with an additional randomization: at each split, only a random subset of $\sqrt{N}$ out of $N$ input features is considered as split candidates (where $N$ is the total feature dimensionality). This decorrelates the trees — if one feature dominates, individual trees do not all exploit it in the same way — which further reduces variance. Random forests are robust, require minimal hyperparameter tuning, and provide a natural out-of-bag error estimate (using the samples not drawn for each tree).

## 5.4 Feed-Forward Neural Networks

A **feed-forward neural network (FFNN)** is a composition of parameterized nonlinear transformations arranged in layers. Let the input layer be $\mathbf{f}^{(0)}(\mathbf{x}) = \mathbf{x} \in \mathbb{R}^N$. The network has $L$ hidden layers followed by an output layer. The $i$-th unit in layer $k \in \{1, \ldots, L\}$ computes

$$f_i^{(k)}(\mathbf{x}, \omega^{(k)}) = \sigma\!\left(w_{i,0}^{(k)} + \sum_j w_{i,j}^{(k)} f_j^{(k-1)}(\mathbf{x}, \omega^{(k-1)})\right),$$

where $w_{i,0}^{(k)}$ is a bias, $w_{i,j}^{(k)}$ is the weight from unit $j$ in layer $k-1$ to unit $i$ in layer $k$, and $\sigma(\cdot)$ is a pointwise nonlinear **activation function**. All parameters for layer $k$ are collected in $\omega^{(k)}$. The network depth equals the number of recursive applications of this formula; each application constitutes one hidden layer. The output layer applies no activation function for regression (raw linear output) or a softmax for classification.

The entire parameter set is $\omega = \{\omega^{(1)}, \ldots, \omega^{(L)}, \omega^{(\text{out})}\}$. By the universal approximation theorem, a single hidden layer with sufficiently many units can approximate any continuous function; in practice, depth is more parameter-efficient than width, and deep architectures learn hierarchical representations.

[TensorFlow Playground](https://playground.tensorflow.org) provides an interactive browser-based visualization of how depth, width, activation choice, and regularization affect the decision boundary of a network trained on 2D data — useful for building intuition before working through the math.

## 5.5 Activation Functions

The activation function $\sigma$ is the source of nonlinearity. Four standard choices are:

**Sigmoid**: $\sigma(x) = 1/(1 + e^{-x})$. Output lies in $(0, 1)$, making it a natural probability model. Saturates for large $|x|$, causing vanishing gradients in deep networks.

**Hyperbolic tangent**: $\tanh(x) = (e^x - e^{-x})/(e^x + e^{-x})$. Output in $(-1, 1)$, zero-centered (unlike sigmoid), but still saturates.

**Rectified Linear Unit (ReLU)**: $\text{ReLU}(x) = \max(0, x)$. Gradient is exactly 1 for $x > 0$, solving the vanishing-gradient problem. However, units with $x < 0$ always output zero and receive zero gradient — the "dying neuron" problem.

**Exponential Linear Unit (ELU)**: $\text{ELU}(x) = x$ if $x \geq 0$, and $\alpha(e^x - 1)$ if $x < 0$, where $\alpha > 0$ is a hyperparameter. ELU is smooth everywhere and has a nonzero gradient for $x < 0$, mitigating the dying-neuron issue while retaining the non-saturating behavior of ReLU for positive inputs.

## 5.6 Skip Connections and Residual Networks

Training very deep networks (tens or hundreds of layers) is difficult because gradients must flow back through every layer without vanishing. **Skip (residual) connections**, introduced by He et al. (*CVPR*, 2016), add the input of a block directly to its output:

$$f_i^{(k)}(\mathbf{x}) = f_i^{(k-1)}(\mathbf{x}) + \sigma\!\left(w_{i,0}^{(k)} + \sum_j w_{i,j}^{(k)} f_j^{(k-1)}(\mathbf{x})\right).$$

The Jacobian of $f_i^{(k)}$ with respect to $f_j^{(k-1)}$ is

$$\frac{\partial f_i^{(k)}}{\partial f_j^{(k-1)}} = \delta_{ij} + \sigma'\!\left(z_i^{(k)}\right) w_{i,j}^{(k)},$$

where $z_i^{(k)} = w_{i,0}^{(k)} + \sum_j w_{i,j}^{(k)} f_j^{(k-1)}$ is the pre-activation and $\delta_{ij}$ is the Kronecker delta. The identity term $\delta_{ij}$ guarantees that the gradient is at least 1 along the diagonal even if $\sigma'$ vanishes, so gradients can propagate without exponential decay. This identity path also smooths the loss landscape and enables architectures hundreds of layers deep (ResNets, Transformers).

## 5.7 Batch Normalization

During training, the distribution of each layer's pre-activations shifts as weights change — a phenomenon called **internal covariate shift** — which slows convergence and makes the network sensitive to the choice of learning rate. **Batch normalization** addresses this by standardizing the activations of each hidden unit $j$ in layer $k$ across the current mini-batch of $P'$ examples.

For unit $j$ at layer $k$, collect activations $\{f_j^{(k)}(\mathbf{x}^{(p)})\}_{p=1}^{P'}$ and compute the mini-batch mean and variance:

$$\mu_{f_j^{(k)}} = \frac{1}{P'}\sum_{p=1}^{P'} f_j^{(k)}(\mathbf{x}^{(p)}), \qquad \sigma^2_{f_j^{(k)}} = \frac{1}{P'}\sum_{p=1}^{P'} \left(f_j^{(k)}(\mathbf{x}^{(p)}) - \mu_{f_j^{(k)}}\right)^2.$$

The normalized activation is $\hat{f}_j^{(k)} = (f_j^{(k)} - \mu_{f_j^{(k)}})/\sigma_{f_j^{(k)}}$. To prevent this normalization from permanently restricting the representational capacity of the network, two **learnable** scalar parameters $\gamma_j^{(k)}$ (scale) and $\varepsilon_j^{(k)}$ (shift) are introduced, and the output of the batch normalization layer is

$$\tilde{f}_j^{(k)} = \gamma_j^{(k)} \hat{f}_j^{(k)} + \varepsilon_j^{(k)}.$$

This allows the network to recover any mean and variance if that is optimal. Because mean and variance are re-computed at every weight update for every layer, batch normalization layers are typically inserted between affine transformations and activation functions. During inference, the running mean and variance accumulated over training are used rather than the current mini-batch statistics.

## 5.8 Optimization: Gradient Descent and Adam

The training loss for a network with parameters $\omega$ over $P$ data points is

$$\mathcal{L}(\omega) = \frac{1}{P} \sum_{p=1}^{P} \ell\!\left(\hat{f}(\mathbf{x}^{(p)}, \omega),\, y^{(p)}\right),$$

where $\ell$ is a pointwise loss (e.g. squared error). Deep network losses are almost always **non-convex**, so no closed-form solution exists and iterative gradient-based methods are required. Computing the full gradient $\nabla_\omega \mathcal{L}$ costs $\mathcal{O}(P)$ and is infeasible for large datasets.

**Stochastic gradient descent (SGD)** replaces the full gradient with an estimate computed over a random mini-batch $\mathcal{B}$ of $P' \ll P$ examples:

$$g = \frac{1}{P'} \sum_{p \in \mathcal{B}} \nabla_\omega \ell\!\left(\hat{f}(\mathbf{x}^{(p)}, \omega), y^{(p)}\right), \qquad \omega_{k+1} = \omega_k - \varepsilon\, g,$$

where $\varepsilon > 0$ is the **learning rate**. One pass through the full dataset is called an **epoch**; a single gradient step is a **step**. Gradients of the composition of layer operations are computed efficiently by **backpropagation** (reverse-mode automatic differentiation), which propagates error signals layer by layer from output to input.

A limitation of plain SGD is that in elongated loss landscapes the negative gradient zig-zags across the valley rather than pointing down it. The **Adam optimizer** addresses this by maintaining exponentially weighted averages of the gradient $d_k^i$ (first moment) and of the squared gradient $h_k^i$ (second moment) for each parameter $i$:

$$d_k^i = \omega_1 d_{k-1}^i + (1 - \omega_1)\, g_i(\omega_k), \qquad d_0^i = g_i(\omega_0),$$

$$h_k^i = \omega_2 h_{k-1}^i + (1 - \omega_2)\, g_i(\omega_k)^2, \qquad h_0^i = g_i(\omega_0)^2,$$

where $\omega_1$ and $\omega_2$ are momentum hyperparameters with defaults $\omega_1 = 0.9$ and $\omega_2 = 0.999$. The update is

$$\omega_{k+1}^i = \omega_k^i - \varepsilon\, \frac{d_k^i}{\sqrt{h_k^i}}.$$

Dividing by $\sqrt{h_k^i}$ normalizes the step by the recent gradient magnitude, so directions with consistently large gradients are slowed and directions with small gradients are amplified — effectively adapting the learning rate per parameter. In practice, bias-correction factors are applied to $d_k^i$ and $h_k^i$ to account for their initialization at zero.

**Learning-rate schedulers** adjust $\varepsilon$ during training according to a predetermined or adaptive rule. A single fixed $\varepsilon$ works poorly across an entire run: a value large enough for fast early progress will overshoot minima later, while a value small enough for precise late-stage convergence makes early training unnecessarily slow. Schedulers resolve this tension. Four standard strategies are used in practice:

- **Exponential decay** multiplies $\varepsilon$ by a constant factor $\gamma < 1$ every $n$ steps, giving $\varepsilon_k = \varepsilon_0 \gamma^{k/n}$. The rate decays steadily regardless of the loss, which is simple but inflexible.
- **ReduceLROnPlateau** monitors the validation loss and reduces $\varepsilon$ by a fixed factor whenever the loss has not improved for a specified number of epochs. It is reactive — the rate only falls when progress genuinely stalls.
- **Cosine annealing with warm restarts** follows a cosine curve from $\varepsilon_{\max}$ down to $\varepsilon_{\min}$ over a cycle of $T$ steps, then resets to $\varepsilon_{\max}$. The periodic restarts allow the optimizer to escape sharp local minima by briefly taking larger steps again before reconverging.
- **Cyclic learning rate** oscillates $\varepsilon$ between a minimum and maximum value. Counterintuitively, periodically increasing the rate can help escape saddle points and sharp minima, often yielding faster convergence than monotone decay.

## 5.9 Regularization

Neural networks are highly overparameterized and prone to overfitting. Several strategies reduce this:

**Data augmentation** exploits known symmetries of the problem to generate additional training examples synthetically. For images, this includes rotations, flips, and color jitter; for molecules, this includes permutation of equivalent atoms. Augmentation is the most principled regularization: it directly encodes prior knowledge. Care must be taken not to apply transformations that change the label (e.g. reflecting the letter "b" to "d" in handwriting recognition).

**Early stopping** trains a high-capacity model and monitors validation loss at regular intervals. Each time the validation loss reaches a new minimum, the current parameters are saved. Training is halted when the validation loss has not improved for $P_{\text{patience}}$ consecutive checkpoints, and the saved best parameters are restored. The algorithm depends on a patience hyperparameter $P_{\text{patience}}$ and a check frequency, but requires no modification to the loss or architecture.

**Dropout** addresses a specific failure mode called co-adaptation: during training, units can become mutually dependent — unit $A$ learns to correct the errors of unit $B$ and vice versa — which causes the network to overfit because that cooperation only works on training data. Dropout breaks this by randomly zeroing each non-output unit independently with probability $p_{\text{drop}}$ on every forward pass during training, so no unit can rely on any specific partner being present. This forces each unit to be individually useful and learn redundant, general-purpose representations.

At inference, all units are active and their outputs are scaled by $(1 - p_{\text{drop}})$ to match the expected activation magnitude seen during training (since on average a fraction $p_{\text{drop}}$ of inputs were zeroed). Dropout has a clean ensemble interpretation: with $N$ droppable units there are $2^N$ possible binary masks, and training with dropout samples a different sub-network at each step — all sharing weights. Inference with weight scaling approximates averaging the predictions of all $2^N$ sub-networks simultaneously. Unlike ordinary bagging, no separate training run is needed for each sub-network, making dropout computationally cheap.

**Weight decay (L2 regularization)** adds a penalty $\lambda \|\omega\|^2$ to the loss, keeping weights small and discouraging the network from relying on any single parameter. L1 regularization ($\lambda \|\omega\|_1$) promotes sparsity.

---

