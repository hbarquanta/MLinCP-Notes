# Chapter 4: Regression & Uncertainty Quantification

The central task of supervised regression is to learn a model $\hat{f}(\boldsymbol{x};\Theta) \approx y$ from a training set $\{(\boldsymbol{x}_p, y_p)\}_{p=1}^{P}$ of $P$ input–output pairs, where $\boldsymbol{x}_p \in \mathbb{R}^N$ is an $N$-dimensional feature vector and $y_p \in \mathbb{R}$ is the scalar label. The model is written as a linear combination of $B$ nonlinear basis functions:

$$\hat{f}(\boldsymbol{x}_p;\Theta) = w_0 + \sum_{i=1}^{B} f_i(\boldsymbol{x}_p)\, w_i = b + \boldsymbol{f}_p^T \boldsymbol{w}$$

where $b = w_0$ is a bias, $\boldsymbol{w} = (w_1,\ldots,w_B)^T$ is the weight vector, and $\boldsymbol{f}_p = (f_1(\boldsymbol{x}_p),\ldots,f_B(\boldsymbol{x}_p))^T$ is the basis-function vector evaluated at point $p$. The parameter set is $\Theta = (b, \boldsymbol{w},\text{ any parameters inside }f_i)$.

## 4.1 Multivariate Linear Regression

**Linear regression** is the special case $B = N$ where the basis functions are the raw input features: $f_i(\boldsymbol{x}) = x_i$. The model is then simply

$$\hat{f}(\boldsymbol{x}_p;\boldsymbol{w}) = \boldsymbol{x}_p^T \boldsymbol{w}$$

where the first component of $\boldsymbol{x}_p$ is set to 1 to absorb the bias. The training objective is to minimize the mean squared loss over all $P$ data points:

$$g(\boldsymbol{w}) = \frac{1}{P}\sum_{p=1}^{P}\!\left(\boldsymbol{x}_p^T\boldsymbol{w} - y_p\right)^2$$

This is convex in $\boldsymbol{w}$ and has a closed-form solution, the **normal equations**:

$$\boldsymbol{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\boldsymbol{y}$$

where $\mathbf{X} \in \mathbb{R}^{P \times N}$ is the data matrix with rows $\boldsymbol{x}_p^T$ and $\boldsymbol{y} = (y_1,\ldots,y_P)^T$.

**Multivariate nonlinear regression** generalizes this by replacing $\boldsymbol{x}_p$ with a $B$-dimensional vector of nonlinear functions $\boldsymbol{f}_p$, while the model remains linear in $\boldsymbol{w}$. The loss becomes

$$g(\Theta) = \frac{1}{P}\sum_{p=1}^{P}\!\left(\boldsymbol{f}_p^T\boldsymbol{w} - y_p\right)^2$$

Stacking the $\boldsymbol{f}_p^T$ as rows into the $P \times B$ feature matrix $\mathbf{F}$, the normal equations generalize to $\boldsymbol{w}^* = (\mathbf{F}^T\mathbf{F})^{-1}\mathbf{F}^T\boldsymbol{y}$.

**Universal approximation.** If $B \geq N$ and at least $N$ of the basis functions are linearly independent, the spanning set has sufficient capacity to act as a **universal approximator**: it can fit any smooth function to arbitrary precision given enough data. Three families of universal approximators exist. *Fixed-shape approximators* use a predetermined set of functions (polynomials, Fourier modes, radial basis functions); the basis is fixed before training, and only the weights $\boldsymbol{w}$ are learned. *Neural networks* stack nonlinear layers whose architecture is fixed but whose weights are trained by gradient descent. *Tree-based approximators* (decision trees, random forests, gradient boosting) recursively partition the feature space and are particularly effective for tabular data. However, a degree-$D$ Fourier basis on an $N$-dimensional input has $B = (2D+1)^N - 1$ terms, growing exponentially with dimension, which is the **scaling problem**. The kernel trick, described next, circumvents this for fixed-shape approximators.

**Parametric and non-parametric models.** Universal approximators split into two structural classes. In a *parametric model*, the number of parameters is fixed in advance and does not grow with the size of the training set: polynomial, Fourier, and neural-network models are parametric because adding more training data changes the weight values but not the weight count. In a *non-parametric model*, the parameter count grows with $P$: the KRR/GPR dual vector $\boldsymbol{\alpha} \in \mathbb{R}^P$ has one entry per training point, so the model size scales with the dataset. Non-parametric models are more expressive but harder to scale; parametric models scale to large datasets at the cost of fixing an architecture capacity in advance.

## 4.2 The Kernel Trick

Recall the setup: we have $P$ training points indexed by $p = 1, \ldots, P$, each with a $B$-dimensional feature vector $\boldsymbol{f}_p$, and we want to fit the weight vector $\boldsymbol{w} \in \mathbb{R}^B$. When the basis is a high-degree polynomial or Fourier set, $B$ can be astronomically large (e.g. $B \sim 10^{11}$ for degree-4 Fourier on 12 inputs), making direct optimization of $\boldsymbol{w}$ infeasible.

The key insight is that $\boldsymbol{w}$ can always be decomposed as $\boldsymbol{w} = \mathbf{F}^T\boldsymbol{\alpha} + \boldsymbol{r}$, where $\mathbf{F} \in \mathbb{R}^{P \times B}$ is the same feature matrix from Section 4.1 (rows $\boldsymbol{f}_p^T$), $\boldsymbol{\alpha} \in \mathbb{R}^P$ is a new coefficient vector with one entry per training point, and $\boldsymbol{r}$ is orthogonal to the row space of $\mathbf{F}$ (i.e. $\mathbf{F}\boldsymbol{r} = \mathbf{0}$). Since $\boldsymbol{r}$ is orthogonal to every training point's feature vector, it contributes nothing to any prediction and can be dropped. The model then becomes

$$\hat{f}(\boldsymbol{x}_p; b, \boldsymbol{\alpha}) = b + \boldsymbol{k}_p^T\boldsymbol{\alpha} = b + \sum_{p'=1}^{P} k(\boldsymbol{x}_p, \boldsymbol{x}_{p'})\,\alpha_{p'}$$

where $k(\boldsymbol{x}_p, \boldsymbol{x}_{p'}) = \boldsymbol{f}_p^T\boldsymbol{f}_{p'}$ is the **kernel** (the inner product of the two feature vectors in the basis space), and $\boldsymbol{k}_p \in \mathbb{R}^P$ is the vector of kernel values between $\boldsymbol{x}_p$ and all training points. The $P \times P$ symmetric **kernel matrix** $\mathbf{K}$ collects all pairwise kernel values: $K_{pp'} = k(\boldsymbol{x}_p, \boldsymbol{x}_{p'})$.

**What is the actual benefit?** The weight vector $\boldsymbol{w}$ lives in $\mathbb{R}^B$, and $B$ can be enormous or even infinite. The dual vector $\boldsymbol{\alpha}$ lives in $\mathbb{R}^P$, and $P$ is just the number of training points, typically far smaller than $B$. More importantly, computing a single kernel entry $k(\boldsymbol{x}_p, \boldsymbol{x}_{p'}) = \boldsymbol{f}_p^T\boldsymbol{f}_{p'}$ often reduces to a simple closed-form formula in the original inputs $\boldsymbol{x}_p, \boldsymbol{x}_{p'} \in \mathbb{R}^N$, costing only $O(N)$ regardless of how large $B$ is. The explicit feature vectors $\boldsymbol{f}_p$ are never constructed. This is the kernel trick: you implicitly work in a very high (or infinite) dimensional feature space, while only ever computing $O(N)$ inner products.

**Generality of the kernel trick.** The substitution $\boldsymbol{w} = \mathbf{F}^T\boldsymbol{\alpha}$ is not specific to squared loss. Any loss that depends on $\boldsymbol{w}$ only through inner products with training feature vectors can be kernelized, because $\boldsymbol{f}_p^T\boldsymbol{w} = \boldsymbol{f}_p^T\mathbf{F}^T\boldsymbol{\alpha} = \boldsymbol{k}_p^T\boldsymbol{\alpha}$ is a pure kernel expression. This means the kernel trick applies equally to logistic regression (replacing the squared loss with the log-loss $\log(1 + e^{-y_p(b + \boldsymbol{k}_p^T\boldsymbol{\alpha})})$) and to support vector machines (hinge loss $\max(0, 1 - y_p(b + \boldsymbol{k}_p^T\boldsymbol{\alpha}))$). In every case the explicit feature vectors $\boldsymbol{f}_p$ never need to be constructed; only pairwise kernel values appear. The choice of loss function determines the decision boundary geometry; the kernel determines the notion of similarity. The two are independently configurable.

**Popular kernels.** All three are written in terms of a pair of training inputs $\boldsymbol{x}_p, \boldsymbol{x}_{p'} \in \mathbb{R}^N$:

- **Polynomial kernel** of degree $D$: $k(\boldsymbol{x}_p, \boldsymbol{x}_{p'}) = (1 + \boldsymbol{x}_p^T\boldsymbol{x}_{p'})^D - 1$. Implicitly evaluates all monomials up to degree $D$ at cost $O(N)$ per entry, regardless of the $B = (D+1)^N$ explicit features.
- **Fourier (Dirichlet) kernel** (1D, $N=1$, scalar inputs $x_p, x_{p'} \in \mathbb{R}$): $k(x_p, x_{p'}) = \dfrac{\sin\!\left[(2D+1)\pi(x_p - x_{p'})\right]}{\sin\!\left[\pi(x_p - x_{p'})\right]} - 1$. Arises in band-limited signal reconstruction, lattice sums, and diffraction patterns.
- **RBF (Gaussian) kernel**: $k(\boldsymbol{x}_p, \boldsymbol{x}_{p'}) = e^{-\beta\|\boldsymbol{x}_p - \boldsymbol{x}_{p'}\|_2^2}$, where $\beta > 0$ controls the length scale. Small $\beta$ gives broad Gaussians (smooth, risk of underfitting); large $\beta$ gives narrow Gaussians (localized, risk of overfitting). Corresponds to an infinite-dimensional feature map.

The choice of kernel encodes the notion of similarity appropriate for the data. For physics applications: translation-invariant phenomena suit Fourier or RBF; distance-based atomic-environment descriptors (as in GAP, FCHL) suit RBF.

## 4.3 Kernel Ridge Regression (KRR)

**The conceptual picture.** Suppose you have $P$ training atomic environments with known energies $y_1, \ldots, y_P$, and you want to predict the energy of a new environment described by $\boldsymbol{x}_*$. The core idea of KRR is simple: the prediction is a weighted sum of similarities to all training environments,

$$\hat{f}(\boldsymbol{x}_*) = \sum_{p=1}^{P} \alpha_p\, k(\boldsymbol{x}_*, \boldsymbol{x}_p),$$

where $k(\boldsymbol{x}_*, \boldsymbol{x}_p)$ measures how similar the new environment is to training environment $p$, and $\alpha_p$ is a learned coefficient. If $\boldsymbol{x}_*$ closely resembles training point $p$, that point contributes strongly to the prediction; distant training points contribute little. The kernel function $k$ encodes what "similar" means: with SOAP descriptors and an RBF kernel, two environments are similar if their SOAP vectors are close in Euclidean distance, which corresponds to physically similar local geometries.

Training amounts to finding the right coefficients $\boldsymbol{\alpha} = (\alpha_1, \ldots, \alpha_P)^T$. The pipeline in practice is:
1. Compute a descriptor $\boldsymbol{x}_p \in \mathbb{R}^d$ (e.g. SOAP or ACE) for each training environment $p = 1,\ldots,P$.
2. Evaluate the kernel on all $P^2$ pairs to build the $P \times P$ kernel matrix $\mathbf{K}$ with $K_{pp'} = k(\boldsymbol{x}_p, \boldsymbol{x}_{p'})$.
3. Solve for $\boldsymbol{\alpha}$ (one matrix inversion; no gradient descent, no epochs).
4. Predict for any new environment by computing $\mathbf{k}_*$ and taking the dot product $\mathbf{k}_*^T\boldsymbol{\alpha}$.

**Derivation.** Using the kernelized prediction $\hat{f}(\boldsymbol{x}_p; \boldsymbol{\alpha}) = \boldsymbol{k}_p^T\boldsymbol{\alpha}$ from Section 4.2 (absorbing the bias for brevity), the regularized squared loss is

$$g(\boldsymbol{\alpha}) = (\mathbf{K}\boldsymbol{\alpha} - \boldsymbol{y})^T(\mathbf{K}\boldsymbol{\alpha} - \boldsymbol{y}) + \lambda\,\boldsymbol{\alpha}^T\mathbf{K}\boldsymbol{\alpha}$$

The first term is the total squared prediction error (the vector of all training predictions is $\mathbf{K}\boldsymbol{\alpha}$ since $\hat{f}(\boldsymbol{x}_p) = (\mathbf{K}\boldsymbol{\alpha})_p$). The second term penalizes large coefficients. Setting $\nabla_{\boldsymbol{\alpha}} g = 0$:

$$2\mathbf{K}(\mathbf{K}\boldsymbol{\alpha} - \boldsymbol{y}) + 2\lambda\mathbf{K}\boldsymbol{\alpha} = 0$$

Since $\mathbf{K}$ is symmetric positive semi-definite we can factor it out, giving the linear system $(\mathbf{K} + \lambda\mathbf{I})\boldsymbol{\alpha} = \boldsymbol{y}$, with solution

$$\boldsymbol{\alpha} = (\mathbf{K} + \lambda\mathbf{I})^{-1}\boldsymbol{y}$$

where $\lambda \geq 0$ is the regularization parameter, the same $\lambda$ as in §4.5. Prediction at any new point uses $\mathbf{k}_* \in \mathbb{R}^P$ with $(\mathbf{k}_*)_p = k(\boldsymbol{x}_*, \boldsymbol{x}_p)$:

$$\hat{f}(\boldsymbol{x}_*) = \mathbf{k}_*^T\boldsymbol{\alpha} = \sum_{p=1}^{P} \alpha_p\, k(\boldsymbol{x}_*, \boldsymbol{x}_p)$$

**How the kernels plug in.** The derivation never assumed anything specific about $k$; it only appears when filling $\mathbf{K}$ entry by entry. This is exactly where the kernel trick pays off: each entry costs $\mathcal{O}(N)$ to evaluate regardless of the implicit feature space dimension. For the polynomial kernel that implicit space has $(D+1)^N$ dimensions; for the RBF kernel it is infinite, yet $\mathbf{K}$ is always a finite $P \times P$ matrix. Once $\mathbf{K}$ is built, the solve $\boldsymbol{\alpha} = (\mathbf{K}+\lambda\mathbf{I})^{-1}\boldsymbol{y}$ is identical no matter which kernel was used. Swapping kernels changes the inductive bias (what notion of similarity is used) without changing the training algorithm. For SOAP descriptors, the RBF kernel is natural: physically similar local environments → nearby SOAP vectors → high $k$ → strong influence on predictions near that environment.

**Computational cost and scaling.** KRR requires forming the $P \times P$ kernel matrix $\mathbf{K}$ and inverting it once. This costs $\mathcal{O}(P^3)$ in time and $\mathcal{O}(P^2)$ in memory. Crucially, the training procedure is a single matrix solve; there are no epochs, no gradient descent, no hyperparameters like learning rate. This is simple and numerically reproducible, which matters for scientific applications.

The $\mathcal{O}(P^3)$ scaling is, however, a hard limit. If you double the training set, training time increases eightfold. At $P \sim 10^3$ the solve takes milliseconds; at $P \sim 10^4$ it becomes slow; at $P \sim 10^5$ it is infeasible on standard hardware. For comparison, a deep learning model like MACE trains via mini-batch stochastic gradient descent: each update touches only a small batch of structures and the cost per step scales roughly linearly in the batch size and number of neighbors per atom (controlled by $r_\text{cut}$). The number of model parameters is fixed regardless of dataset size. This allows MACE to train on datasets of $10^6$ or more structures, a regime completely inaccessible to exact KRR. The trade-off is that KRR gives an analytic, globally optimal solution (for fixed $\lambda$), whereas MACE requires iterative optimization over many epochs and is sensitive to initialization and optimizer hyperparameters.

## 4.4 Gaussian Process Regression (GPR)

**The conceptual picture.** GPR is the Bayesian version of KRR. Instead of just finding a single best-fit function, GPR maintains a probability distribution over all functions consistent with the training data. Before seeing any data you specify a **prior** over functions: a belief about what the function looks like. After observing the training labels you update this to a **posterior** via Bayes' theorem. The posterior gives both a mean prediction (identical to KRR) and a principled uncertainty estimate: how confident the model is at each test point. The uncertainty is large where training data is sparse and small where data is dense, automatically, without any extra computation beyond what KRR already does.

**What is a Gaussian process?** A Gaussian process (GP) is a probability distribution over functions. The defining property is that for any finite collection of inputs $\{\boldsymbol{x}_1, \ldots, \boldsymbol{x}_P\}$, the corresponding function values $\boldsymbol{f} = (f(\boldsymbol{x}_1), \ldots, f(\boldsymbol{x}_P))^T$ follow a multivariate Gaussian distribution. A GP is fully specified by two objects:

- a **mean function** $m(\boldsymbol{x}) = \mathbb{E}[f(\boldsymbol{x})]$, set to zero in most regression applications
- a **covariance kernel** $k(\boldsymbol{x}, \boldsymbol{x}') = \mathrm{Cov}[f(\boldsymbol{x}), f(\boldsymbol{x}')]$, which encodes how strongly function values at two inputs are correlated

We write $f \sim \mathcal{GP}(0, k)$. For any finite set of training inputs, the prior over $\boldsymbol{f}$ is then $\boldsymbol{f} \sim \mathcal{N}(\boldsymbol{0}, \mathbf{K})$, where $\mathbf{K}$ is the $P \times P$ kernel matrix with $K_{pp'} = k(\boldsymbol{x}_p, \boldsymbol{x}_{p'})$, the same matrix as in KRR.

The kernel hyperparameters carry physical meaning. For the RBF kernel $k(\boldsymbol{x},\boldsymbol{x}') = v\,e^{-\sum_i (x_i - x_i')^2 / (2\ell_i^2)}$: $v > 0$ is the **signal variance**, controlling how much $f$ fluctuates about its mean; $\ell_i > 0$ is the **length scale** in the $i$-th input dimension, controlling smoothness; larger $\ell_i$ means the function varies more slowly along that direction.

**From prior to posterior.** Suppose we observe $P$ noisy measurements $\boldsymbol{y} = \boldsymbol{f} + \boldsymbol{\varepsilon}$ where $\boldsymbol{\varepsilon} \sim \mathcal{N}(\boldsymbol{0}, \lambda\mathbf{I})$ is independent Gaussian noise with variance $\lambda$. The joint distribution of the training observations $\boldsymbol{y}$ and the function value $f_* = f(\boldsymbol{x}_*)$ at a new test point $\boldsymbol{x}_*$ is

$$\begin{pmatrix} \boldsymbol{y} \\ f_* \end{pmatrix} \sim \mathcal{N}\!\left(\boldsymbol{0},\; \begin{pmatrix} \mathbf{K} + \lambda\mathbf{I} & \mathbf{k}_* \\ \mathbf{k}_*^T & k_{**} \end{pmatrix}\right)$$

where $\mathbf{k}_* \in \mathbb{R}^P$ is the same test-point kernel vector defined in Section 4.3, with entries $(\mathbf{k}_*)_p = k(\boldsymbol{x}_*, \boldsymbol{x}_p)$, and $k_{**} = k(\boldsymbol{x}_*, \boldsymbol{x}_*)$ is the prior variance at the test point. Conditioning this joint Gaussian on the observed $\boldsymbol{y}$ (via the standard Gaussian conditioning formula) yields the posterior, which is also Gaussian with mean and variance:

$$\mu(\boldsymbol{x}_*) = \mathbf{k}_*^T(\mathbf{K} + \lambda\mathbf{I})^{-1}\boldsymbol{y}$$

$$\sigma^2(\boldsymbol{x}_*) = k_{**} - \mathbf{k}_*^T(\mathbf{K} + \lambda\mathbf{I})^{-1}\mathbf{k}_*$$

The predictive mean $\mu(\boldsymbol{x}_*) = \mathbf{k}_*^T\boldsymbol{\alpha}$ is identical to the KRR prediction from §4.3; GPR and KRR agree on the mean, but GPR additionally provides a principled uncertainty estimate. The predictive variance $\sigma^2(\boldsymbol{x}_*)$ has a clear interpretation: the first term $k_{**}$ is the prior variance (how uncertain we were before seeing any data), and the second term $\mathbf{k}_*^T(\mathbf{K}+\lambda\mathbf{I})^{-1}\mathbf{k}_* \geq 0$ is the variance reduction due to the training data. In data-dense regions $\mathbf{k}_*$ is large and the reduction is large, giving low uncertainty; in data-sparse regions $\mathbf{k}_* \approx \boldsymbol{0}$ and $\sigma^2 \approx k_{**}$, recovering the prior uncertainty.

**Hyperparameter tuning.** The kernel hyperparameters ($v$, $\ell_i$) and noise level $\lambda$ can be optimized by maximizing the **log-marginal likelihood** of the training data:

$$\log p(\boldsymbol{y}) = -\tfrac{1}{2}\boldsymbol{y}^T(\mathbf{K}+\lambda\mathbf{I})^{-1}\boldsymbol{y} - \tfrac{1}{2}\log\det(\mathbf{K}+\lambda\mathbf{I}) - \tfrac{P}{2}\log 2\pi$$

This is a single, principled objective that balances data fit (first term) against model complexity (second term, the log-determinant). KRR has no equivalent; $\lambda$ must be set by cross-validation.

**Practical limitation and comparison to deep learning.** Both GPR and KRR require forming and inverting the $P \times P$ kernel matrix, which scales as $\mathcal{O}(P^3)$ in time and $\mathcal{O}(P^2)$ in memory. This limits exact GPR/KRR to training sets of order $P \sim 10^3$–$10^4$ atomic environments, typical for high-quality DFT datasets used in MLIPs. Beyond this, deep learning models are necessary. A GNN such as MACE trains via mini-batch SGD with cost per step linear in batch size and neighbor count, and a fixed parameter count independent of $P$, making it viable at $P \sim 10^6$. The advantage of GPR over deep learning is that uncertainty quantification is analytic and free: $\sigma^2(\boldsymbol{x}_*)$ comes out of the same matrix inversion already performed for prediction. In a GNN, obtaining comparable uncertainty estimates requires additional machinery such as deep ensembles or Monte Carlo dropout. In the MLIP context, GPR with a SOAP kernel is exactly the GAP (Gaussian Approximation Potential) framework (see Chapter 7).

The widget below makes the posterior update tangible. Click anywhere in the plot to add a training point; the shaded band is the ±2σ predictive uncertainty. In empty regions it falls back to the prior (flat band of width $2\sqrt{\sigma_v^2}$); near data it collapses. Adjust the length scale ℓ and noise σₙ to see how they control smoothness and interpolation vs. smoothing behaviour.

<div id="gpr-widget" style="margin:1.5rem 0;">
  <div style="font-size:0.83rem;margin-bottom:0.5rem;opacity:0.75;">
    Click on the plot to place training points; posterior updates immediately.
  </div>
  <div id="gpr-plot" style="width:100%;height:380px;cursor:crosshair;"></div>
  <div style="display:flex;gap:1.5rem;flex-wrap:wrap;align-items:flex-end;margin-top:0.7rem;">
    <div style="font-size:0.85rem;">
      <label>ℓ = <span id="gv-ell">1.00</span></label><br>
      <input type="range" id="gs-ell" min="0.05" max="2" step="0.05" value="1"
             oninput="gprUpdate('ell',+this.value)" style="width:130px;accent-color:#86BCBD;">
    </div>
    <div style="font-size:0.85rem;">
      <label>σᵥ = <span id="gv-sv">1.0</span></label><br>
      <input type="range" id="gs-sv" min="0.1" max="3" step="0.1" value="1"
             oninput="gprUpdate('sv',+this.value)" style="width:110px;accent-color:#86BCBD;">
    </div>
    <div style="font-size:0.85rem;">
      <label>σₙ = <span id="gv-sn">0.100</span></label><br>
      <input type="range" id="gs-sn" min="0.001" max="0.8" step="0.005" value="0.1"
             oninput="gprUpdate('sn',+this.value)" style="width:110px;accent-color:#86BCBD;">
    </div>
    <button onclick="gprClear()"
      style="padding:4px 14px;border-radius:20px;border:1.5px solid #BA5A5A;
             background:transparent;color:#BA5A5A;cursor:pointer;font-size:0.82rem;margin-bottom:2px;">
      Clear
    </button>
  </div>
</div>

<script>
(function(){
  var _xTr=[], _yTr=[];
  var _ell=1.0, _sv=1.0, _sn=0.1;
  var _xG=Array.from({length:200},function(_,i){return i/199;});
  var _ready=false;

  function _k(x1,x2){var d=x1-x2;return _sv*Math.exp(-0.5*d*d/(_ell*_ell));}

  function _matvec(M,v){
    return M.map(function(r){return r.reduce(function(s,m,j){return s+m*v[j];},0);});
  }
  function _matInv(M){
    var n=M.length;
    var a=M.map(function(r,i){
      var row=r.slice();for(var j=0;j<n;j++)row.push(j===i?1:0);return row;
    });
    for(var col=0;col<n;col++){
      var mx=col;
      for(var row=col+1;row<n;row++)if(Math.abs(a[row][col])>Math.abs(a[mx][col]))mx=row;
      var tmp=a[col];a[col]=a[mx];a[mx]=tmp;
      var piv=a[col][col];if(Math.abs(piv)<1e-12)continue;
      for(var j=0;j<2*n;j++)a[col][j]/=piv;
      for(var row=0;row<n;row++){
        if(row===col)continue;
        var f=a[row][col];
        for(var j=0;j<2*n;j++)a[row][j]-=f*a[col][j];
      }
    }
    return a.map(function(r){return r.slice(n);});
  }

  function _gpr(){
    var n=_xTr.length;
    if(n===0){
      var s=Math.sqrt(_sv);
      return{means:_xG.map(function(){return 0;}),stds:_xG.map(function(){return s;})};
    }
    var K=_xTr.map(function(xi){return _xTr.map(function(xj){return _k(xi,xj);});});
    for(var i=0;i<n;i++)K[i][i]+=_sn;
    var Kinv=_matInv(K);
    var alpha=_matvec(Kinv,_yTr);
    var means=_xG.map(function(xs){
      var ks=_xTr.map(function(xi){return _k(xs,xi);});
      return ks.reduce(function(s,k,i){return s+k*alpha[i];},0);
    });
    var stds=_xG.map(function(xs){
      var ks=_xTr.map(function(xi){return _k(xs,xi);});
      var Kk=_matvec(Kinv,ks);
      var v=_sv-ks.reduce(function(s,k,i){return s+k*Kk[i];},0);
      return Math.sqrt(Math.max(0,v));
    });
    return{means:means,stds:stds};
  }

  function _dark(){return document.body&&document.body.getAttribute('data-md-color-scheme')==='slate';}

  function _draw(){
    var el=document.getElementById('gpr-plot');
    if(!el||!window.Plotly)return;
    var res=_gpr();
    var dk=_dark(),bg=dk?'#1e2228':'#ffffff',fg=dk?'#e0e0e0':'#333333';
    var gc=dk?'rgba(255,255,255,0.07)':'rgba(0,0,0,0.07)';
    var hi=res.means.map(function(m,i){return m+2*res.stds[i];});
    var lo=res.means.map(function(m,i){return m-2*res.stds[i];});

    Plotly.react('gpr-plot',[
      {x:_xG.concat(_xG.slice().reverse()),y:hi.concat(lo.slice().reverse()),
       mode:'lines',fill:'toself',fillcolor:'rgba(134,188,189,0.25)',
       line:{color:'transparent'},showlegend:false,hoverinfo:'skip'},
      {x:_xG,y:res.means,mode:'lines',name:'mean',
       line:{color:'#86BCBD',width:2.5},showlegend:false},
      {x:_xTr,y:_yTr,mode:'markers',name:'data',
       marker:{color:'#BA5A5A',size:9,line:{color:'#fff',width:1.5}},showlegend:false},
    ],{
      paper_bgcolor:bg,plot_bgcolor:bg,
      font:{color:fg,family:'inherit',size:12},
      margin:{t:12,b:45,l:50,r:15},
      xaxis:{title:'x',range:[0,1],gridcolor:gc,zerolinecolor:gc},
      yaxis:{title:'y',range:[-4,4],gridcolor:gc,zerolinecolor:gc},
    },{displayModeBar:false,responsive:true});

    if(!_ready){
      _ready=true;
      el.addEventListener('click',function(e){
        var fl=el._fullLayout;if(!fl)return;
        var inner=el.querySelector('.cartesianlayer .xy');if(!inner)return;
        var bb=inner.getBoundingClientRect();
        if(e.clientX<bb.left||e.clientX>bb.right||e.clientY<bb.top||e.clientY>bb.bottom)return;
        var xa=fl.xaxis,ya=fl.yaxis;
        var xd=xa.range[0]+(e.clientX-bb.left)/(bb.right-bb.left)*(xa.range[1]-xa.range[0]);
        var yd=ya.range[1]-(e.clientY-bb.top)/(bb.bottom-bb.top)*(ya.range[1]-ya.range[0]);
        _xTr.push(xd);_yTr.push(yd);_draw();
      });
    }
  }

  window.gprUpdate=function(key,val){
    if(key==='ell'){_ell=val;document.getElementById('gv-ell').textContent=val.toFixed(2);}
    if(key==='sv') {_sv =val;document.getElementById('gv-sv').textContent =val.toFixed(1);}
    if(key==='sn') {_sn =val;document.getElementById('gv-sn').textContent =val.toFixed(3);}
    _draw();
  };
  window.gprClear=function(){_xTr=[];_yTr=[];_draw();};

  function _init(){
    if(!document.getElementById('gpr-plot'))return;
    _ready=false;
    if(!window.Plotly){
      if(!document.getElementById('plotly-cdn')){
        var s=document.createElement('script');s.id='plotly-cdn';
        s.src='https://cdn.plot.ly/plotly-2.27.0.min.js';
        s.onload=function(){_draw();};document.head.appendChild(s);
      }
    } else{_draw();}
  }
  if(typeof document$!=='undefined'){document$.subscribe(function(){setTimeout(_init,80);});}
  else if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',_init);}
  else{_init();}
})();
</script>

## 4.5 Model Selection, Cross-Validation & Ensemble Methods

Two key questions arise when building any regression model: (1) what is the optimal model capacity (how many and what type of basis functions)? and (2) how finely should the weights be optimized? Both questions are answered empirically via **cross-validation** (CV), a procedure that estimates generalization error using only the available labeled data.

**Overfitting** occurs when a high-capacity model fits training noise in addition to the true signal, yielding low training error but high validation error. It is worsened by having too many features, features that correlate poorly with labels, or an over-optimized weight vector.

The widget below makes this concrete: drag the degree slider to watch the polynomial fit change and the error curves evolve. At low $D$ the model underfits (both errors high); at high $D$ it overfits (train error low, generalisation error high); the optimal $D$ minimizes the generalisation curve.

<div id="ov-widget" style="margin:1.5rem 0;">
  <div style="display:flex;gap:0.75rem;align-items:center;margin-bottom:0.65rem;flex-wrap:wrap;">
    <span style="font-size:0.88rem;font-weight:600;">Polynomial degree <em>D</em> =</span>
    <input type="range" id="ov-slider" min="1" max="12" value="3" step="1"
           oninput="ovSetD(+this.value)" style="width:150px;accent-color:#BA5A5A;">
    <span id="ov-dval" style="font-weight:700;min-width:1.5em;">3</span>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;">
    <div id="ov-fit" style="height:280px;"></div>
    <div id="ov-err" style="height:280px;"></div>
  </div>
  <div style="font-size:0.8rem;opacity:0.65;margin-top:0.35rem;">True function: sin(2&#960;x). Left: polynomial fit (blue) vs true function (dashed green) and 16 noisy training points (red). Right: train MSE and generalisation MSE vs degree; dotted line marks current <em>D</em>.</div>
</div>

<script>
(function(){
  function _rng(s){var st=s>>>0;return function(){st=(st*1664525+1013904223)>>>0;return st/0x100000000;};}
  var r=_rng(271828),_N=16,_xTr=[],_yTr=[];
  function _true(x){return Math.sin(2*Math.PI*x);}
  for(var i=0;i<_N;i++){var xi=r();_xTr.push(xi);_yTr.push(_true(xi)+(r()-0.5)*0.7);}
  var _NF=80,_xF=[],_yF=[];
  for(var i=0;i<_NF;i++){var xf=i/(_NF-1);_xF.push(xf);_yF.push(_true(xf));}

  function _mInv(M){
    var n=M.length,a=M.map(function(r,i){var row=r.slice();for(var j=0;j<n;j++)row.push(j===i?1:0);return row;});
    for(var col=0;col<n;col++){
      var mx=col;for(var row=col+1;row<n;row++)if(Math.abs(a[row][col])>Math.abs(a[mx][col]))mx=row;
      var tmp=a[col];a[col]=a[mx];a[mx]=tmp;
      var piv=a[col][col];if(Math.abs(piv)<1e-12)continue;
      for(var j=0;j<2*n;j++)a[col][j]/=piv;
      for(var row=0;row<n;row++){if(row===col)continue;var f=a[row][col];for(var j=0;j<2*n;j++)a[row][j]-=f*a[col][j];}
    }
    return a.map(function(r){return r.slice(n);});
  }
  function _mv(M,v){return M.map(function(r){return r.reduce(function(s,m,j){return s+m*v[j];},0);});}
  function _mm(A,B){return A.map(function(ai){return Array.from({length:B[0].length},function(_,j){return ai.reduce(function(s,_,l){return s+ai[l]*B[l][j];},0);});});}
  function _vander(xs,D){return xs.map(function(x){return Array.from({length:D+1},function(_,i){return Math.pow(x,i);});});}
  function _fit(xs,ys,D){
    var V=_vander(xs,D),Vt=V[0].map(function(_,j){return V.map(function(r){return r[j];});});
    var VtV=_mm(Vt,V);for(var i=0;i<=D;i++)VtV[i][i]+=1e-8;
    return _mv(_mInv(VtV),_mv(Vt,ys));
  }
  function _pred(xs,w){return xs.map(function(x){return w.reduce(function(s,wi,i){return s+wi*Math.pow(x,i);},0);});}
  function _mse(a,b){return a.reduce(function(s,ai,i){return s+(ai-b[i])*(ai-b[i]);},0)/a.length;}

  var _MAXD=12,_eTr=[],_eG=[],_ds=[];
  for(var d=1;d<=_MAXD;d++){_ds.push(d);var w=_fit(_xTr,_yTr,d);_eTr.push(_mse(_yTr,_pred(_xTr,w)));_eG.push(_mse(_yF,_pred(_xF,w)));}
  var _D=3;

  function _dark(){return document.body&&document.body.getAttribute('data-md-color-scheme')==='slate';}
  function _draw(){
    var fe=document.getElementById('ov-fit'),ee=document.getElementById('ov-err');
    if(!fe||!ee||!window.Plotly)return;
    var dk=_dark(),bg=dk?'#1e2228':'#fff',fg=dk?'#e0e0e0':'#333',gc=dk?'rgba(255,255,255,0.07)':'rgba(0,0,0,0.07)';
    var w=_fit(_xTr,_yTr,_D),yFit=_pred(_xF,w).map(function(y){return Math.max(-2.5,Math.min(2.5,y));});
    Plotly.react(fe,[
      {x:_xF,y:_yF,mode:'lines',name:'true f(x)',line:{color:'#A4CE8B',width:1.8,dash:'dot'},showlegend:true},
      {x:_xF,y:yFit,mode:'lines',name:'D = '+_D,line:{color:'#86BCBD',width:2.5},showlegend:true},
      {x:_xTr,y:_yTr,mode:'markers',name:'train',marker:{color:'#BA5A5A',size:7,line:{color:'#fff',width:1}},showlegend:true},
    ],{paper_bgcolor:bg,plot_bgcolor:bg,font:{color:fg,size:10.5},margin:{t:10,b:38,l:38,r:10},
       xaxis:{range:[0,1],title:'x',gridcolor:gc,zerolinecolor:gc},
       yaxis:{range:[-2.5,2.5],title:'y',gridcolor:gc,zerolinecolor:gc},
       legend:{x:0.01,y:0.99,bgcolor:'transparent',font:{size:9.5}},
    },{displayModeBar:false,responsive:true});
    var maxE=Math.max.apply(null,_eG)*1.3||1;
    Plotly.react(ee,[
      {x:_ds,y:_eTr,mode:'lines+markers',name:'train MSE',line:{color:'#A4CE8B',width:2},marker:{size:5}},
      {x:_ds,y:_eG,mode:'lines+markers',name:'generalisation MSE',line:{color:'#BA5A5A',width:2},marker:{size:5}},
      {x:[_D,_D],y:[0,maxE],mode:'lines',line:{color:fg,width:1.2,dash:'dot'},showlegend:false,hoverinfo:'skip'},
    ],{paper_bgcolor:bg,plot_bgcolor:bg,font:{color:fg,size:10.5},margin:{t:10,b:38,l:48,r:10},
       xaxis:{title:'degree D',dtick:2,gridcolor:gc,zerolinecolor:gc},
       yaxis:{title:'MSE',type:'log',gridcolor:gc,zerolinecolor:gc},
       legend:{x:0.98,y:0.98,xanchor:'right',bgcolor:'transparent',font:{size:9.5}},
    },{displayModeBar:false,responsive:true});
  }
  window.ovSetD=function(d){_D=d;document.getElementById('ov-dval').textContent=d;_draw();};
  function _init(){
    if(!document.getElementById('ov-fit'))return;
    if(!window.Plotly){
      if(!document.getElementById('plotly-cdn')){var s=document.createElement('script');s.id='plotly-cdn';s.src='https://cdn.plot.ly/plotly-2.27.0.min.js';s.onload=function(){_draw();};document.head.appendChild(s);}
    }else{_draw();}
  }
  if(typeof document$!=='undefined'){document$.subscribe(function(){setTimeout(_init,80);});}
  else if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',_init);}
  else{_init();}
})();
</script>

Five distinct CV strategies exist, each suited to different situations:

**1. Naive cross-validation.** Partition the dataset once into a training set and a hold-out validation set. Train the model on the training set and evaluate it on the validation set. Simple, but the result depends heavily on the particular split chosen, may miss the ideal model, and is expensive if the model must be trained from scratch for every candidate hyperparameter.

**2. Cross-validation via boosting.** Rather than training a full high-capacity model at once, build it "one unit at a time": start by optimizing only a few parameters (the rest are frozen), then unfreeze more units and optimize more parameters in each successive CV round. Subsequent rounds are cheap because the model is already partially trained. This provides fine-grained control over capacity and is the standard approach for tree-based models.

**3. Cross-validation via regularization** (two strategies):
- *Early stopping*: initialize a high-capacity model with small weights and train it; monitor validation error and stop as soon as it begins to rise. The learning rate must be tuned carefully. Effective for neural networks.
- *Regularized loss*: modify the objective to $g(\Theta) + \lambda h(\Theta)$, where $h(\Theta)$ is a penalty (e.g. $\|\boldsymbol{w}\|^2$ for ridge / Tikhonov regularization) and $\lambda \geq 0$ controls its strength. Begin with large $\lambda$ (simple model), decrease it gradually while fully optimizing each time, and select the $\lambda$ that minimizes validation error. The KRR parameter $\lambda$ in Section 4.3 is exactly this regularizer.

**4. Bagging of cross-validated models.** Train multiple models on different bootstrap subsets (random samples with replacement) of the training data. In regression, take the mean or median of their predictions; in classification, take a majority vote. Models can even be drawn from different model families (different universal approximators). Bagging is embarrassingly parallel and reduces variance, but the resulting ensemble is harder to interpret.

**5. K-fold cross-validation.** Partition the $P$ data points into $K$ equally-sized folds. In each of $K$ rounds, train on the $K-1$ remaining folds and evaluate on the held-out fold. The generalization error estimate is the mean error across rounds, and its standard deviation quantifies uncertainty in the estimate. K-fold CV is used for model validation (accuracy and standard deviation), hyperparameter optimization (grid search or random search), outlier detection, and diagnosing unbalanced datasets. Typical values are $K = 5$ or $K = 10$.

**Leave-one-out cross-validation (LOOCV)** is the limiting case $K = P$: each training run uses all but one data point, and that single held-out point is the validation set. LOOCV is nearly unbiased (it trains on almost the full dataset each time) but has high variance and costs $P$ full training runs, practical only for small datasets or cheap models.

<div id="kf-widget" style="margin:1.5rem 0;">
  <div style="display:flex;gap:0.75rem;align-items:center;margin-bottom:0.9rem;flex-wrap:wrap;">
    <span style="font-size:0.88rem;font-weight:600;"><em>K</em> =</span>
    <input type="range" id="kf-slider" min="2" max="10" value="5" step="1"
           oninput="kfRedraw(+this.value)" style="width:130px;accent-color:#86BCBD;">
    <span id="kf-kval" style="font-weight:700;min-width:1.2em;">5</span>
    <span style="font-size:0.8rem;opacity:0.65;">folds</span>
  </div>
  <div id="kf-grid" style="overflow-x:auto;"></div>
  <div style="display:flex;gap:1.5rem;margin-top:0.65rem;font-size:0.82rem;flex-wrap:wrap;">
    <span><span style="display:inline-block;width:13px;height:13px;background:#A4CE8B;border-radius:3px;vertical-align:middle;margin-right:4px;"></span>Training</span>
    <span><span style="display:inline-block;width:13px;height:13px;background:#86BCBD;border-radius:3px;vertical-align:middle;margin-right:4px;"></span>Validation</span>
  </div>
</div>

<script>
(function(){
  function _dark(){return document.body&&document.body.getAttribute('data-md-color-scheme')==='slate';}
  function kfRedraw(k){
    document.getElementById('kf-kval').textContent=k;
    var grid=document.getElementById('kf-grid');if(!grid)return;
    var dk=_dark(),fg=dk?'#c8c8c8':'#333';
    var bgTra=dk?'rgba(100,160,100,0.3)':'#A4CE8B',bgVal=dk?'rgba(80,150,160,0.45)':'#86BCBD';
    var colTra=dk?'#a0d090':'#2d6020',colVal=dk?'#80d0d8':'#1a4a50';
    var fW=Math.max(48,Math.floor(340/k))+'px';
    var html='<table style="border-collapse:separate;border-spacing:4px;font-size:0.8rem;">';
    html+='<tr><th style="color:'+fg+';padding:2px 6px;font-weight:600;text-align:left;">Split</th>';
    for(var j=1;j<=k;j++)html+='<th style="color:'+fg+';padding:2px 4px;font-weight:500;text-align:center;min-width:'+fW+'">Fold '+j+'</th>';
    html+='<th></th></tr>';
    for(var i=1;i<=k;i++){
      html+='<tr><td style="color:'+fg+';padding:3px 6px;font-weight:600;">'+i+'</td>';
      for(var j=1;j<=k;j++){
        var v=(j===i),bg=v?bgVal:bgTra,col=v?colVal:colTra;
        html+='<td style="padding:3px 4px;"><div style="background:'+bg+';color:'+col+';border-radius:5px;padding:5px 4px;text-align:center;font-weight:'+(v?700:400)+';font-size:0.77rem;">'+(v?'val':'train')+'</div></td>';
      }
      html+='<td style="color:'+fg+';padding:3px 6px;font-size:0.78rem;white-space:nowrap;">→ sp<sub>'+i+'</sub></td></tr>';
    }
    html+='</table>';
    html+='<p style="margin:0.5rem 0 0;font-size:0.82rem;color:'+fg+';opacity:0.8;">Mean of sp<sub>1</sub>…sp<sub>K</sub> estimates generalisation error; the hyperparameter setting that minimizes this mean is selected.</p>';
    grid.innerHTML=html;
  }
  window.kfRedraw=kfRedraw;
  function _init(){if(document.getElementById('kf-grid'))kfRedraw(5);}
  if(typeof document$!=='undefined'){document$.subscribe(function(){setTimeout(_init,80);});}
  else if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',_init);}
  else{_init();}
})();
</script>

**Ensemble methods** combine multiple models to reduce bias and variance simultaneously and, for free, provide an estimate of predictive uncertainty:

| Method | Core idea | Notes |
|--------|-----------|-------|
| Bagging | Train on bootstrap subsets; average predictions | Embarrassingly parallel |
| Boosting | Sequentially fit residuals; combine weak learners | Standard for trees |
| Stacking | Base-model predictions feed a meta-model | Strongest ensembles |
| Voting/Averaging | Simple average or majority vote over existing models | No retraining needed |
| Deep ensembles | Multiple NNs with different random seeds | Standard UQ for NNs |
| Mixtures of experts | Gating network routes inputs to specialist sub-models | Central to LLMs |

Bagging and boosting are complementary: bagging trains models in parallel on resampled data, reducing variance; boosting trains sequentially to reduce bias.

## 4.6 Kernel Methods for ML Interatomic Potentials

Many state-of-the-art ML interatomic potentials (GAP, FCHL, sGDML) are built directly on kernel methods. They work because: (1) atomic-environment descriptors give a moderate feature dimension $N$, while training sets are small enough ($P \ll N$ often) that the $O(P^3)$ cost is manageable; (2) the RBF kernel encodes a physically natural notion of similarity between environments; and (3) the convex training problem is numerically reproducible, which matters for scientific reproducibility. When training sets grow into the millions, kernel methods fail and neural-network potentials take over (covered in Chapter 7 and Chapter 8).

---

