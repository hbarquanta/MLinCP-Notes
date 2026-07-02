# Chapter 3 — Dimensionality Reduction & Clustering

Unsupervised learning extracts structure from unlabeled data $\{x^{(p)}\}_{p=1}^P$, where $x^{(p)} \in \mathbb{R}^D$. The two central tasks are **dimensionality reduction** — finding a compact low-dimensional representation $z^{(p)} \in \mathbb{R}^d$ with $d \ll D$ — and **clustering** — partitioning the data into groups of similar points.

---


---

## 3.1 Principal Component Analysis (PCA)

PCA finds the $K$-dimensional **linear subspace** of $\mathbb{R}^N$ that retains the maximum variance of the data, where $N$ is the feature dimension and $K < N$ is the target (reduced) dimension. The method is motivated by the following observation: if we want to encode data $x_p \in \mathbb{R}^N$ (data point index $p = 1,\ldots,M$, where $M$ is the number of samples) via a spanning set of $K$ orthonormal vectors $c_1,\ldots,c_K \in \mathbb{R}^N$, then the optimal encoding $w_p = C^T x_p \in \mathbb{R}^K$ and decoding $C w_p \approx x_p$ (where $C = [c_1|\cdots|c_K] \in \mathbb{R}^{N\times K}$) minimize the reconstruction loss

$$\mathcal{L} = \sum_p \|x_p - C C^T x_p\|^2$$

when $C$ contains the eigenvectors of the covariance matrix with the $K$ largest eigenvalues. This motivates the following steps.

**Step 1 — Preprocess: center the data.**
Compute the sample mean $\bar{x} = \frac{1}{M}\sum_{p=1}^M x_p \in \mathbb{R}^N$ and subtract it. PCA requires the data to pass through the origin:

$$x_p \leftarrow x_p - \bar{x}$$

Stacking the (now centered) data points as columns gives the data matrix $X \in \mathbb{R}^{N \times M}$.

**Step 2 — Covariance: compute the covariance matrix.**

$$\Sigma = \frac{1}{M} X X^T \in \mathbb{R}^{N \times N}$$

$\Sigma$ is symmetric and positive semi-definite. Its $(i,j)$ entry is the sample covariance between feature dimensions $i$ and $j$.

**Step 3 — Eigen-decomposition: compute eigenvalues and eigenvectors.**

$$\Sigma = V D V^T$$

where $V \in \mathbb{R}^{N \times N}$ is orthogonal (columns are eigenvectors $c_k$) and $D = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_N)$ with eigenvalues sorted $\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_N \ge 0$. The eigenvector with the largest eigenvalue $\lambda_1$ is the 1st principal component; the $k$-th largest eigenvalue gives the $k$-th principal component.

**Step 4 — Variance: choose the number of components $K$.**
The eigenvalue $\lambda_k$ equals the variance of the data projected onto $c_k$. The fraction of total variance retained by $K$ components is:

$$\text{explained variance ratio} = \frac{\sum_{k=1}^K \lambda_k}{\sum_{k=1}^N \lambda_k}$$

There is no single right answer — PCA is a tool for data exploration. Scree plots (plotting $\lambda_k$ vs. $k$) help identify an "elbow" where additional components yield diminishing returns.

**Step 5 — Subspace projection.**
Form the projection matrix $C = [c_1 | c_2 | \cdots | c_K] \in \mathbb{R}^{N\times K}$ from the $K$ selected eigenvectors. Encode each data point:

$$w_p = C^T x_p \in \mathbb{R}^K$$

The full dataset is encoded as $W = C^T X \in \mathbb{R}^{K\times M}$.

**Step 6 — (Optional) Reconstruct.**
The approximate reconstruction from the low-dimensional code is:

$$\hat{x}_p = C w_p = C C^T x_p \approx x_p$$

The per-point reconstruction error is $\|x_p - \hat{x}_p\|^2$; the total equals the discarded variance $\sum_{k=K+1}^N \lambda_k$.

**Limitations.** PCA can only discover linear structure. If the data lies on a curved, nonlinear manifold in $\mathbb{R}^N$, PCA represents it poorly and nonlinear methods are needed.

**Physics context.** In MD simulations, PCA applied to atomic position trajectories identifies the principal modes of motion (e.g., protein conformational changes). TICA (Time-lagged Independent Component Analysis) is a variant that maximizes autocorrelation at a lag time $\tau$, and is better at identifying slow collective variables relevant to rare events.

---

## 3.2 Kernel PCA

Kernel PCA extends PCA to nonlinear manifolds using the **kernel trick**: data is implicitly mapped into a high-dimensional feature space $\mathcal{H}$ via $\phi: \mathbb{R}^N \to \mathcal{H}$, and PCA is performed there. The mapping $\phi$ never needs to be computed explicitly — only pairwise inner products $k(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle_\mathcal{H}$ are needed.

**Gram (kernel) matrix.** Compute the $M \times M$ matrix:

$$K_{ij} = k(x_i, x_j), \quad i,j = 1,\ldots,M$$

**Centering in feature space.** To center the implicit feature vectors, adjust:

$$\tilde{K} = K - \mathbf{1}_M K - K \mathbf{1}_M + \mathbf{1}_M K \mathbf{1}_M$$

where $\mathbf{1}_M \in \mathbb{R}^{M\times M}$ has all entries equal to $1/M$.

**Eigendecomposition.** Solve $\tilde{K}\,\alpha_k = \mu_k\,\alpha_k$ for eigenvectors $\alpha_k \in \mathbb{R}^M$ and eigenvalues $\mu_k$. The $k$-th coordinate of the embedding of a new point $x$ is:

$$w_k(x) = \sum_{p=1}^M \alpha_k^{(p)}\, k(x, x_p)$$

Common kernels: RBF $k(x,x') = \exp(-\|x-x'\|^2 / 2\sigma^2)$; polynomial $k(x,x') = (x^T x' + c)^q$.

Kernel PCA can capture nonlinear structure, but requires $\mathcal{O}(M^2)$ memory and $\mathcal{O}(M^3)$ compute — impractical for large datasets.

---

## 3.3 Autoencoders (Nonlinear Dimensionality Reduction)

The linear autoencoder formulation from PCA — encoding $w_p = C^T x_p$ and decoding $C w_p \approx x_p$ — generalizes directly to nonlinear models. A **nonlinear autoencoder** replaces the linear spanning set $C$ with neural networks:

- An **encoder** compresses $x_p \in \mathbb{R}^N$ to a low-dimensional latent code $w_p \in \mathbb{R}^K$ through a stack of nonlinear layers.
- A **decoder** reconstructs $\hat{x}_p \approx x_p$ from $w_p$ through a second nonlinear network.

Training minimizes the same reconstruction loss $\mathcal{L} = \sum_p \|x_p - \hat{x}_p\|^2$. With nonlinear activations (e.g. ReLU, tanh), the encoder can map data lying on a curved manifold to a flat latent space — something PCA cannot do. The latent code $w_p$ can then be used for visualization, clustering, or downstream supervised learning.

If the encoder and decoder are linear (no activations), the autoencoder's optimal solution is identical to PCA — confirming that PCA is a special case of the autoencoder framework.

**Limitation.** The latent space of a standard autoencoder is not regularized and may be discontinuous or contain holes where the decoder produces unrealistic outputs. Variational autoencoders (VAEs, L9) address this by imposing a probabilistic prior on $w$.

---

## 3.4 t-SNE (t-distributed Stochastic Neighbor Embedding)

t-SNE maps high-dimensional data $X = \{x_1, \ldots, x_N\}$ (where $N$ is the number of data points, each $x_i \in \mathbb{R}^D$ with feature dimension $D$) to a low-dimensional visualization $\{w_i\}$ with $w_i \in \mathbb{R}^2$ or $\mathbb{R}^3$, preserving local neighborhood structure. It should not be used for general dimensionality reduction — **it is primarily a visualization tool**.

**Step 1 — Compute pairwise distances and similarity matrix in $X$.**

Compute Euclidean pairwise distances $d(x_i, x_j)$ for all pairs. Convert to a joint probability distribution capturing pairwise similarities. The conditional probability that point $j$ is a neighbor of point $i$ under a Gaussian centered at $i$ is:

$$p_{j|i} = \frac{\exp\!\left(-d(x_i,x_j)^2 / 2\sigma_i^2\right)}{\displaystyle\sum_{k \neq i} \exp\!\left(-d(x_i,x_k)^2 / 2\sigma_i^2\right)}$$

The bandwidth $\sigma_i$ is chosen per point to achieve a target **perplexity** (typical values 5–50), which controls the effective number of neighbors. Symmetrize: $p_{ij} = (p_{j|i} + p_{i|j}) / 2N$.

**Step 2 — Initialize and define low-dimensional similarities.**

Create an initial set of low-dimensional points $\{w_i\}$. Define similarities in the low-dimensional space using a Student-$t$ distribution with one degree of freedom:

$$q_{ij} = \frac{\left(1 + \|w_i - w_j\|^2\right)^{-1}}{\displaystyle\sum_{k \neq l}\left(1 + \|w_k - w_l\|^2\right)^{-1}}$$

The heavy tail of the $t$-distribution alleviates the **crowding problem**: in low dimensions, insufficient volume would otherwise force moderately similar points too close together.

**Step 3 — Minimize the Kullback-Leibler divergence.**

Optimize $\{w_i\}$ by minimizing:

$$\mathcal{L} = \text{KL}(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

via gradient descent.

The **KL divergence** $\text{KL}(P\|Q)$ measures how much distribution $Q$ differs from a reference distribution $P$. It is always $\ge 0$, and equals zero only when $P = Q$. Importantly, it is **asymmetric**: $\text{KL}(P\|Q) \neq \text{KL}(Q\|P)$. The direction used here penalizes cases where $P$ assigns high probability but $Q$ does not — concretely, if two points are close in high dimensions ($p_{ij}$ large) but far apart in the embedding ($q_{ij}$ small), the log ratio $\log(p_{ij}/q_{ij})$ is large and the penalty is severe. The reverse — pairs that are close in the embedding but were far apart in high dimensions — contributes little to the loss. This asymmetry is what forces t-SNE to prioritize preserving local neighborhoods over global structure.

t-SNE focuses on preserving small pairwise similarities (local structure); PCA instead maximizes large pairwise distances (global variance). As a consequence, global distances and inter-cluster separations in the t-SNE embedding are not meaningful and should not be interpreted.

---

## 3.5 UMAP (Uniform Manifold Approximation and Projection)

UMAP uses tools from topological data analysis to build a fuzzy graph representation of the manifold underlying the data, then optimizes a low-dimensional embedding to match it. Compared to t-SNE, UMAP is faster and better preserves global structure, while both are primarily used for visualization. The axes of either embedding carry no direct physical meaning.

---

### Comparison of dimensionality reduction methods

| Method | Linear? | Preserves global structure | Typical cost | Primary use |
|--------|---------|--------------------------|-------------|------------|
| PCA | Yes | Yes | $\mathcal{O}(MN^2)$ | General reduction, collective variables |
| Kernel PCA | No | Partial | $\mathcal{O}(M^3)$ | Nonlinear, small $M$ |
| Autoencoder | No | Yes (if trained well) | $\mathcal{O}(M)$ per epoch | Large-scale nonlinear |
| t-SNE | No | No | $\mathcal{O}(N\log N)$ | 2D/3D visualization |
| UMAP | No | Partial | $\mathcal{O}(N^{1.14})$ | 2D/3D visualization |

Here $M$ = number of data points, $N$ = feature dimension (PCA context); for t-SNE/UMAP, $N$ = number of data points.

**Physics context.** PCA and TICA applied to MD trajectories yield collective variables (CVs) $\xi$ parameterizing slow conformational changes. The free energy surface along a CV is $F(\xi) = -k_\mathrm{B}T \ln P(\xi)$, where $k_\mathrm{B}$ is Boltzmann's constant, $T$ is temperature, and $P(\xi)$ is the probability density of observing value $\xi$ in an equilibrium simulation.

---


Clustering partitions a dataset $\{x^{(p)}\}_{p=1}^P$ into groups of similar points without any labels. We discuss five families of methods.

---

## 3.6 K-Means Clustering

K-means places $N$ observations into $K$ sets $\mathcal{C} = \{C_1, C_2, \ldots, C_K\}$ (where $N$ is the number of data points, each $x_i \in \mathbb{R}^D$ with feature dimension $D$) by minimizing the total internal cluster variance:

$$\mathcal{L}(\mathcal{C}, \{\mu_k\}) = \sum_{k=1}^K \sum_{i \in C_k} \left\|x_i - \mu_k\right\|^2$$

where $\mu_k = \frac{1}{|C_k|}\sum_{i \in C_k} x_i$ is the centroid (mean) of cluster $C_k$.

**Algorithm (Lloyd's algorithm):**
1. **Initialization**: choose the number of clusters $K$ and randomly assign each of the $N$ points to a cluster (or use K-means++, which greedily places initial centroids far apart to improve convergence).
2. **Distance metric**: choose a distance measure in the feature space (typically Euclidean).
3. **Centroid**: compute the centroid $\mu_k$ of each cluster.
4. **Assignment**: reassign each point to the nearest centroid based on the chosen distance, iterating until the clusters stop changing.

**Convergence.** The loss $\mathcal{L}$ decreases at every step; convergence is guaranteed because the number of possible assignments is finite. However, convergence is only to a **local minimum** — K-means is sensitive to initialization. Running multiple times with different random starts and keeping the lowest-loss result is standard practice.

**Choosing $K$.**
- **Elbow method**: plot $\mathcal{L}$ vs. $K$ and pick the "elbow" where adding more clusters yields diminishing improvement.
- **Silhouette score**: for each point $i$, let $a_i$ = mean intra-cluster distance and $b_i$ = mean distance to the nearest other cluster. The silhouette coefficient $s_i = (b_i - a_i)/\max(a_i, b_i) \in [-1, 1]$; higher = better separated.

**Limitations of K-means:**
1. **No dual membership**: even boundary points must be assigned to exactly one cluster.
2. **Clusters are discrete**: no overlap or nesting between clusters.
3. **Unpredictability**: the algorithm is random and finds only local minima.

Additionally, K-means implicitly assumes clusters are convex and roughly spherical — it fails on non-convex shapes (crescents, rings) and is sensitive to outliers (which pull centroids toward them).

**Gaussian Mixture Models (GMMs) — soft K-means.** GMMs model the data as a mixture of $K$ Gaussians:

$$p(x; \theta) = \sum_{k=1}^K \pi_k\, \mathcal{N}(x;\, \mu_k,\, \Sigma_k)$$

where $\pi_k > 0$ are mixing weights with $\sum_k \pi_k = 1$, $\mu_k \in \mathbb{R}^D$ are cluster means, and $\Sigma_k \in \mathbb{R}^{D \times D}$ are covariance matrices. Unlike K-means, GMMs produce **soft assignments**: the responsibility $r_{ik} = p(z=k \,|\, x_i;\, \theta)$ quantifies the posterior probability that point $x_i$ belongs to cluster $k$. Parameters are estimated by the EM (Expectation-Maximization) algorithm: the E-step computes $r_{ik}$ given current parameters; the M-step updates $\pi_k$, $\mu_k$, $\Sigma_k$ to maximize expected log-likelihood. K-means is the hard-assignment limit of GMMs with isotropic covariances $\Sigma_k = \sigma^2 I$, $\sigma \to 0$.

---

## 3.7 Hierarchical Clustering

Hierarchical clustering produces a tree of nested cluster merges (agglomerative, bottom-up) or splits (divisive, top-down), represented as a **dendrogram**. The desired number of clusters $K$ need not be specified in advance; one cuts the dendrogram at a chosen height to obtain any $K$ from 1 to $N$.

**Agglomerative algorithm:**
1. Initialize: each of the $N$ data points is its own cluster.
2. Compute pairwise distances $d(x_i, x_j)$ between all points.
3. Merge the two clusters $A$ and $B$ with the smallest inter-cluster distance $d(A,B)$.
4. Update the distance matrix to reflect the newly merged cluster.
5. Repeat steps 3–4 until a single cluster remains.

The choice of **linkage criterion** defines $d(A,B)$ and strongly affects the shape of the resulting clusters:

| Linkage | Distance $d(A, B)$ | Typical cluster shape |
|---------|-------------------|-----------------------|
| Single | $\min_{a \in A,\, b \in B} d(a,b)$ | Elongated; prone to chaining |
| Complete | $\max_{a \in A,\, b \in B} d(a,b)$ | Compact, equal-diameter |
| Average (UPGMA) | $\frac{1}{\|A\|\|B\|}\sum_{a \in A, b \in B} d(a,b)$ | Intermediate |
| Ward | $\Delta$ in total within-cluster variance upon merge | Spherical, equal-sized |

**Strengths**: no need to pre-specify $K$; the dendrogram reveals multi-scale cluster structure; works with any distance metric.

**Weaknesses**: $\mathcal{O}(P^2)$ memory for the distance matrix; earlier merges cannot be revised (greedy); sensitive to choice of linkage; does not natively handle outliers.

---

## 3.8 Density-Based Clustering: DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) defines clusters as **dense connected regions**, separated by sparse regions. Points in sparse regions are treated as noise rather than forced into a cluster.

**Parameters**: $\varepsilon > 0$ (neighborhood radius) and $\mathrm{MinPts} \ge 1$ (minimum number of points to define a dense region).

**Point classification:**
- A point $x_i$ is a **core point** if at least $\mathrm{MinPts}$ other points lie within its $\varepsilon$-neighborhood: $|\mathcal{N}_\varepsilon(x_i)| \ge \mathrm{MinPts}$, where $\mathcal{N}_\varepsilon(x_i) = \{j : d(x_i, x_j) \le \varepsilon\}$.
- A point $x_j$ is **density-reachable** from $x_i$ if there exists a chain of core points $x_i = p_1, p_2, \ldots, p_m = x_j$ where each $p_{t+1} \in \mathcal{N}_\varepsilon(p_t)$ and each $p_t$ for $t < m$ is a core point.
- A **border point** is not a core point but is density-reachable from some core point.
- A **noise point** (outlier) is neither core nor border.

**Algorithm:**
1. Classify all points as core, border, or noise using the definitions above.
2. Each core point seeds a cluster; the cluster grows by recursively adding all density-reachable points.
3. Border points are assigned to whichever cluster first reaches them. Noise points are left unlabeled.

**Strengths**: finds clusters of arbitrary shape; automatically identifies and labels outliers; does not require $K$ to be specified.

**Weaknesses**: the single global $\varepsilon$ makes it difficult to handle clusters of varying density; both $\varepsilon$ and $\mathrm{MinPts}$ require careful tuning (a poor choice can merge distinct clusters or split a single cluster into many); naive implementation costs $\mathcal{O}(P^2)$, though this reduces to $\mathcal{O}(P \log P)$ with a spatial index.

---

## 3.9 HDBSCAN (Hierarchical DBSCAN)

HDBSCAN addresses DBSCAN's main limitation — a single global $\varepsilon$ fails when clusters have varying densities — by running DBSCAN over all scales $\varepsilon$ simultaneously, building a cluster hierarchy, and extracting the most persistent (stable) clusters.

**Core distance.** The core distance of point $x_i$ with respect to $\mathrm{MinPts}$ is the distance to its $\mathrm{MinPts}$-th nearest neighbor:

$$d_\mathrm{core}(i) = d(x_i,\; \mathrm{MinPts}\text{-th nearest neighbor of } x_i)$$

Points in dense regions have small core distances; sparse points have large core distances.

**Mutual reachability distance.** The pairwise distance is inflated to:

$$d_\mathrm{mreach}(i, j) = \max\!\left(d_\mathrm{core}(i),\; d_\mathrm{core}(j),\; d(x_i, x_j)\right)$$

This inflates distances for sparse points (whose large core distances dominate), making the method robust to outliers and capable of handling varying cluster densities.

**Algorithm:**
1. Build the minimum spanning tree (MST) of the complete graph on $\{x_i\}$ with edge weights $d_\mathrm{mreach}(i,j)$.
2. Construct the cluster hierarchy (dendrogram) by removing MST edges in decreasing weight order — each removal splits a component into two.
3. **Condense the tree**: track cluster membership as the density scale increases. When a split produces a child with fewer than $\mathrm{MinPts}$ points, those points fall out as noise rather than forming a new cluster.
4. **Extract stable clusters**: define $\lambda = 1/\varepsilon$ (higher $\lambda$ = denser neighborhood). For each cluster $C$ in the condensed tree, the stability is:

$$\text{stability}(C) = \sum_{i \in C} \left(\lambda_i^\mathrm{out} - \lambda_C^\mathrm{in}\right)$$

where $\lambda_C^\mathrm{in}$ is the $\lambda$-value at which cluster $C$ was born and $\lambda_i^\mathrm{out}$ is the $\lambda$-value at which point $i$ left $C$ (into a child cluster or as noise). The selected flat clustering maximizes total stability over the condensed tree.

**Strengths over DBSCAN**: handles varying-density clusters; requires only $\mathrm{MinPts}$ (no $\varepsilon$); returns a full hierarchy; more principled outlier detection.

**Weaknesses**: more complex to implement; worst-case $\mathcal{O}(N^2)$ memory for the distance matrix, though efficient approximations exist.

---

## 3.10 Clustering Method Summary

| Method | Cluster shape | Requires $K$? | Handles outliers | Best suited for |
|--------|-------------|--------------|-----------------|-----------------|
| K-Means | Spherical | Yes | No | Large data, known $K$, convex clusters |
| GMM | Elliptical | Yes | No | Soft assignments, probabilistic model |
| Hierarchical | Any | No (post-hoc cut) | No | Small data, hierarchical structure |
| DBSCAN | Any | No | Yes | Arbitrary shapes, uniform density, outliers |
| HDBSCAN | Any | No | Yes | Arbitrary shapes, varying density — most general |

---

