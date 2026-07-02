# Chapter 9 — Generative Models

## 9.1 Problem Statement
Given data $x \in X$ distributed with $\mu_X(x)$, draw new samples $x' \sim \mu_X$.

**Two regimes**:
- **Learning unknown distributions from data** (molecules, crystals, proteins)
- **Sampling from known distributions** (Boltzmann: $\rho(x) \propto e^{-\beta U(x)}$)

## 9.2 Taxonomy of Generative Models
```
Generative Models
├── Explicit Density
│   ├── Tractable Density
│   │   ├── Autoregressive Models (FVBN, NADE, LLMs)
│   │   └── Normalizing Flows
│   └── Approximate Density
│       ├── Variational Autoencoders (VAEs)
│       └── Diffusion Models (DDPMs)
├── Implicit Density
│   └── Generative Adversarial Networks (GANs)
└── Score-Matching
    ├── Diffusion Models (Score Matching)
    └── Flow Matching
```

## 9.3 Generative Adversarial Networks (GANs)

GANs learn to generate realistic samples by training two networks adversarially against each other, with no explicit probability model. The **generator** $g(z;\theta^g)$ maps random noise $z\sim\mathcal{N}(0,I)$ to data space — it never sees real data directly. The **discriminator** $d(x;\theta^d) \in [0,1]$ tries to tell real from fake. They play a minimax game:

$$v(\theta^g,\theta^d) = \mathbb{E}_{x\sim p_\text{data}}\log d(x) + \mathbb{E}_{z\sim p_z}\log(1-d(g(z)))$$

$$(\theta^g, \theta^d)^* = \arg\min_{\theta^g}\max_{\theta^d}\, v$$

At Nash equilibrium: generator reproduces $p_\text{data}$ exactly and $d(x)=\tfrac{1}{2}$ everywhere. Because there is no explicit $P(x;\theta)$, GANs are **implicit density models** — good at generating, cannot evaluate likelihoods. Training is unstable: (1) **oscillations** from non-convex minimax, (2) **vanishing gradients** when discriminator dominates early (JS-divergence saturates), (3) **mode collapse** when the generator fixates on a subset of modes that fools the discriminator.

## 9.4 Variational Autoencoders (VAEs)

A standard **autoencoder** compresses $x$ into a low-dimensional code $z$ (the **latent space**) and reconstructs it. The latent space is irregular — you cannot sample a random $z$ and decode something meaningful, so standard AEs cannot generate new data.

A **VAE** fixes this by making the encoder output a *distribution* over $z$ instead of a single point: a Gaussian $\mathcal{N}(\mu_\phi(x), \sigma_\phi(x))$. Sampling from this distribution and regularizing it toward $\mathcal{N}(0,I)$ forces the latent space to be smooth, so any randomly drawn $z$ decodes into a realistic sample.

**The generative model**: to generate $x$, first sample a latent code $z$ from a simple **prior** $\mathcal{N}(0,I)$, then pass it through the **decoder** $P(x|z;\theta)$ (a neural network). The total probability of $x$ is obtained by summing (marginalizing) over all possible $z$:

$$P(x;\theta) = \int dz\; \underbrace{P(x|z;\theta)}_{\text{likelihood / decoder}} \underbrace{\mathcal{N}(z;0,I)}_{\text{prior}}$$

**Why this integral is intractable**: $z$ is typically 100-dimensional. Evaluating the integral requires summing over the entire 100D latent space — the integrand is nearly zero almost everywhere, and numerical integration over such a space is hopeless. So we cannot compute $P(x;\theta)$, and therefore cannot maximize $\log P(x;\theta)$ directly.

**Bayes' theorem and the posterior**: applying Bayes gives the distribution over which $z$ values explain a given $x$:

$$\underbrace{P(z|x;\theta)}_{\text{posterior}} = \frac{\overbrace{P(x|z;\theta)}^{\text{likelihood}}\;\overbrace{\mathcal{N}(z;0,I)}^{\text{prior}}}{\underbrace{P(x;\theta)}_{\text{evidence (intractable)}}}$$

The posterior tells us: "given I observed $x$, which latent codes $z$ most likely produced it?" The numerator is computable (decoder times prior), but the denominator is the same intractable integral. We are stuck in a circle: cannot compute the likelihood, cannot compute the posterior.

**The fix — approximate the posterior with the encoder**: instead of computing $P(z|x;\theta)$, approximate it with a Gaussian parametrized by a neural network — the **encoder** $Q(z|x;\phi) = \mathcal{N}(z;\mu_\phi(x),\sigma_\phi(x))$. Feed $x$ in, get a Gaussian, sample $z$. Now optimize a tractable lower bound on $\log P(x;\theta)$ — the ELBO.

### Evidence Lower Bound (ELBO)

Introduce $Q(z|x;\phi)$ inside the log by multiplying and dividing, then apply Jensen's inequality ($\log$ concave $\Rightarrow$ $\log\mathbb{E}[Y]\geq\mathbb{E}[\log Y]$):

$$\log P(x;\theta) = \log\,\mathbb{E}_{z\sim Q}\!\left[\frac{P(x|z;\theta)\,\mathcal{N}(z;0,I)}{Q(z|x;\phi)}\right] \geq \mathbb{E}_{z\sim Q}\!\left[\log\frac{P(x|z;\theta)\,\mathcal{N}(z;0,I)}{Q(z|x;\phi)}\right]$$

Splitting the log fraction gives the ELBO in its standard form:

$$\mathcal{L} = \underbrace{\mathbb{E}_{z\sim Q_\phi}[\log P(x|z;\theta)]}_{\text{reconstruction}} - \underbrace{\text{KL}(Q(z|x;\phi)\,\|\,\mathcal{N}(0,I))}_{\text{regularization}} \leq \log P(x;\theta)$$

The gap to the true log-likelihood is exactly $\text{KL}(Q(z|x;\phi)\,\|\,P(z|x;\theta))\geq 0$ — hence "lower bound". Both terms are now fully tractable: the reconstruction term is a decoder forward pass, and the KL between two Gaussians has a closed-form formula. Maximizing $\mathcal{L}$ simultaneously improves the decoder (reconstruction) and forces the encoder toward the true posterior (the gap shrinks). The KL term also prevents the encoder from collapsing to a deterministic point map, keeping the latent space smooth.

### Reparametrization Trick

The encoder outputs a Gaussian and we sample $z$ from it — but sampling is not differentiable, so backpropagation cannot reach the encoder parameters $\phi$. Fix: push the randomness into a parameter-free noise variable $\varepsilon$:

$$z = \mu_\phi(x) + \sigma_\phi(x)\odot\varepsilon, \qquad \varepsilon\sim\mathcal{N}(0,I)$$

Now $z$ is a deterministic function of $\phi$, and gradients flow through $\mu_\phi$ and $\sigma_\phi$ normally.

**Application**: molecular design (Gómez-Bombarelli et al., ACS Cent. Sci. 2018) — encode SMILES strings into a smooth latent $z$, do gradient ascent on a predicted property, decode back to a molecule.

## 9.5 Autoregressive Models

An autoregressive model generates data sequentially: each new value is predicted from all previously generated values, so the model only ever needs to learn conditional distributions rather than the full joint distribution at once. Unlike VAEs or GANs, this gives an **exact, tractable likelihood** with no approximations:

$$P(x;\theta) = \prod_{i=1}^D P(x_i|x_{<i};\theta), \qquad x_{<i} := \{x_{i-1},\ldots,x_1\}$$

Each conditional is modelled as a Gaussian: $P(x_i|x_{<i};\theta) = \mathcal{N}(x_i;\,\mu_i(x_{<i};\theta),\,\sigma_i(x_{<i};\theta))$. Training maximizes log-likelihood. The key challenge is **how to parametrize** $\mu_i$ and $\sigma_i$ efficiently across all $D$ conditionals:

**FVBN** (Fully Visible Belief Network): each conditional $i$ has its own independent linear weights. Parameters scale as $\mathcal{O}(D^2)$ and capacity is poor.

**NAN** (Nonlinear Autoregressive Network): adds a nonlinear hidden layer $h_i = \tanh(W_i x_{<i} + b_i)$ per conditional. More expressive, but still $\mathcal{O}(D^2)$ parameters — each conditional still has its own weight matrix $W_i$.

**NADE** (Neural Autoregressive Distribution Estimator): the key insight is to **share a single weight matrix** $W$ across all conditionals:

$$h_i = \tanh\!\left(c + \sum_{j<i} W_{:,j}\, x_j\right)$$

The same columns of $W$ are reused for every $h_i$. This makes NADE both expressive and parameter-efficient — $\mathcal{O}(D)$ instead of $\mathcal{O}(D^2)$.

**Application**: **cG-SchNet** applies autoregressive generation to molecules — places atoms one by one, conditioned on target properties (HOMO-LUMO gap, polarizability). Enables inverse molecular design.

## 9.6 Large Language Models (LLMs)

LLMs are autoregressive models over discrete **tokens** (subwords), using a Transformer to compute context-dependent features. They model:

$$P(x;\theta) = \prod_{t=1}^T P(x_t|x_{<t};\theta), \qquad h_t = \text{Transformer}(x_{<t}), \qquad P(x_t|x_{<t}) = \text{Softmax}(Wh_t+b)$$

The Transformer allows each token to attend to all previous tokens in parallel during training (teacher forcing), making LLMs far more scalable than earlier RNNs. In chemistry: protein language models (ESM) predict structure from sequence; other LLMs generate crystal structures and stable inorganic materials.

## 9.7 The Transformer Architecture

### From Token to Prediction: Dimensions Throughout

A sentence is first split into **tokens** (subword pieces). Each token is looked up in an **embedding matrix** $W_E \in \mathbb{R}^{|\text{vocab}|\times d_\text{model}}$, producing a vector $e_t \in \mathbb{R}^{d_\text{model}}$. A **positional encoding** $p_t \in \mathbb{R}^{d_\text{model}}$ is added, giving the input matrix:

$$X = [e_1+p_1,\;\ldots,\;e_n+p_n] \in \mathbb{R}^{n\times d_\text{model}}$$

This $d_\text{model}$-dimensional vector per token is the central object that flows through the entire network. Every transformer block takes $X\in\mathbb{R}^{n\times d_\text{model}}$ and outputs a refined $X'\in\mathbb{R}^{n\times d_\text{model}}$ of the same shape — the embedding dimension never changes between layers. Only at the very end is the final vector for the last token projected back to vocabulary size via an **unembedding matrix** $W_U\in\mathbb{R}^{d_\text{model}\times|\text{vocab}|}$, then softmaxed to give $P(x_{t+1}|x_{\leq t})$.

Each transformer block consists of two sub-layers: (1) **multi-head self-attention**, then (2) a **feed-forward MLP**, each followed by a residual connection and layer normalization.

### Why Attention? The Core Problem

After the embedding step, each token's vector encodes only what that token is — not its meaning in context. The word *mole* has the same initial vector regardless of whether the surrounding text is about chemistry, biology, or an animal. The attention mechanism is what allows each token's embedding to absorb information from all other tokens and update its meaning in context.

### Queries, Keys, and Values (Single Head)

Each token $i$ simultaneously broadcasts three things, computed by multiplying the embedding by three learned weight matrices:

$$q_i = e_i W^Q \in \mathbb{R}^{d_k}, \quad k_i = e_i W^K \in \mathbb{R}^{d_k}, \quad v_i = e_i W^V \in \mathbb{R}^{d_v}$$

Intuitively (following 3Blue1Brown): the **query** encodes what token $i$ is *looking for* ("are there any adjectives before me?"), the **key** encodes what token $j$ *offers* ("I am an adjective"), and the **value** encodes *what token $j$ will send* if found relevant ("here is the information to add to your embedding").

Alignment between token $i$'s query and token $j$'s key is measured by a dot product and normalized:

$$a_{ij} = \text{softmax}_j\!\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right)$$

The $\sqrt{d_k}$ scaling prevents dot products from becoming too large (which would saturate the softmax). Collecting all pairs into matrix form gives the **attention pattern** $A = \text{softmax}(QK^T/\sqrt{d_k}) \in \mathbb{R}^{n\times n}$, where $A_{ij}$ is how much token $i$ attends to token $j$.

The value vectors $v_j$ encode "what token $j$ will contribute if attended to". The update to token $i$'s embedding is a weighted average of all value vectors, weighted by how much token $i$ attends to each token $j$:

$$\Delta e_i = \sum_j a_{ij}\,v_j$$

Concretely: if token $i$ is *creature* and *fluffy*, *blue* got high attention weights, then $\Delta e_i \approx v_\text{fluffy} + v_\text{blue}$ — the embedding gets updated toward "fluffy blue creature". Stacking all $\Delta e_i$ as rows and writing $A = \text{softmax}(QK^T/\sqrt{d_k})$:

$$\Delta E = AV = \underbrace{\text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)}_{A\,\in\,\mathbb{R}^{n\times n}} V \in \mathbb{R}^{n\times d_v}$$

This is added back via a residual connection: $E' = E + \Delta E$.

### Multi-Head Attention

A single head can only learn one type of contextual relationship. Running $h$ heads in parallel, each with its own $W^Q_i, W^K_i, W^V_i$, lets the model simultaneously capture many different relationships (e.g., adjective–noun, subject–verb, coreference). Each head operates in a lower-dimensional subspace $d_k = d_v = d_\text{model}/h$:

$$\text{head}_i = \text{Attention}(XW^Q_i,\; XW^K_i,\; XW^V_i) \in \mathbb{R}^{n\times d_v}$$

The outputs are concatenated and projected back to $d_\text{model}$:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)\,W^O, \quad W^O\in\mathbb{R}^{hd_v\times d_\text{model}}$$

Two variants:

- **Self-attention** — $K = Q = V = XW$, all from the same sequence $X \in \mathbb{R}^{n \times d_\text{model}}$.
- **Cross-attention** (encoder-decoder, e.g. translation) — given two sequences $X \in \mathbb{R}^{n \times d_\text{model}}$ and $Y \in \mathbb{R}^{m \times d_\text{model}}$, both $K$ and $Q$ come from $X$ while $V$ comes from $Y$: $K = Q = XW^X \in \mathbb{R}^{n \times d_k}$, $V = YW^Y \in \mathbb{R}^{m \times d_v}$.

### Positional Encoding

Self-attention is **permutation-equivariant**: $\text{Attention}(PX) = P\,\text{Attention}(X)$ for any permutation $P$. This means without positional information, "the dog bites the man" and "the man bites the dog" look identical. Positional encodings $p_t$ added to the embeddings break this symmetry: after permuting tokens, the position information no longer matches the token, so the model can distinguish order.

### Causal Masking and Teacher Forcing

For autoregressive generation, a token must not attend to future tokens (that would reveal the answer). Before softmax, all entries where $j > i$ in $QK^T$ are set to $-\infty$, which maps to 0 after softmax — the **causal mask**.

**Teacher forcing**: during training, all ground-truth tokens are fed simultaneously as input (rather than feeding the model's own predictions). This allows fully parallel training, since each position's loss $\log P(x_i|x_{<i};\theta)$ can be computed in one forward pass. The causal mask enforces that position $i$ can only see positions $< i$.

### Block Structure

**Encoder block** (N×): Multi-head attention → Add & Norm → FFN → Add & Norm

**Decoder block** (N×): Masked multi-head attention → Add & Norm → Cross-attention → Add & Norm → FFN → Add & Norm

LLMs (GPT-style) use decoder-only blocks with causal masking throughout.

## 9.8 The Problem of Sampling Physical Configurations

Standard simulation methods for sampling a target distribution $\mu_X(x) \propto e^{-\beta U(x)}$ — Monte Carlo (random moves accepted by Metropolis criterion) and Molecular Dynamics (Langevin integration) — both require many force evaluations per independent sample. Even with MLIPs reducing the cost per evaluation, these methods remain expensive for high-dimensional systems due to the large number of steps needed to generate decorrelated configurations. This motivates **latent generative models** that directly map simple noise to samples, producing decorrelated configurations in one forward pass.

## 9.9 Normalizing Flows

A normalizing flow is an **invertible** neural network $F(\cdot;\theta): \mathbb{R}^n \to \mathbb{R}^n$ that maps samples from a simple **prior** $\mu_Z$ (typically a standard Gaussian) to samples that approximate a complex **target** distribution — in physics, the Boltzmann distribution $\mu_X(x) \propto e^{-\beta U(x)}$. Invertibility plus the change-of-variables formula gives exact access to the induced likelihood in both directions:

$$p_X(F(z;\theta)) = \mu_Z(z)\,|\det J(z;\theta)|^{-1}$$

$$\mu_Z(F^{-1}(x;\theta)) = p_X(x)\,|\det J^{-1}(x;\theta)|^{-1}$$

where $J(z;\theta) = \partial F/\partial z$ is the Jacobian and $p_X$ is the distribution the flow actually generates (an approximation of $\mu_X$). This exact likelihood enables both **training** and **reweighting of observables** — two capabilities that approximate generative models (GANs, VAEs) do not provide.

### Split-Coupling Architecture (Tractable Jacobian)

A naïve neural network has an $n\times n$ Jacobian with $\mathcal{O}(n^3)$ determinant cost. Split-coupling flows partition the input into two halves $z = (z^I, z^{II})$ and apply:

$$F(z;\theta) = \begin{cases} x^I = z^I \\ x^{II} = z^{II} \odot \exp(S(z^I;\theta)) + T(z^I;\theta) \end{cases}$$

where $S$ and $T$ are arbitrary (non-invertible) neural networks acting only on $z^I$. The Jacobian is block-lower-triangular, so its determinant is just the product of diagonal entries:

$$\log|\det J(z;\theta)| = \sum_i S_i(z^I;\theta)$$

The inverse is equally cheap:

$$F^{-1}(x;\theta) = \begin{cases} z^I = x^I \\ z^{II} = (x^{II} - T(x^I;\theta)) \odot \exp(-S(x^I;\theta)) \end{cases}$$

Both forward and inverse passes are $\mathcal{O}(n)$ in the Jacobian computation. By stacking multiple coupling layers and alternating which half is transformed, the full $n$-dimensional space is mixed.

### Training: By Energy vs. By Example

Because the flow is invertible, training can proceed in **two directions**:

**Training by example** (requires samples $x_B \sim \mu_B$) — run the flow backwards via the NLL loss:

$$\mathcal{L}_\text{NLL} = -\mathbb{E}_{x_B \sim \mu_B}\!\left[\log \mu_A(F^{-1}(x_B;\theta)) + \log|\det J^{-1}(x_B;\theta)|\right]$$

Minimises $\text{KL}(\mu_B \| p_B(\cdot;\theta))$. Requires target samples but not the analytical form of $\mu_B$.

**Training by energy** (requires only $U_B$, no target samples) — run the flow forwards via the KLD loss:

$$\mathcal{L}_\text{KLD} = \mathbb{E}_{x_A \sim \mu_A}\!\left[\beta_B U_B(F(x_A;\theta)) - \log|\det J(x_A;\theta)|\right]$$

Minimises $\text{KL}(p_A(\cdot;\theta) \| \mu_A)$. Only the energy function $U_B$ is needed — the natural choice when target samples are unavailable. Both losses can be **combined simultaneously**:

$$\mathcal{L} = \lambda_\text{NLL}\,\mathcal{L}_\text{NLL} + \lambda_\text{KLD}\,\mathcal{L}_\text{KLD}$$

with tunable weights $\lambda$, allowing bidirectional training when samples from both systems are available.

### Reweighting Observables and rESS

Once trained, the flow maps samples $x_A \sim \mu_A$ to an approximate distribution $p_B$ that may differ from the true target $\mu_B$. Observables are corrected by **importance reweighting**:

$$\langle O \rangle_B = \frac{\sum_i \omega(x_B^i)\,O(x_B^i)}{\sum_i \omega(x_B^i)}, \qquad \omega(x_B) = \frac{\mu_B(x_B)}{p_B(x_B)}$$

For generated samples $x_B = F(x_A)$, substituting the change-of-variables expression for $p_B$ gives the fully computable weight:

$$\omega_B(F(x_A)) \propto \exp\!\left[-\beta_B U_B(F(x_A)) + \beta_A U_A(x_A) + \log|\det J(x_A;\theta)|\right]$$

For the Gaussian-prior special case ($\beta_A U_A = \tfrac{1}{2}\|x_A\|^2$) this reduces to $e^{-\beta U(F(z)) + \frac{1}{2}\|z\|^2 + \log|\det J|}$. All three terms are available analytically.

The quality of training (and the reliability of reweighting) is measured by the **relative effective sample size (rESS)**:

$$\text{rESS} = \frac{1}{N}\frac{(\sum_i w_i)^2}{\sum_i w_i^2} \in (0,1]$$

$\text{rESS} = 1$ when all weights are equal (perfect match); $\text{rESS} \to 0$ when one sample dominates (poor coverage).

### Choice of Prior

Mapping a standard Gaussian to a complex physical distribution is too difficult in high dimensions. The key design choice is to pick a prior $\mu_Z$ that is **close to the target** $\mu_X$:

**For solids**: Use a mixture of Gaussians centred on the target lattice sites. The flow only needs to learn small local displacements. Enables training by energy with no target samples.

**For liquids**: Use a different model potential (e.g., WCA instead of Lennard-Jones). The flow learns the perturbation between models. Enables analysis of the full phase diagram using simulations at one state point, and permits use of cheap models for independent-configuration generation.

In general, the prior need not be a Gaussian at all — it can be any tractable distribution, including another Boltzmann distribution at a different temperature or with a different potential. This Boltzmann-to-Boltzmann framing (Coretti et al., J. Chem. Phys. 162, 184102 (2025)) unifies many applications: the training losses and reweighting weights carry through unchanged with $\mu_A(x_A) \propto e^{-\beta_A U_A(x_A)}$ replacing the Gaussian prior.

### Advanced Coupling Architectures

The affine split-coupling is the simplest invertible coupling. The general framework allows any invertible-in-$z^{II}$ function $f$:

$$F(z;\theta) = \begin{cases} x^I = z^I \\ x^{II} = f\!\left[z^{II},\, h(z^I;\theta)\right] \end{cases} \qquad F^{-1}(x;\theta) = \begin{cases} z^I = x^I \\ z^{II} = f^{-1}\!\left[x^{II},\, h(x^I;\theta)\right] \end{cases}$$

The log-Jacobian is still $\mathcal{O}(n)$ because the block-triangular structure is preserved regardless of the choice of $f$:

$$\log|\det J(z;\theta)| = \sum_\alpha \log\frac{\partial f[z^{II}_\alpha, h(z^I;\theta)]}{\partial z^{II}_\alpha} = \sum_\alpha S_\alpha(z^I;\theta)$$

$$\log|\det J^{-1}(x;\theta)| = -\sum_\alpha S_\alpha(x^I;\theta)$$

The affine coupling $f[z^{II}, h] = z^{II} \odot e^S + T$ is the special case where $h$ outputs $(S, T)$ and each $S_\alpha = S_\alpha(z^I;\theta)$.

**Neural Spline Flows (NSF)**: Replace the affine $f$ with a **rational-quadratic (RQ) spline**. A neural network $h(z^I;\theta)$ outputs the knot positions and derivatives of the spline. The RQ spline is strictly monotonic (hence invertible in $z^{II}$), has a closed-form inverse and analytic derivative (so $S_\alpha$ is cheap to compute), and can represent highly non-linear, multi-modal transforms — making it far more expressive than the affine coupling for the same architectural cost. This is the architecture behind the BoltzmannGenerator family of models.

### Continuous Normalizing Flows (CNFs)

Stacking discrete coupling layers can be extended to the **continuous limit**: instead of $K$ discrete layers $z_k = F_k(z_{k-1};\theta_k)$, define a vector field $g(z_t, t;\phi)$ and integrate an ODE:

$$\frac{dz_t}{dt} = g(z_t, t;\phi)$$

$$x = F(z_0;\phi) = z_T = z_0 + \int_{t_0}^T g(z_t, t;\phi)\,dt$$

The inverse is obtained by integrating backwards over the same ODE (the integral changes sign):

$$z_0 = F^{-1}(x;\phi) = z_T - \int_{t_0}^T g(z_t, t;\phi)\,dt$$

The log-density evolves via the **instantaneous change of variables**:

$$\frac{d\log p(z_t)}{dt} = -\text{Tr}\{J(z_t;\phi)\}$$

Sample and log-probability can be integrated simultaneously in one ODE solve:

$$\begin{bmatrix} x \\ \log p_X(x) \end{bmatrix} = \begin{bmatrix} z \\ \log \mu_Z(z) \end{bmatrix} + \int_{t_0}^T dt \begin{bmatrix} g(t, z_t;\phi) \\ -\text{Tr}\{J(z_t;\phi)\} \end{bmatrix}$$

Computing $\text{Tr}\{J\}$ exactly costs $\mathcal{O}(n^2)$ (one backward pass per dimension). In practice it is estimated cheaply via the **Hutchinson trace estimator**: $\text{Tr}(A) \approx \mathbb{E}_v[v^\top A v]$ with $v \sim \mathcal{N}(0,I)$, which requires only a single Jacobian-vector product.

---

## 9.10 Diffusion Models

The core idea of diffusion models is simple: **gradually destroy structure by adding noise, then learn to reverse this process**. Training is stable and scalable (no adversarial game, no encoder approximations), which is why diffusion models have become the dominant generative framework for images, molecules, and physical configurations.

There are two main formulations — discrete-time (DDPMs) and continuous-time (score-based SDEs) — which are deeply connected.

### Denoising Diffusion Probabilistic Models (DDPMs)

A DDPM is a **latent variable model** with $T$ latent variables $x_1, \ldots, x_T$, where each latent is a progressively noisier version of the data $x_0$. The model has two parts: a fixed **encoder** (the forward noising process) and a learned **decoder** (the reverse denoising process). At generation time you start from pure noise $x_T \sim \mathcal{N}(0,I)$ and iteratively denoise to produce a sample $x_0$.

### Forward Process

Starting from a data sample $x_0 \sim \mu_X$, the forward chain adds Gaussian noise at each step with a variance schedule $\beta_t \in (0,1)$:

$$q(x_t|x_{t-1}) = \mathcal{N}\!\left(x_t;\,\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t I\right)$$

Crucially, the marginal at any timestep $t$ can be computed in closed form by defining $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$:

$$q(x_t|x_0) = \mathcal{N}\!\left(x_t;\,\sqrt{\bar{\alpha}_t}\,x_0,\,(1-\bar{\alpha}_t)I\right)$$

which allows direct sampling via the reparametrization trick:

$$x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon, \qquad \varepsilon \sim \mathcal{N}(0,I)$$

For large $T$ with a suitable schedule, $q(x_T) \approx \mathcal{N}(0,I)$ — the prior is effectively standard Gaussian.

### Reverse Process

The reverse chain is parameterized by learned networks $\mu_\theta$ and $\Sigma_\theta$:

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}\!\left(x_{t-1};\,\mu_\theta(x_t, t),\,\Sigma_\theta(x_t, t)\right)$$

Training minimizes the KL divergence between the full joint distributions $q(x_0,\ldots,x_T)$ and $p_\theta(x_0,\ldots,x_T)$. Applying the same ELBO decomposition as in VAEs, the loss factorizes into per-timestep terms:

$$\mathcal{L} = L_0 + \sum_{t=2}^T L_{t-1} + L_T$$

where each $L_{t-1} = \mathbb{E}_q\!\left[\text{KL}\!\left(q(x_{t-1}|x_t,x_0)\,\big\|\,p_\theta(x_{t-1}|x_t)\right)\right]$ is a KL between two Gaussians (computable analytically), $L_0 = -\mathbb{E}_q\log p_\theta(x_0|x_1)$ is the reconstruction term, and $L_T$ penalizes the prior mismatch.

With fixed diagonal variance $\Sigma_\theta = \sigma_t^2 I$, the loss simplifies to a **noise prediction objective**:

$$\mathcal{L}_\text{simple} = \mathbb{E}_{x_0,\varepsilon,t}\!\left[\|\varepsilon_\theta(x_t, t) - \varepsilon\|^2\right]$$

where the network $\varepsilon_\theta$ is trained to predict the noise $\varepsilon$ added to $x_0$ to obtain $x_t$ (Ho et al., NeurIPS 2020).

### Sampling

Starting from $x_T \sim \mathcal{N}(0,I)$, iterate from $t = T$ down to $1$:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\,\varepsilon_\theta(x_t, t)\right) + \sigma_t\, z, \qquad z \sim \mathcal{N}(0,I)\ (z=0\ \text{at}\ t=1)$$

where $\alpha_t = 1-\beta_t$. The network $\varepsilon_\theta$ predicts the noise, which is subtracted to get a cleaner estimate of $x_0$, and $\sigma_t z$ adds a small controlled amount of noise back to avoid collapse. This is exactly the reverse Gaussian kernel $p_\theta(x_{t-1}|x_t)$ applied in sequence.

**Connection to VAEs**: A DDPM is essentially a VAE with $T$ latent layers and a non-learnable (fixed) encoder. The latent variables are $\{x_1,\ldots,x_T\}$, the encoder is the fixed forward diffusion process $q$, and only the decoder (reverse chain) is learned.

### Score-Based Diffusion Models

The reason denoising works can be understood via the **score function** $s(x) = \nabla_x \log p(x)$. For the Boltzmann target $\mu_X(x) \propto e^{-\beta U(x)}$, the score is proportional to the force:

$$s(x) = \nabla_x \log \mu_X(x) = -\beta\,\nabla_x U(x) \propto F(x)$$

This is exactly the drift term in Langevin dynamics: following the score moves toward regions of higher probability density. Denoising a noisy sample is thus equivalent to running a few steps of Langevin dynamics.

To sample from a target distribution, one can inject noise via a forward SDE:

$$dx = f(x,t)\,dt + g(t)\,dW_t$$

and reverse it by solving the corresponding reverse-time SDE:

$$dx = \left[f(x,t) - g(t)^2\,\nabla_x \log p_t(x)\right]dt + g(t)\,d\bar{W}_t$$

Since the true score $\nabla_x \log p_t(x)$ is intractable, it is approximated by a learned network $s_\theta(x,t)$ trained via **score matching**. The loss after integration by parts (to eliminate the need for $p(x)$ explicitly) becomes:

$$\mathcal{L}(\theta) = \mathbb{E}_{p(x)}\!\left[\tfrac{1}{2}\|s_\theta(x)\|^2 + \nabla_x \cdot s_\theta(x)\right]$$

**Probability flow ODE.** For every forward SDE there is a deterministic **probability flow ODE** that shares the same marginal distributions $p_t(x)$ but involves no stochastic term:

$$dx = \left[f(x,t) - \tfrac{1}{2}\,g(t)^2\,s_t(x)\right]dt$$

This defines a velocity field $v(x,t) = f(x,t) - \frac{1}{2}g(t)^2 s_t(x)$. Integrating it numerically from $x_T \sim \mathcal{N}(0,I)$ back to $t=0$ yields samples — exactly as in a CNF, but derived from the diffusion model.

## 9.11 Flow Matching

CNFs (introduced above under Normalizing Flows) are expensive to train because computing the likelihood requires simulating the ODE. **Flow Matching** provides a direct training objective that avoids this. Given a pair $(z, x)$ from the prior and data respectively, define the linear interpolant:

$$x_t = (1-t)z + tx, \qquad \frac{dx_t}{dt} = x - z$$

A network $v_\theta(x_t, t)$ is trained to match this per-pair velocity:

$$\mathcal{L}_\text{FM} = \mathbb{E}\!\left[\|v_\theta(x_t, t) - (x-z)\|^2\right], \qquad z \sim \mu_Z,\; x \sim \mu_X,\; t \sim \mathcal{U}(0,1)$$

At test time, integrating $dz/dt = v_\theta(z_t, t)$ numerically from $z_0 \sim \mu_Z$ produces samples from (approximately) $\mu_X$. Flow matching bypasses the need to simulate the ODE during training, making it significantly cheaper than maximum-likelihood training of CNFs.

---

