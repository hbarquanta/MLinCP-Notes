# Chapter 10 — ML Methods for Rare Events Sampling

## 10.1 Rare Events: Motivation

Rare events are transitions between long-lived **metastable states** separated by a free energy barrier $\Delta F$. Their rates are exponentially suppressed by the barrier height (Arrhenius/Kramers law):

$$k = k_0 \exp\!\left(-\beta \Delta F\right)$$

where $\beta = 1/k_BT$. Examples include nucleation, chemical reactions, and biomolecular conformational changes. For simulations this is a fundamental problem: the system spends nearly all its time in one metastable state waiting for the event, so direct MD wastes almost all computational effort.

---

## 10.2 Bias Methods

The central idea is to add a bias to the potential energy to force the system into the transition region:

$$U_B(x) = U(x) + U_\text{bias}[\zeta(x)]$$

where $\zeta(x)$ is a **collective variable (CV)** — a function of the Cartesian coordinates that describes the reaction progress (e.g. an interatomic distance or bond angle). Simulating under $U_B$ gives a biased distribution $\rho_B$; the original unbiased distribution is recovered by **reweighting**. Reweighting works because the bias is known analytically, so every generated configuration can be assigned its correct statistical weight — this is the principle of importance sampling.

The **potential of mean force (PMF)** $F(\zeta)$ is the effective free energy as a function of the CV, obtained by integrating out all other degrees of freedom at fixed $\zeta$:

$$F(\zeta) = -k_BT\log\!\int dx\,e^{-\beta U(x)}\,\delta(\zeta(x) - \zeta) + \text{const}$$

It acts as an effective potential governing the CV dynamics. The unbiased PMF follows from the biased distribution via:

$$F(\hat\zeta) = -k_BT\log\rho(\hat\zeta) = -k_BT\log\!\left[\frac{Z_B}{Z}\right] - U_\text{bias}(\hat\zeta) - k_BT\log\rho_B(\hat\zeta)$$

A crucial limitation of all bias methods: they distort the dynamics and therefore do not preserve kinetic information. They give thermodynamics (free energies, equilibrium populations) but not rates or transition mechanisms.

**Umbrella sampling** runs multiple independent simulations, each with a harmonic bias centred at a different target $\bar\zeta^{(i)}$:

$$U_B^{(i)}(x) = U(x) + \frac{k}{2}\!\left(\zeta(x) - \bar\zeta^{(i)}\right)^2$$

The windows are chosen to overlap in $\zeta$-space; the PMF fragments from each window are stitched together via **WHAM** (weighted histogram analysis method), which finds the self-consistent set of free energies that best reconciles the overlapping histograms. Advantages: simple, yields the full PMF. Drawback: requires a good CV, dynamics is distorted, windows must be chosen in advance.

**MetaDynamics** (Laio & Parrinello, 2002) builds a time-dependent adaptive bias by depositing Gaussians at the currently visited $\zeta$ value:

$$U_\text{bias}(\zeta(x), t) = W\sum_{k=0}^{N(t)} \exp\!\left(-\frac{[\zeta(x)-\bar\zeta_k]^2}{2\sigma_\zeta^2}\right)$$

The system is discouraged from revisiting states it has already explored ("filling the free energy wells with sand"). In the original (conventional) formulation all Gaussians have fixed height $W$, and at long times the accumulated bias converges to the negative PMF:

$$U_\text{bias}(\zeta(x),\, t\to\infty) = -F(\zeta) + C$$

In the modern **well-tempered** variant, Gaussian heights are scaled down as the simulation progresses by a factor that depends on how much bias has already been deposited at that location. The bias then converges to only a fraction of the negative PMF:

$$U_\text{bias}(\zeta(x),\, t\to\infty) = -\!\left(1 - \frac{1}{\gamma}\right)F(\zeta) + C$$

where $\gamma > 1$ is a user-chosen bias factor. This prevents the system from being pushed completely out of equilibrium and improves convergence. No predefined windows are needed. Drawback: still requires a CV, and the dynamics during the simulation is distorted.

---

## 10.3 Transition Path Sampling (TPS)

Bias methods distort the dynamics and require a pre-chosen CV. **Transition path sampling** avoids both by performing MCMC directly in the space of reactive trajectories.

The **transition path ensemble** is the set of all trajectories connecting stable states $A$ and $B$:

$$P(x|\text{TP}) \propto P(\text{TP}|x)\,\rho(x)$$

TPS generates new paths from old ones via the **shooting algorithm**:
1. Select a shooting point $x_k$ from the current path.
2. Integrate a new trajectory forward and backward from $x_k$.
3. Accept the new path if it is reactive ($A\to B$ or $B\to A$); otherwise reject (detailed balance criterion).

Advantages: dynamics is unbiased, no reaction coordinate required, and — crucially — kinetic information (transition rates, mechanism) is preserved because the real dynamics is used throughout. Drawback: each move requires a full trajectory integration, and like any MCMC method it suffers from correlations between successive paths.

**Shooting from the Top** is a TPS variant that restricts shooting points to the transition state region (near the top of the barrier) rather than the full path. This eliminates long decorrelation times between accepted moves. The drawback is that it reintroduces a CV to define "the top", making it somewhat a hybrid between TPS and bias methods.

**Normalizing flows for TPS** (Falkner et al. 2023) address a fundamental bottleneck of standard TPS: the shooting algorithm is inherently sequential (each new path depends on the previous one), producing correlated paths and resisting parallelization. The key idea is to decouple shooting point generation from the trajectory integration step.

**Conditioned Boltzmann generators.** A normalizing flow is conditioned on a bias center $\bar{r}$ (a value of a collective variable $\zeta(x)$), producing the transformation $x = F(z|\bar{r};\theta)$ with the conditioned change-of-variables formula

$$q_x(x|\bar{r}) = p_z\!\left(F^{-1}(x|\bar{r};\theta)\right)|\det J_{F^{-1}}(x|\bar{r};\theta)|$$

Sampling is done by drawing $z \sim \mathcal{N}(0,I)$ and mapping $x = F(z|\bar{r};\theta)$ for a chosen $\bar{r}$. By varying $\bar{r}$, a single trained network generates configurations in any targeted region of phase space — analogous to umbrella windows, but without running separate simulations for each window.

**Parallel path sampling.** A large batch of shooting points is generated before any trajectory integration begins. From each shooting point, forward and backward trajectory segments are integrated independently until a stable state is reached. This makes path generation embarrassingly parallel — there are no dependencies between trajectories.

**Path reweighting.** Trajectories launched from the shooting point distribution $p_\text{SP}(x)$ do not form a properly weighted transition path ensemble, because paths initiated in high-probability regions are overrepresented. The correction factor for a trajectory $X(\tau) = \{x_0, x_{\Delta t},\ldots, x_\tau\}$ is

$$\Omega[X(\tau)] = \left[\sum_{k=0}^{\tau/\Delta t} \frac{p_\text{SP}(x_{k\Delta t})}{p_\text{eq}(x_{k\Delta t})}\right]^{-1}$$

where $p_\text{eq}(x) \propto e^{-\beta U(x)}$. Observable averages over the transition path ensemble are then computed as the weighted average $\langle A \rangle = \sum_i \Omega[X_i] A[X_i] / \sum_i \Omega[X_i]$.

**Training.** The conditioned flow is trained using a combined forward/reverse KL loss. Training by example (forward KL) uses configurations from biased simulations at discrete bias centers $\bar{r}_j$:

$$\mathcal{L}_\text{fwd} = \mathbb{E}_{\bar{r}}\,\mathbb{E}_{x}\!\left[\frac{1}{\sigma^2}\|F^{-1}(x|\bar{r};\theta)\|^2 - \log|\det J_{F^{-1}}(x|\bar{r};\theta)|\right]$$

Training by energy (reverse KL) samples from the latent Gaussian and evaluates the biased potential:

$$\mathcal{L}_\text{rev} = \mathbb{E}_{\bar{r}}\,\mathbb{E}_{z}\!\left[\beta U(F(z|\bar{r};\theta)) + \frac{\beta k}{2}[\zeta(F(z|\bar{r};\theta)) - \bar{r}]^2 - \log|\det J(z|\bar{r};\theta)|\right]$$

The final loss is $\mathcal{L} = \lambda_\text{fwd}\mathcal{L}_\text{fwd} + \lambda_\text{rev}\mathcal{L}_\text{rev}$.

**Bonus outputs.** Because the network generates configurations at many bias centers, the PMF $F(\zeta)$ can be reconstructed directly via WHAM on the generated samples. The transition path probability $p(\text{TP}|\zeta(x))$ is also accessible from the path ensemble, providing a data-driven estimate of the transition state region without additional simulations.

---

## 10.4 The Committor and the Backward Kolmogorov Equation

The **committor** $p_B(x)$ is the probability that a trajectory initiated at $x$ with random momenta reaches $B$ before $A$. It is the theoretically optimal reaction coordinate: it is defined by the physics of the system rather than an arbitrary choice of CV. Transition states are precisely the configurations where $p_B(x) = 1/2$, and the committor is related to the path ensemble by:

$$P(\text{TP}|x) = 2p_B(x)(1 - p_B(x))$$

For overdamped Langevin dynamics $dX_t = -D\nabla U\,dt + \sqrt{2D}\,dW_t$, the committor satisfies the **backward Kolmogorov equation**:

$$Lp_B = -D\nabla U\cdot\nabla p_B + D\Delta p_B = 0$$

with boundary conditions $p_B = 0$ on $A$ and $p_B = 1$ on $B$. In one dimension this has the closed-form solution:

$$p_B(x) = \frac{\int_a^x dx'\,e^{\beta U(x')}}{\int_a^b dx'\,e^{\beta U(x')}}$$

In higher dimensions no closed form exists, motivating a machine learning approach.

---

## 10.5 ML for the Committor and AIMMD

**Variational approach.** The committor can be learned by representing it as a neural network and minimizing the variational loss (from the stationarity of the Kolmogorov equation):

$$\mathcal{I}[p_B] = Z^{-1}\int e^{-\beta U(x)}|\nabla p_B(x)|^2\,dx$$

The problem is that configurations near the barrier (where $p_B \approx 1/2$) are never spontaneously visited. The solution is to bias the system toward the barrier using:

$$U_\text{bias}(x) = -k_BT\log|\nabla p_B(x)|^2$$

and run a self-consistent cycle: sample with biased potential → update dataset → retrain committor → update bias.

**AIMMD** (AI for Molecular Mechanism Discovery) integrates committor learning with TPS into a self-consistent loop:

1. Run $N$ TPS steps: select shooting point, shoot forward/backward, record outcomes $(n_A, n_B)$.
2. Train committor model $p_B(x) \approx \sigma(q(x;\theta))$ by maximizing the binomial log-likelihood:

$$\mathcal{L}(\theta) = \sum_i\left[n_B^{(i)}\log\sigma(q_i) + n_A^{(i)}\log(1-\sigma(q_i))\right]$$

3. Use the learned committor to bias shooting point selection toward $p_B \approx 1/2$.

This closes the loop: better shooting points give more informative trajectories, which train a better committor, which selects even better shooting points.

**Symbolic regression** can then be applied to extract an interpretable reaction coordinate from the trained committor model. The pipeline has two steps:

1. **Attribution analysis:** evaluate the training loss on the full dataset, then perturb one input coordinate at a time across the entire batch and recompute the loss. The increase in loss measures how much that coordinate contributes to the committor — high sensitivity means high importance. This ranks all input degrees of freedom by their relevance to the reaction.

2. **Symbolic regression:** take the subset of most important inputs and run an evolutionary algorithm to distill a simple analytical expression from the trained network. The result is a compact, human-interpretable reaction coordinate, e.g.:

$$q_\text{sr}(x_1, x_3) = -8.38\,x_1 + 5.62\exp[-0.285\,x_3]$$

This two-step approach answers both *which* degrees of freedom matter and *how* they combine into a reaction coordinate.

---

