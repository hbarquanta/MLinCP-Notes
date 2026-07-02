# Chapter 8 — Machine Learning Interatomic Potentials II

## 8.1 From Test Error to Production-Quality Potentials

Training an MLIP and achieving a low test RMSE on a held-out dataset is only the first step. A potential that performs well on static benchmark configurations may still fail catastrophically when deployed in a molecular dynamics simulation, and a potential that keeps MD stable may still produce incorrect thermodynamic averages or kinetics. This gap between paper benchmarks and real-world utility motivates L8, which is concerned with the practical challenges of deploying MLIPs and the strategies developed to address them. Five open problems structure the discussion:

**MD stability** — small errors can accumulate over millions of time steps and drive the system into unphysical configurations. Low test RMSE does not guarantee stability.

**Speed vs. accuracy** — more expressive architectures (equivariant, high body-order) are slower to evaluate, while faster models sacrifice physical fidelity. Practical deployment always involves this tradeoff.

**Data efficiency** — every training point requires an expensive DFT calculation. Generating enough data to cover the relevant portion of chemical space is a major bottleneck.

**Long-range interactions** — GNNs with local cutoffs cannot capture electrostatics ($1/r$) or London dispersion ($1/r^6$), which are crucial in polar systems, ionic materials, and van der Waals complexes.

**Quantum observables beyond energy and forces** — many applications need electronic spectra, NMR shifts, or dielectric functions, none of which are provided by the standard MLIP energy-and-forces framework.

### Three Levels of MLIP Accuracy

It is useful to distinguish three qualitatively different quality levels when evaluating an MLIP:

**Level 1 — Low RMSE.** The model achieves a small mean-squared error on a static test set. This is the entry requirement, but it says nothing about dynamical behaviour. A potential can have excellent RMSE yet still produce trajectories that diverge within picoseconds because the test configurations do not represent the rare but critical regions of phase space visited during dynamics.

**Level 2 — MD stable.** The model keeps the simulation stable over arbitrarily long times ($\tau_\text{stab} \to \infty$): no atom collisions, no non-physical geometries, no integrator blow-ups. Stability is a necessary but not sufficient condition for physical correctness.

**Level 3 — MD accurate.** The model reproduces correct thermodynamic averages $\langle A \rangle$ and dynamical observables such as time-correlation functions $\langle A(0) A(t) \rangle$. This is the ultimate standard for production simulations and is often the hardest to achieve because it requires the potential energy surface to be correct not just at minima but across the full accessible portion of configurational space.

## 8.2 Dynamic Observables and Time-Correlation Functions

Many physically relevant quantities are not static energies but time-averaged properties derived from long trajectories. The general framework is the **time-correlation function** (TCF) between two dynamical variables $a$ and $b$:

$$C_{ab}(t) = \langle a(t_0)\, b(t_0 + t) \rangle_{t_0} = \frac{1}{Q_\text{cl}} \int d\mathbf{r}_0\, d\mathbf{p}_0\; e^{-\beta H}\, a(t_0)\cdot b(t_0 + t).$$

The ensemble average over initial conditions $t_0$ is equivalent to a Boltzmann-weighted phase-space integral. For the special case $a = b$, this becomes the autocorrelation function, satisfying $C_{aa}(0) = \langle a(t_0)^2 \rangle$.

Equations that relate TCFs to macroscopic transport properties are called **Green-Kubo relations**. The central example is the **diffusion coefficient**, defined as the time-integral of the velocity autocorrelation function (VACF):

$$D = \frac{1}{3} \int_0^\infty dt\; C_{vv}(t) = \frac{1}{3} \int_0^\infty dt\; \langle \mathbf{v}(t) \cdot \mathbf{v}(0) \rangle.$$

Physically, the VACF measures how long a particle remembers its direction of motion. It typically decays fast (within tenths of a ps in a liquid) but can show a rebound — a brief anti-correlation — as the particle bounces off its cage of neighbours before diffusing away. The diffusion coefficient is the area under this curve.

The same Green-Kubo framework applies to a range of other observables:

**Thermal transport** — the thermal conductivity $\kappa$ is the integral of the heat-flux autocorrelation function, where $\mathbf{J}_Q$ is the instantaneous energy-current density through the simulation cell:

$$\kappa \propto \int_0^\infty dt\; \langle \mathbf{J}_Q(0) \cdot \mathbf{J}_Q(t) \rangle.$$

**IR spectrum** — the infrared absorption spectrum is the Fourier transform of the dipole-moment autocorrelation function:

$$I_\text{IR}(\omega) \propto \int_{-\infty}^\infty dt\; e^{i\omega t}\, \langle \boldsymbol{\mu}(0) \cdot \boldsymbol{\mu}(t) \rangle.$$

A vibration absorbs IR radiation when it modulates the dipole moment; the TCF approach captures anharmonicity and finite-temperature broadening automatically, which static normal-mode analysis cannot.

**SFG spectrum** — sum-frequency generation (SFG) is a surface-sensitive nonlinear spectroscopy whose spectrum comes from the cross-correlation of dipole moment and polarizability $\boldsymbol{\alpha}$:

$$I_\text{SFG}(\omega) \propto \int_{-\infty}^\infty dt\; e^{i\omega t}\, \langle \boldsymbol{\mu}(0) \cdot \boldsymbol{\alpha}(t) \rangle.$$

Because both a dipole change and a polarizability change are required simultaneously, SFG is sensitive only to non-centrosymmetric environments such as interfaces.

**Chemical reaction rates** — the rate constant is related to the flux-flux autocorrelation function, where $F$ is the reactive flux through the transition state dividing surface:

$$k \propto \int_0^\infty dt\; \langle F(0)\, F(t) \rangle.$$

Integrating this TCF recovers the classical rate constant and forms the basis of reactive MD simulations.

All of these require long, statistically converged MD trajectories and thus demand Level 3 accuracy from the underlying MLIP — correct thermodynamics and dynamics, not just structural accuracy.

## 8.3 Training Data Strategies

How the training set is constructed has a major impact on both the quality and the data efficiency of the resulting model. Four strategies of increasing sophistication are used in practice.

**Strategy I — Fixed pre-generated dataset.** The simplest approach is to generate a large set of configurations using, for example, AIMD or random structure searches, compute their energies and forces with DFT, and train on this fixed corpus. The main limitation is that the dataset is constructed without reference to the specific observable of interest: configurations that are important during a long production run may not appear in the training data, particularly for rare events or high-temperature dynamics.

**Strategy II — Iterative training.** A closed-loop procedure that alternates between fitting and simulation. Each iteration extends coverage into regions of phase space actually visited during dynamics; the process converges when the model runs arbitrarily long MD without encountering novel configurations.

1. Fit an MLIP on the current dataset.
2. Run MD using that MLIP.
3. Identify configurations where the model fails or is uncertain.
4. Recompute those structures with the reference method (DFT or higher).
5. Add the new labelled structures to the dataset and go to 1.

**Strategy III — Active learning (unsupervised iterative).** Iterative training still requires a human-defined criterion for selecting "failed" configurations. Active learning replaces this with an automatic selection criterion based on the model's own uncertainty estimate — configurations where the model is least confident are selected for reference-method labelling, with no human judgement required. The central question is therefore: *how do we obtain that uncertainty estimate?* Two approaches are used in practice.

**(a) Bayesian / analytical uncertainty** — for linear and kernel models such as GAP and ACE, the posterior variance is available in closed form, exactly as derived in Section 4.4 for GPR. Given training observations $\boldsymbol{y}$ and a test point $\boldsymbol{x}_*$, the posterior predictive variance is

$$\sigma^2(\boldsymbol{x}_*) = k_{**} - \mathbf{k}_*^T(\mathbf{K} + \lambda\mathbf{I})^{-1}\mathbf{k}_*,$$

where $\mathbf{K}$ is the kernel (Gram) matrix, $\mathbf{k}_*$ is the vector of kernel evaluations between $\boldsymbol{x}_*$ and the training points, and $k_{**} = k(\boldsymbol{x}_*, \boldsymbol{x}_*)$ is the prior variance. For ACE and other explicit-basis models with basis vector $\mathbf{B}(\boldsymbol{x}_*)$ and posterior weight covariance $\boldsymbol{\Sigma}$, this reduces to the equivalent parametric form $\sigma = \sqrt{1/\lambda + \mathbf{B}^T\boldsymbol{\Sigma}\mathbf{B}}$. Either way, the uncertainty is exact and well-calibrated, but computing it requires inverting the $N_\text{basis} \times N_\text{basis}$ matrix $\mathbf{K} + \lambda\mathbf{I}$, which scales as $\mathcal{O}(N_\text{basis}^2)$ in storage and becomes prohibitive for large basis sets.

**(b) Committee (ensemble) uncertainty** — for nonlinear neural networks such as MACE, no tractable analytical posterior exists. Instead, a **committee** of $K$ independently trained models (differing in random seed, data subset, or hyperparameters) is used, and their disagreement serves as a surrogate uncertainty:

$$\tilde{\sigma} = \sqrt{\frac{1}{\lambda} + \frac{1}{K} \sum_{k=1}^K \left(E^k - \bar{E}\right)^2}.$$

Committee disagreement correlates well with the true error and requires no modification to the underlying model class — just $K$ forward passes instead of one. The $1/\lambda$ term is a Bayesian regularisation contribution; for NN committees it is often dropped, reducing $\tilde{\sigma}$ to the plain variance across predictions. Configurations where $\tilde{\sigma}$ exceeds a threshold are selected for reference-method recalculation and added to the training set.

**Strategy IV — Hyper-active Learning (HAL)** (van der Oord et al., *npj Comput. Mater.* **9**, 168, 2023). Rather than waiting for ordinary MD to stumble into uncertain regions, HAL biases the PES to actively steer the simulation there:

$$E_\text{HAL} = E - \tau \sigma.$$

Subtracting $\tau\sigma$ adds a bias force $\tau\,\nabla_i\sigma$ on top of the physical force, pointing in the direction of increasing uncertainty. The stopping criterion is the softmax-normalised relative force uncertainty $s(f_i)$: when $\max_i s(f_i) > s^\text{tol}$ (default 0.5), the run stops, the configuration is labelled by DFT, and the cycle restarts. The bias strength $\tau$ is a key hyperparameter — too small and the bias has no effect; too large and the trajectory escapes to unphysical geometries. Compared to plain AL, HAL reaches uncertain configurations in far fewer MD steps and requires fewer DFT evaluations to converge. HAL was developed for ACE but extends to MACE by replacing the analytical $\sigma$ with the committee spread $\tilde\sigma$.

## 8.4 Speed–Accuracy Benchmarks

Different MLIP architectures occupy different points in the speed–accuracy tradeoff. The following wall-clock times for a single MD trajectory on a single CPU core illustrate the spread:

| Model | Architecture | Equivariant | Effective body order | Accuracy | Speed (1 traj, 1 core) |
|-------|-------------|:-----------:|:--------------------:|:--------:|------------------------|
| DFT | Ab initio QM | ✓ | exact | reference | ~1 year |
| PaiNN | Equivariant MPNN (vector features) | ✓ | ~2-body msg | high | ~55 min |
| SchNet | Invariant MPNN (cfconv, scalar) | ✗ | ~2-body msg | moderate | ~45 min |
| MACE | Equivariant MPNN + ACE many-body | ✓ | up to 4-body | highest | ~1 min |
| REANN | MPNN with equivariant GTO basis, scalar messages | ✓ (basis) / ✗ (messages) | up to N-body via recursion | moderate–good | ~0.9 min |
| ACE | Linear model on many-body B-basis | ✗ (invariant descriptors) | up to N-body | good | ~33 s |

The speed gap between SchNet/PaiNN and MACE/ACE is mainly architectural: equivariance requires CG tensor contractions at every message-passing step, but MACE recovers speed through GPU efficiency and the fact that its many-body expansion is computed in a single pass. ACE is the fastest non-DFT option because inference reduces to a single dot product once the B-basis is precomputed.

## 8.5 Foundation Models

Training a separate MLIP for every chemical system is unsustainable. The foundation model approach instead trains a single large model on a massively diverse dataset spanning many elements and chemical environments, aiming for zero-shot generalization to new systems — much like large language models generalize across tasks. The first notable example was M3GNet (Chen & Ong, *Nat. Comput. Sci.*, 2022), trained on the Materials Project database. The most widely adopted model in this paradigm is **MACE-MP-0** (Batatia et al., 2024): trained on 89 elements and ~1.5M configurations from the Materials Project (XC = PBE+U), using 2 MACE layers, $l_\text{max}=3$, correlation order 3 (4-body), 128 channels, and $R_\text{cut}=6$ Å. It demonstrated that a single model can provide reasonable accuracy across inorganic materials without system-specific retraining, though it requires fine-tuning for demanding applications such as surface chemistry.

**Open Materials 2024 (OMat24)** pushed the dataset frontier further: 100M+ non-equilibrium structures generated via Boltzmann sampling at multiple temperatures and AIMD, providing far denser coverage of the PES than equilibrium-only datasets.

**Meta UMA** (Wood et al., *NeurIPS*, 2025) is a notable recent foundation model trained on ~500M structures, discussed here for its distinctive **Mixture of Linear Experts (MoLE)** architecture.

**Mixture of Experts (MoE).** Rather than one monolithic network, MoE uses $K$ parallel expert sub-networks plus a lightweight router that assigns a weight $\alpha_k$ to each expert depending on the input:

$$y = \sum_k \alpha_k(x)\, f_k(x), \qquad \sum_k \alpha_k = 1.$$

In sparse MoE (standard in LLMs), only the top-2 or top-3 experts are activated per input, so a model with $K$ experts worth of parameters only runs a small fraction per forward pass. The idea is specialization: the router learns to direct different types of inputs to different experts, without any single expert having to learn everything.

**Mixture of Linear Experts (MoLE).** UMA takes MoE further by restricting experts to linear weight matrices $W_k$, which allows the weighted sum to be collapsed before it ever touches the input:

$$y = \sum_k \alpha_k (W_k x) = \underbrace{\left(\sum_k \alpha_k W_k\right)}_{W^*} x.$$

The routing weights $\alpha_k$ depend only on global, time-invariant information (element composition, charge, spin, DFT task) — not on atomic positions. So $W^*$ is precomputed once before the simulation and reused at every MD step. The model gains the capacity of $K$ experts at the inference cost of one. UMA-M has 1.4B total parameters but only 50M active per step.

## 8.6 Transfer Learning and Fine-Tuning

**Transfer learning** is the idea of taking a model pre-trained on a large, general dataset and adapting it to a specific downstream task, rather than training from scratch. The pretrained model has already learned useful representations — in the MLIP context, a foundation model like MACE-MP-0 has learned general interatomic interactions across 89 elements. Fine-tuning then steers those representations toward a specific system (e.g. a particular surface or molecule) using a small amount of system-specific DFT data. This is far more data-efficient than training from scratch because the network does not need to relearn chemistry from zero.

Four fine-tuning strategies exist, each with different tradeoffs:

**Naive fine-tuning (full unfreeze).** All weights are updated on the new dataset. This gives the model maximum flexibility to adapt, but risks **catastrophic forgetting**: gradient updates for the new task overwrite the general representations learned during pretraining, degrading performance on everything outside the fine-tuning distribution. It also requires more data to work well.

**Freeze backbone, train head only.** The pretrained layers are frozen and only a small task-specific output head is trained. The general features are preserved exactly, and very little data is needed. The limitation is that the backbone cannot adapt at all, so if the target system involves chemistry poorly covered by the pretraining data, the fixed features may be insufficient regardless of how the head is trained.

**Multi-head.** A shared pretrained backbone is kept, and separate output heads are trained for each task. This avoids catastrophic forgetting entirely — old task heads are untouched — while allowing multiple specializations to coexist on top of the same feature extractor.

**LoRA (Low-Rank Adaptation).** The key observation is that the weight update $\Delta W$ needed to adapt to a new task has low *intrinsic rank* — adaptation does not require moving in all directions of parameter space, only a small subspace. LoRA makes this explicit: the pretrained weight matrix $W \in \mathbb{R}^{d \times k}$ is frozen, and only a low-rank correction is trained:

$$W' = W + \Delta W = W + BA,$$

where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with $r \ll \min(d, k)$. The number of trainable parameters drops from $dk$ to $r(d+k)$ — for a $768 \times 768$ matrix with $r=8$ this is roughly 50$\times$ fewer. $B$ is initialized to zero so $\Delta W = 0$ at the start, meaning training begins exactly from the pretrained weights with no disruption. After fine-tuning, $W + BA$ can be merged into a single matrix, so inference is identical to a non-LoRA model with no overhead.

**In the MLIP context** (Radova et al., 2025): fine-tuning MACE-MP-0 on just 10% of system-specific data outperforms training ACE from scratch on the full dataset, and beats a from-scratch MACE trained on fewer than 200 DFT points. MACE-MP-0 out of the box is inadequate for surface chemistry — fine-tuning bridges that gap efficiently. A practical workflow is MACE-MP-0 → fine-tuned MACE → distill into ACE for fast production MD.

## 8.7 Long-Range Interactions

GNNs and MLIPs are inherently local: the energy of atom $i$ depends only on atoms within a cutoff radius $R_\text{cut}$ (typically 5–12 Å). The dominant physical interaction that decays slowly enough to require special treatment is **electrostatics** ($E \sim 1/r$, essentially infinite range). Dispersion ($E \sim 1/r^6$) decays much faster and is often adequately captured within the cutoff or handled with a simple analytic correction (e.g. Grimme D3); it is not the primary motivation for the approaches below.

The standard framework for local neural network potentials is the **High-Dimensional Neural Network Potential (HDNNP)** introduced by Behler and Parrinello (2007). The field has since organized successive extensions into a generational taxonomy based on how electrostatics are handled.

**3rd generation HDNNPs — explicit Coulomb term.** The core idea is to augment the local short-range NN with an explicit Coulomb sum. The NN predicts atomic partial charges $\{q_i\}$ from the local chemical environment, and the total energy is:

$$E = E_\text{SR}^\text{NN} + E_\text{Coulomb}, \qquad E_\text{Coulomb} = \sum_{i<j} \frac{q_i q_j}{r_{ij}}$$

The $1/r$ sum runs over all pairs regardless of cutoff, solving the range problem for electrostatics. The limitation is that each $q_i$ is predicted from the local environment alone — there is no mechanism for charge transfer between distant atoms.

**PhysNet** (Unke & Meuwly, J. Chem. Theory Comput. 2019) is the prototypical 3rd gen model: a message-passing network that predicts local charges, energies, and dipole moments, with the explicit Coulomb term added on top. **SpookyNet** (Unke et al., Nat. Commun. 2021) goes further: it accepts the total molecular charge and spin state as explicit global inputs, and incorporates non-local effects via self-attention. This lets SpookyNet distinguish molecules in different electronic states that are otherwise locally identical — something PhysNet cannot do — but long-range charge transfer is still not fully captured.

**4th generation HDNNPs — charge equilibration (QEq).** To allow charge transfer, the charge assignment is made a global optimization problem (Ko et al., Nat. Commun. 12, 398, 2021). The NN predicts two atom-specific quantities from the local environment: an electronegativity $\chi_i$ and a chemical hardness $J_i$. The atomic charges $\{Q_i\}$ are then found by minimizing the total electrostatic energy:

$$E_\text{Qeq} = E_\text{elec} + \sum_i \left(\chi_i Q_i + \tfrac{1}{2}J_i Q_i^2\right)$$

$$E_\text{elec} = \sum_{i<j}\frac{\operatorname{erf}\!\left(r_{ij}/\sqrt{2}\,\gamma_{ij}\right)}{r_{ij}}Q_iQ_j + \sum_i\frac{Q_i^2}{2\sigma_i\sqrt{\pi}}$$

subject to charge conservation $\sum_i Q_i = Q_\text{tot}$. This is a quadratic problem solvable by matrix inversion. Because the minimization couples all $Q_i$ simultaneously through $E_\text{elec}$, the resulting charges depend on the global arrangement of all atoms — charge transfer is now possible. The $\operatorname{erf}$ factor is a Gaussian smearing that regularizes the short-range divergence. A concrete example is 12-cumulene, a fully conjugated carbon chain in which the $\pi$-electron system is delocalized over the whole molecule: a charge perturbation at one end polarizes the other end, which no local model captures.

**MACE-POLAR** (2025) implements a more sophisticated variant of this idea on the MACE backbone. Rather than simple QEq, it uses a polarizable field formalism where atomic charge and spin densities are represented as multipole expansions in a Gaussian-type orbital basis. Global charge and spin constraints are enforced through learnable Fukui equilibration functions, allowing the model to handle arbitrary charge and spin states and respond to external electric fields while remaining computationally efficient.

**MACE-LES — Latent Ewald Summation.** A more recent approach (npj Comp. Mat. 11, 80, 2025) avoids committing to physical charges entirely. The MACE network produces **latent charges** $q_i^\text{les} = \text{MLP}(h_i)$ from the learned node embeddings $h_i$ — abstract scalars with no constraint to reproduce physical partial charges, optimized end-to-end for accuracy. The long-range energy is computed via Ewald summation in reciprocal space:

$$E = E_\text{SR} + E_\text{LR}$$

$$E_\text{LR} = \frac{1}{\Omega}\sum_{0<|\mathbf{k}|<k_\text{cut}}\frac{1}{k^2}e^{-\sigma^2k^2/2}\left|\sum_i q_i^\text{les}\,e^{-i\mathbf{k}\cdot\mathbf{r}_i}\right|^2$$

where $\Omega$ is the simulation cell volume, $\mathbf{k}$ are reciprocal lattice vectors, and $\sigma$ controls the Gaussian damping that separates short-range from long-range contributions. The inner sum $\sum_i q_i^\text{les} e^{-i\mathbf{k}\cdot\mathbf{r}_i}$ is the structure factor at wavevector $\mathbf{k}$; it couples all atoms simultaneously in reciprocal space, giving the model global sensitivity.

## 8.8 Beyond MLIPs: Electronic Structure ML

Standard MLIPs learn the mapping $\{R_I, Z_I\} \to E$, from which forces follow by differentiation. The next level of ambition is to learn the **quantum mechanical Hamiltonian** directly:

$$\{R_I, Z_I\} \to H \xrightarrow{H\Psi=E\Psi} \Psi \to O = \langle\Psi|\hat{O}|\Psi\rangle$$

If $H$ is known, any quantum mechanical observable — density of states, band structure, wavefunctions, non-adiabatic couplings — follows by diagonalization or perturbation theory. This is fundamentally more powerful than learning individual properties one by one.

### The DFT Hamiltonian in an atomic orbital basis

In a DFT calculation using a localized atomic orbital basis $\{\chi_\mu\}$, the Kohn-Sham Hamiltonian is represented as a matrix with elements:

$$H_{\mu\nu}(\mathbf{R}) = \int \chi_\mu^*(\mathbf{r}) \left[\frac{1}{2}\nabla^2 + V_\text{eff}(\mathbf{r})\right] \chi_\nu(\mathbf{r}-\mathbf{R})\,d\mathbf{r}$$

where $V_\text{eff}$ is the effective Kohn-Sham potential. The basis functions are centered on atoms, so the matrix naturally decomposes into atom-pair blocks $H_{IJ}$: the block between atoms $I$ and $J$ collects all matrix elements between orbitals centered on $I$ and orbitals centered on $J$. The goal of electronic structure ML is to learn $H_{IJ}$ as a function of the local atomic geometry around that pair.

### SchNOrb

SchNOrb (Schütt et al., Nat. Commun. 10, 5024, 2019) was the first attempt to predict $H$ and the overlap matrix $S$ using a message-passing network (SchNet). It takes atomic distances $|\mathbf{r}_{ij}|$ and displacement vectors $\mathbf{r}_{ij}$ as inputs and outputs the Hamiltonian matrix elements in a local orbital basis.

Three fundamental problems limit SchNOrb. First, it is **not rotationally equivariant**: the Hamiltonian matrix elements transform under rotations via Wigner-D matrices (because they involve orbital angular momentum), but SchNOrb uses scalar features and cannot reproduce this transformation correctly. Second, it is **not transferable**: a separate model must be trained per molecule, making it impractical for general use. Third, it only handles **finite molecules** — no condensed phase or periodic systems.

### MACE-H

MACE-H (Qian et al., npj Comp. Mater., 2026) addresses all three problems by building on the equivariant MACE architecture. It introduces three types of computational blocks:

**MACE blocks** aggregate the local chemical environment into higher-body-order equivariant messages — the same mechanism as in the standard MACE potential, providing equivariant node features $h_i^{(t)}$.

**Node degree expansion (NDE) blocks** elevate the maximum angular momentum quantum number $l$ of the node features. This is necessary because the atomic orbital basis used in DFT (s, p, d, f orbitals) requires features at higher $l$ than a typical MLIP needs for energy prediction.

**Edge update blocks** convert the node-wise features into edge-wise features $e_{ij}$. This step is essential because the Hamiltonian block $H_{IJ}$ is a property of an atom *pair*, not an individual atom — the output must live on edges of the graph, not on nodes.

The resulting model achieves MAE = 0.278 meV per matrix element on Bi₂Te₃ bilayer, reproducing band structure and DOS from DFT, and has been applied to magic-angle bilayer graphene — a system of current interest for correlated electron physics — including spin-orbit coupling effects. Ongoing extensions include analytic derivatives, non-adiabatic couplings, and variational formulations for excited states.

---

