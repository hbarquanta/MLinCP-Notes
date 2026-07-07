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

<div id="speed-widget" style="background:#faf8f5;border:1px solid #e0dbd4;border-radius:8px;padding:1.2rem 1.2rem 0.9rem;margin:1.4rem 0;font-family:inherit;">
<div style="text-align:center;font-size:0.84rem;font-weight:700;color:#444;margin-bottom:0.9rem;">Speed &ndash; Accuracy Overview</div>
<div style="display:grid;grid-template-columns:70px 1fr 74px 96px;gap:0.28rem 0.6rem;align-items:center;">
<div style="font-size:0.63rem;color:#bbb;">Model</div>
<div style="font-size:0.63rem;color:#bbb;">Wall-clock time (log scale) &#x27F6;</div>
<div style="font-size:0.63rem;color:#bbb;text-align:right;">Time</div>
<div style="font-size:0.63rem;color:#bbb;text-align:center;">Accuracy</div>
<div style="font-size:0.77rem;font-weight:600;color:#444;">ACE</div>
<div style="height:20px;position:relative;"><div style="position:absolute;left:0;top:0;width:5%;height:100%;background:rgba(134,188,189,0.55);border-radius:0 3px 3px 0;border:1px solid #86BCBD;border-left:2.5px solid #86BCBD;"></div></div>
<div style="text-align:right;font-size:0.73rem;color:#777;">~33 s</div>
<div style="text-align:center;font-size:0.67rem;background:rgba(134,188,189,0.15);border:1px solid #86BCBD;border-radius:3px;padding:1px 4px;color:#1f6668;">good</div>
<div style="font-size:0.77rem;font-weight:600;color:#444;">REANN</div>
<div style="height:20px;position:relative;"><div style="position:absolute;left:0;top:0;width:8%;height:100%;background:rgba(164,206,139,0.45);border-radius:0 3px 3px 0;border:1px solid #A4CE8B;border-left:2.5px solid #A4CE8B;"></div></div>
<div style="text-align:right;font-size:0.73rem;color:#777;">~54 s</div>
<div style="text-align:center;font-size:0.67rem;background:rgba(164,206,139,0.15);border:1px solid #A4CE8B;border-radius:3px;padding:1px 4px;color:#2a5a1a;">mod&ndash;good</div>
<div style="font-size:0.77rem;font-weight:700;color:#2a5a1a;">MACE &#9733;</div>
<div style="height:20px;position:relative;"><div style="position:absolute;left:0;top:0;width:9%;height:100%;background:rgba(90,154,64,0.5);border-radius:0 3px 3px 0;border:1.5px solid #5a9a40;border-left:2.5px solid #5a9a40;"></div></div>
<div style="text-align:right;font-size:0.73rem;color:#777;">~1 min</div>
<div style="text-align:center;font-size:0.67rem;background:rgba(90,154,64,0.15);border:1.5px solid #5a9a40;border-radius:3px;padding:1px 4px;color:#2a5a1a;font-weight:600;">highest</div>
<div style="font-size:0.77rem;font-weight:600;color:#444;">SchNet</div>
<div style="height:20px;position:relative;"><div style="position:absolute;left:0;top:0;width:35%;height:100%;background:rgba(200,173,58,0.35);border-radius:0 3px 3px 0;border:1px solid #c8ad3a;border-left:2.5px solid #c8ad3a;"></div></div>
<div style="text-align:right;font-size:0.73rem;color:#777;">~45 min</div>
<div style="text-align:center;font-size:0.67rem;background:rgba(200,173,58,0.12);border:1px solid #c8ad3a;border-radius:3px;padding:1px 4px;color:#5a4c00;">moderate</div>
<div style="font-size:0.77rem;font-weight:600;color:#444;">PaiNN</div>
<div style="height:20px;position:relative;"><div style="position:absolute;left:0;top:0;width:37%;height:100%;background:rgba(134,188,189,0.35);border-radius:0 3px 3px 0;border:1px solid #86BCBD;border-left:2.5px solid #86BCBD;"></div></div>
<div style="text-align:right;font-size:0.73rem;color:#777;">~55 min</div>
<div style="text-align:center;font-size:0.67rem;background:rgba(134,188,189,0.15);border:1px solid #86BCBD;border-radius:3px;padding:1px 4px;color:#1f6668;">high</div>
<div style="font-size:0.77rem;font-weight:600;color:#444;">DFT</div>
<div style="height:20px;position:relative;"><div style="position:absolute;left:0;top:0;width:100%;height:100%;background:rgba(186,90,90,0.3);border-radius:0 3px 3px 0;border:1px solid #BA5A5A;border-left:2.5px solid #BA5A5A;"></div></div>
<div style="text-align:right;font-size:0.73rem;color:#777;">~1 year</div>
<div style="text-align:center;font-size:0.67rem;background:rgba(186,90,90,0.12);border:1px solid #BA5A5A;border-radius:3px;padding:1px 4px;color:#BA5A5A;">reference</div>
</div>
<div style="font-size:0.63rem;color:#ccc;text-align:center;margin-top:0.5rem;">Bar width &prop; log<sub>10</sub>(time). MACE/ACE/REANN &asymp; 1 min; PaiNN/SchNet &asymp; 1 h; DFT &asymp; 1 year (10<sup>6</sup>&times; slower than MACE).</div>
</div>

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


<div id="fm-widget" style="background:#faf8f5;border:1px solid #e0dbd4;border-radius:8px;padding:1.1rem 1rem 0.8rem;margin:1.8rem 0;font-family:inherit;">
<div style="text-align:center;font-size:0.84rem;font-weight:700;color:#444;margin-bottom:0.8rem;">Foundation Model Scale &mdash; Training Data &amp; Parameters</div>
<div style="display:flex;gap:0.6rem;align-items:flex-end;height:150px;padding:0 0.4rem;margin-bottom:0;">
<div style="flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%;">
<div style="font-size:0.61rem;color:#aaa;margin-bottom:0.1rem;">170k params</div>
<div style="width:100%;height:15%;background:rgba(134,188,189,0.3);border-radius:4px 4px 0 0;border:1.5px solid #86BCBD;border-bottom:none;"></div></div>
<div style="flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%;">
<div style="font-size:0.61rem;color:#aaa;margin-bottom:0.1rem;">~5M params</div>
<div style="width:100%;height:37%;background:rgba(164,206,139,0.35);border-radius:4px 4px 0 0;border:1.5px solid #A4CE8B;border-bottom:none;"></div></div>
<div style="flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%;">
<div style="font-size:0.61rem;color:#aaa;margin-bottom:0.1rem;">large (eSEN)</div>
<div style="width:100%;height:83%;background:rgba(200,173,58,0.25);border-radius:4px 4px 0 0;border:1.5px solid #c8ad3a;border-bottom:none;"></div></div>
<div style="flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%;">
<div style="font-size:0.61rem;color:#aaa;margin-bottom:0.1rem;">1.4B (50M active)</div>
<div style="width:100%;height:100%;background:rgba(186,90,90,0.25);border-radius:4px 4px 0 0;border:1.5px solid #BA5A5A;border-bottom:none;"></div></div>
</div>
<div style="display:flex;gap:0.6rem;padding:0 0.4rem;">
<div style="flex:1;text-align:center;border-top:2px solid #86BCBD;padding-top:0.3rem;"><div style="font-size:0.72rem;font-weight:700;color:#444;">M3GNet</div><div style="font-size:0.62rem;color:#888;">2022 &middot; 89 el.</div><div style="font-size:0.62rem;color:#aaa;">190k struct.</div></div>
<div style="flex:1;text-align:center;border-top:2px solid #A4CE8B;padding-top:0.3rem;"><div style="font-size:0.72rem;font-weight:700;color:#444;">MACE-MP-0</div><div style="font-size:0.62rem;color:#888;">2024 &middot; 89 el.</div><div style="font-size:0.62rem;color:#aaa;">1.5M struct.</div></div>
<div style="flex:1;text-align:center;border-top:2px solid #c8ad3a;padding-top:0.3rem;"><div style="font-size:0.72rem;font-weight:700;color:#444;">OMat24</div><div style="font-size:0.62rem;color:#888;">2024 &middot; many el.</div><div style="font-size:0.62rem;color:#aaa;">100M+ struct.</div></div>
<div style="flex:1;text-align:center;border-top:2px solid #BA5A5A;padding-top:0.3rem;"><div style="font-size:0.72rem;font-weight:700;color:#444;">Meta UMA</div><div style="font-size:0.62rem;color:#888;">2025 &middot; MoLE arch.</div><div style="font-size:0.62rem;color:#aaa;">500M struct.</div></div>
</div>
<div style="font-size:0.64rem;color:#ccc;text-align:center;margin-top:0.45rem;">Bar height &prop; log(structures), range-scaled to show spread. Parameter counts above bars.</div>
</div>

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


<div id="ft-widget" style="background:#faf8f5;border:1px solid #e0dbd4;border-radius:8px;padding:1.1rem 1rem;margin:1.8rem 0;font-family:inherit;">
<div style="text-align:center;font-size:0.84rem;font-weight:700;color:#444;margin-bottom:0.75rem;">Fine-Tuning Strategies &mdash; Which Parameters are Trained?</div>
<div style="display:flex;gap:0.8rem;justify-content:center;flex-wrap:wrap;">

<div style="text-align:center;min-width:100px;">
<div style="display:inline-flex;flex-direction:column;gap:3px;align-items:center;margin-bottom:0.35rem;">
<div style="font-size:0.62rem;color:#5a9a40;font-style:italic;">output</div>
<div style="width:68px;height:13px;background:#A4CE8B;border-radius:2px;border:1px solid #7ab865;"></div>
<div style="width:68px;height:13px;background:#A4CE8B;border-radius:2px;border:1px solid #7ab865;"></div>
<div style="width:68px;height:13px;background:#A4CE8B;border-radius:2px;border:1px solid #7ab865;"></div>
<div style="width:68px;height:13px;background:#A4CE8B;border-radius:2px;border:1px solid #7ab865;"></div>
<div style="width:68px;height:13px;background:#A4CE8B;border-radius:2px;border:1px solid #7ab865;"></div>
<div style="font-size:0.62rem;color:#5a9a40;font-style:italic;">input</div>
</div>
<div style="font-size:0.73rem;font-weight:700;color:#444;">Naive</div>
<div style="font-size:0.63rem;color:#888;">All updated</div>
<div style="font-size:0.62rem;color:#BA5A5A;">Forgetting risk</div>
</div>

<div style="text-align:center;min-width:100px;">
<div style="display:inline-flex;flex-direction:column;gap:3px;align-items:center;margin-bottom:0.35rem;">
<div style="font-size:0.62rem;color:#5a9a40;font-style:italic;">output</div>
<div style="width:68px;height:13px;background:#A4CE8B;border-radius:2px;border:1px solid #7ab865;"></div>
<div style="width:68px;height:13px;background:#f0ece6;border-radius:2px;border:1px solid #c8c2ba;"></div>
<div style="width:68px;height:13px;background:#f0ece6;border-radius:2px;border:1px solid #c8c2ba;"></div>
<div style="width:68px;height:13px;background:#f0ece6;border-radius:2px;border:1px solid #c8c2ba;"></div>
<div style="width:68px;height:13px;background:#f0ece6;border-radius:2px;border:1px solid #c8c2ba;"></div>
<div style="font-size:0.62rem;color:#aaa;font-style:italic;">frozen</div>
</div>
<div style="font-size:0.73rem;font-weight:700;color:#444;">Freeze backbone</div>
<div style="font-size:0.63rem;color:#888;">Head only</div>
<div style="font-size:0.62rem;color:#888;">Very data-efficient</div>
</div>

<div style="text-align:center;min-width:110px;">
<div style="display:inline-flex;flex-direction:column;gap:3px;align-items:center;margin-bottom:0.35rem;">
<div style="font-size:0.62rem;color:#5a9a40;font-style:italic;">task A &nbsp; task B</div>
<div style="display:flex;gap:3px;">
<div style="width:32px;height:13px;background:#A4CE8B;border-radius:2px;border:1px solid #7ab865;"></div>
<div style="width:32px;height:13px;background:#86BCBD;border-radius:2px;border:1px solid #5a9a9c;"></div>
</div>
<div style="width:68px;height:13px;background:#f0ece6;border-radius:2px;border:1px solid #c8c2ba;"></div>
<div style="width:68px;height:13px;background:#f0ece6;border-radius:2px;border:1px solid #c8c2ba;"></div>
<div style="width:68px;height:13px;background:#f0ece6;border-radius:2px;border:1px solid #c8c2ba;"></div>
<div style="width:68px;height:13px;background:#f0ece6;border-radius:2px;border:1px solid #c8c2ba;"></div>
<div style="font-size:0.62rem;color:#aaa;font-style:italic;">shared frozen</div>
</div>
<div style="font-size:0.73rem;font-weight:700;color:#444;">Multi-head</div>
<div style="font-size:0.63rem;color:#888;">Shared backbone</div>
<div style="font-size:0.62rem;color:#888;">No forgetting</div>
</div>

<div style="text-align:center;min-width:110px;">
<div style="display:inline-flex;flex-direction:column;gap:3px;align-items:flex-start;margin-bottom:0.35rem;">
<div style="font-size:0.62rem;color:#5a9a40;font-style:italic;text-align:center;width:100%;">output</div>
<div style="width:68px;height:13px;background:#A4CE8B;border-radius:2px;border:1px solid #7ab865;"></div>
<div style="display:flex;align-items:center;gap:2px;">
<div style="width:68px;height:13px;background:#f0ece6;border-radius:2px;border:1px solid #c8c2ba;"></div>
<div style="width:18px;height:9px;background:#f7e49b;border-radius:2px;border:1px solid #c8ad3a;font-size:5.5px;color:#555;text-align:center;line-height:9px;">BA</div>
</div>
<div style="display:flex;align-items:center;gap:2px;">
<div style="width:68px;height:13px;background:#f0ece6;border-radius:2px;border:1px solid #c8c2ba;"></div>
<div style="width:18px;height:9px;background:#f7e49b;border-radius:2px;border:1px solid #c8ad3a;font-size:5.5px;color:#555;text-align:center;line-height:9px;">BA</div>
</div>
<div style="display:flex;align-items:center;gap:2px;">
<div style="width:68px;height:13px;background:#f0ece6;border-radius:2px;border:1px solid #c8c2ba;"></div>
<div style="width:18px;height:9px;background:#f7e49b;border-radius:2px;border:1px solid #c8ad3a;font-size:5.5px;color:#555;text-align:center;line-height:9px;">BA</div>
</div>
<div style="display:flex;align-items:center;gap:2px;">
<div style="width:68px;height:13px;background:#f0ece6;border-radius:2px;border:1px solid #c8c2ba;"></div>
<div style="width:18px;height:9px;background:#f7e49b;border-radius:2px;border:1px solid #c8ad3a;font-size:5.5px;color:#555;text-align:center;line-height:9px;">BA</div>
</div>
<div style="font-size:0.62rem;color:#aaa;font-style:italic;">frozen + adapters</div>
</div>
<div style="font-size:0.73rem;font-weight:700;color:#444;">LoRA</div>
<div style="font-size:0.63rem;color:#888;">Frozen + rank-<em>r</em> BA</div>
<div style="font-size:0.62rem;color:#5a9a40;">r &#8810; d,k</div>
</div>
</div>

<hr style="margin:0.75rem 0;border:none;border-top:1px solid #e5dfd7;">
<div style="text-align:center;font-size:0.8rem;font-weight:700;color:#444;margin-bottom:0.6rem;">LoRA Parameter Savings &mdash; Interactive</div>
<svg id="lora-svg" style="display:block;width:100%;height:200px;"></svg>
<div id="lora-info" style="background:#f5f2ee;border-radius:6px;padding:0.55rem 0.75rem;border:1px solid #e5dfd7;font-size:0.79rem;max-width:520px;margin:0.5rem auto 0;"></div>
<div style="margin-top:0.55rem;font-size:0.74rem;color:#555;display:grid;grid-template-columns:auto 1fr 52px;gap:0.12rem 0.4rem;align-items:center;">
<label>Dim <em>d</em></label>
<input type="range" id="lora-d" min="128" max="2048" step="128" value="768" style="width:100%;height:16px;" oninput="loraUp()">
<span id="lora-dv" style="text-align:right;font-variant-numeric:tabular-nums;">768</span>
<label>Dim <em>k</em></label>
<input type="range" id="lora-k" min="128" max="2048" step="128" value="768" style="width:100%;height:16px;" oninput="loraUp()">
<span id="lora-kv" style="text-align:right;font-variant-numeric:tabular-nums;">768</span>
<label>Rank <em>r</em></label>
<input type="range" id="lora-r" min="1" max="64" value="8" style="width:100%;height:16px;" oninput="loraUp()">
<span id="lora-rv" style="text-align:right;font-variant-numeric:tabular-nums;">8</span>
</div>
</div>
<script>
(function(){
var NS='http://www.w3.org/2000/svg';
function mk(tag,at,tx){var e=document.createElementNS(NS,tag);for(var k in at)e.setAttribute(k,at[k]);if(tx!=null)e.textContent=tx;return e;}
function kx(s){try{return katex.renderToString(s,{throwOnError:false,displayMode:false});}catch(e){return s;}}
window.loraUp=function(){
  var d=parseInt(document.getElementById('lora-d').value);
  var kk=parseInt(document.getElementById('lora-k').value);
  var r=parseInt(document.getElementById('lora-r').value);
  document.getElementById('lora-dv').textContent=d;
  document.getElementById('lora-kv').textContent=kk;
  document.getElementById('lora-rv').textContent=r;
  var full=d*kk, lp=r*(d+kk), red=(full/lp).toFixed(1), pct=(lp/full*100).toFixed(1);
  var svg=document.getElementById('lora-svg');
  if(!svg)return;
  var W=Math.max(svg.getBoundingClientRect().width||0,360), H=200;
  svg.setAttribute('viewBox','0 0 '+W+' '+H);
  svg.innerHTML='';
  var asp=d/kk, bw=90, bh=Math.min(110,Math.max(44,Math.round(bw*asp)));
  var by=Math.round((H-bh)/2);
  var bmw=Math.max(9,Math.min(28,Math.round(r*1.1)));
  var amh=Math.max(9,Math.min(28,Math.round(r*1.1)));
  var amw=bw;
  var tw=bw+20+bmw+5+amw+20+bw;
  var ox=Math.max(8,Math.round((W-tw)/2));
  // W (frozen)
  svg.appendChild(mk('rect',{x:ox,y:by,width:bw,height:bh,fill:'#f0ece6',stroke:'#c8c2ba','stroke-width':1.5,rx:3}));
  svg.appendChild(mk('text',{x:ox+bw/2,y:by-6,'text-anchor':'middle','font-size':9.5,'fill':'#aaa'},'frozen'));
  svg.appendChild(mk('text',{x:ox+bw/2,y:by+bh/2+5,'text-anchor':'middle','font-size':13,'fill':'#777','font-weight':'600'},'W'));
  svg.appendChild(mk('text',{x:ox+bw/2,y:by+bh+14,'text-anchor':'middle','font-size':8,'fill':'#bbb'},d+'\xd7'+kk));
  // + sign
  svg.appendChild(mk('text',{x:ox+bw+10,y:H/2+6,'text-anchor':'middle','font-size':18,'fill':'#aaa'},'+'));
  // B (d\xd7r)
  var bmx=ox+bw+20, bmy=by;
  svg.appendChild(mk('rect',{x:bmx,y:bmy,width:bmw,height:bh,fill:'rgba(164,206,139,0.3)',stroke:'#A4CE8B','stroke-width':1.5,rx:2}));
  svg.appendChild(mk('text',{x:bmx+bmw/2,y:bmy-6,'text-anchor':'middle','font-size':9.5,'fill':'#5a9a40'},'B'));
  svg.appendChild(mk('text',{x:bmx+bmw/2,y:bmy+bh+14,'text-anchor':'middle','font-size':8,'fill':'#bbb'},d+'\xd7'+r));
  // A (r\xd7k)
  var amx=bmx+bmw+5, amy=by+Math.round(bh/2-amh/2);
  svg.appendChild(mk('rect',{x:amx,y:amy,width:amw,height:amh,fill:'rgba(164,206,139,0.3)',stroke:'#A4CE8B','stroke-width':1.5,rx:2}));
  svg.appendChild(mk('text',{x:amx+amw/2,y:amy-6,'text-anchor':'middle','font-size':9.5,'fill':'#5a9a40'},'A'));
  svg.appendChild(mk('text',{x:amx+amw/2,y:amy+amh+14,'text-anchor':'middle','font-size':8,'fill':'#bbb'},r+'\xd7'+kk));
  // =
  var eqx=amx+amw+10;
  svg.appendChild(mk('text',{x:eqx,y:H/2+6,'text-anchor':'middle','font-size':16,'fill':'#aaa'},'='));
  // W’ dashed
  var wpx=eqx+10;
  svg.appendChild(mk('rect',{x:wpx,y:by,width:bw,height:bh,fill:'rgba(164,206,139,0.07)',stroke:'#A4CE8B','stroke-width':2,rx:3,'stroke-dasharray':'4,2.5'}));
  svg.appendChild(mk('text',{x:wpx+bw/2,y:by-6,'text-anchor':'middle','font-size':9.5,'fill':'#5a9a40'},"W’=W+BA"));
  svg.appendChild(mk('text',{x:wpx+bw/2,y:by+bh+14,'text-anchor':'middle','font-size':8,'fill':'#bbb'},d+'\xd7'+kk));
  // info panel
  var info=document.getElementById('lora-info');
  if(!info)return;
  info.innerHTML='<div style="font-weight:700;color:#555;margin-bottom:0.3rem;">Parameters</div>'+
    '<table style="border-collapse:collapse;width:100%;font-size:0.8rem;"><tbody>'+
    '<tr><td style="color:#888;padding-right:0.5rem;padding-bottom:0.18rem;">Full W:</td><td style="font-variant-numeric:tabular-nums;padding-bottom:0.18rem;font-weight:600;">'+(full).toLocaleString()+'</td></tr>'+
    '<tr><td style="color:#5a9a40;padding-right:0.5rem;">B + A:</td><td style="color:#5a9a40;font-weight:600;font-variant-numeric:tabular-nums;">'+(lp).toLocaleString()+'</td></tr>'+
    '</tbody></table>'+
    '<hr style="margin:0.3rem 0;border:none;border-top:1px solid #e0dbd4;">'+
    '<div style="font-size:0.82rem;"><strong>'+red+'\xd7</strong> fewer params&nbsp;&nbsp;<span style="color:#aaa;font-size:0.72rem;">'+pct+'% of original</span></div>'+
    '<hr style="margin:0.3rem 0;border:none;border-top:1px solid #e0dbd4;">'+
    "<div style='font-size:0.73rem;color:#bbb;'>"+kx("W'=W+BA")+'; &thinsp;B&thinsp;init to&thinsp;0</div>';
};
function init(){if(!document.getElementById('lora-svg'))return;loraUp();}
if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',init);}else{init();}
if(typeof document$!=='undefined'){document$.subscribe(function(){setTimeout(init,50);});}
})();
</script>

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


<div id="lr-widget" style="background:#faf8f5;border:1px solid #e0dbd4;border-radius:8px;padding:1.1rem 1rem;margin:1.8rem 0;font-family:inherit;">
<div style="text-align:center;font-size:0.84rem;font-weight:700;color:#444;margin-bottom:0.75rem;">Long-Range Interactions &mdash; Generational Progression</div>
<div style="display:flex;gap:0.5rem;flex-wrap:wrap;">

<div style="flex:1;min-width:140px;border:1.5px solid #c8c2ba;border-radius:6px;overflow:hidden;">
<div style="background:rgba(200,194,186,0.2);padding:0.3rem 0.6rem;font-size:0.7rem;font-weight:700;color:#666;letter-spacing:0.03em;border-bottom:1px solid #c8c2ba;">2nd gen HDNNP</div>
<div style="padding:0.35rem 0.6rem;">
<div style="font-size:0.65rem;color:#888;margin-bottom:0.25rem;">Local SR network only</div>
<span data-eq="E = E_\text{SR}^\text{NN}" data-dm="1" style="display:block;margin:0.2rem 0;"></span>
<div style="font-size:0.63rem;color:#BA5A5A;margin-top:0.3rem;">&#10007; No long-range<br>&#10007; Fails ionic/polar</div>
</div>
</div>

<div style="display:flex;align-items:center;color:#aaa;font-size:1rem;">&#8594;</div>

<div style="flex:1;min-width:140px;border:1.5px solid #86BCBD;border-radius:6px;overflow:hidden;">
<div style="background:rgba(134,188,189,0.12);padding:0.3rem 0.6rem;font-size:0.7rem;font-weight:700;color:#0f3a3c;letter-spacing:0.03em;border-bottom:1px solid #86BCBD;">3rd gen &middot; PhysNet / SpookyNet</div>
<div style="padding:0.35rem 0.6rem;">
<div style="font-size:0.65rem;color:#888;margin-bottom:0.25rem;">NN predicts local charges; explicit Coulomb</div>
<span data-eq="E = E_\text{SR} + \sum_{i<j}\frac{q_i q_j}{r_{ij}}" data-dm="1" style="display:block;margin:0.2rem 0;"></span>
<div style="font-size:0.63rem;color:#5a9a40;margin-top:0.3rem;">&#10003; 1/r electrostatics<br>&#10007; No charge transfer</div>
</div>
</div>

<div style="display:flex;align-items:center;color:#aaa;font-size:1rem;">&#8594;</div>

<div style="flex:1;min-width:140px;border:1.5px solid #c8ad3a;border-radius:6px;overflow:hidden;">
<div style="background:rgba(200,173,58,0.12);padding:0.3rem 0.6rem;font-size:0.7rem;font-weight:700;color:#5a4c00;letter-spacing:0.03em;border-bottom:1px solid #c8ad3a;">4th gen &middot; QEq (Ko et al.)</div>
<div style="padding:0.35rem 0.6rem;">
<div style="font-size:0.65rem;color:#888;margin-bottom:0.25rem;">Global charge equilibration; NN predicts &chi;<sub>i</sub>, J<sub>i</sub></div>
<span data-eq="\min_{\{Q_i\}} E_\text{Qeq} \text{ s.t. } \textstyle\sum_i Q_i = Q_\text{tot}" data-dm="1" style="display:block;margin:0.2rem 0;"></span>
<div style="font-size:0.63rem;color:#5a9a40;margin-top:0.3rem;">&#10003; Charge transfer<br>&#10007; Physical charge req'd</div>
</div>
</div>

<div style="display:flex;align-items:center;color:#aaa;font-size:1rem;">&#8594;</div>

<div style="flex:1;min-width:140px;border:1.5px solid #A4CE8B;border-radius:6px;overflow:hidden;">
<div style="background:rgba(164,206,139,0.15);padding:0.3rem 0.6rem;font-size:0.7rem;font-weight:700;color:#2a5a1a;letter-spacing:0.03em;border-bottom:1px solid #A4CE8B;">MACE-LES &middot; Latent Ewald</div>
<div style="padding:0.35rem 0.6rem;">
<div style="font-size:0.65rem;color:#888;margin-bottom:0.25rem;">Learned latent charges; Ewald in reciprocal space</div>
<span data-eq="E_\text{LR} = \frac{1}{\Omega}\sum_\mathbf{k}\frac{e^{-\sigma^2 k^2}}{k^2}\bigl|\textstyle\sum_i q_i^\text{les} e^{-i\mathbf{k}\cdot\mathbf{r}_i}\bigr|^2" data-dm="1" style="display:block;margin:0.2rem 0;"></span>
<div style="font-size:0.63rem;color:#5a9a40;margin-top:0.3rem;">&#10003; No physical charges<br>&#10003; E2E trainable</div>
</div>
</div>

</div>
</div>
<script>
(function(){
function lrInit(){
  var el=document.getElementById('lr-widget');
  if(!el||el.dataset.lrInit)return;
  el.dataset.lrInit='1';
  el.querySelectorAll('[data-eq]').forEach(function(e){
    try{var eq=e.getAttribute('data-eq'),dm=e.getAttribute('data-dm')==='1';
    e.innerHTML=katex.renderToString(eq,{throwOnError:false,displayMode:dm});}catch(err){}
  });
}
if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',lrInit);}else{lrInit();}
if(typeof document$!=='undefined'){document$.subscribe(function(){setTimeout(lrInit,80);});}
})();
</script>

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


<div id="maceh-widget" style="background:#faf8f5;border:1px solid #e0dbd4;border-radius:8px;padding:1.1rem 0.9rem;margin:1.8rem 0;font-family:inherit;">
<div style="text-align:center;font-size:0.84rem;font-weight:700;color:#444;margin-bottom:0.75rem;">MACE-H &mdash; Equivariant Hamiltonian Learning</div>

<div style="display:flex;justify-content:center;gap:0.7rem;margin-bottom:0.15rem;flex-wrap:wrap;">
<div style="background:#f0ece6;border:1.5px solid #c8c2ba;border-radius:5px;padding:0.22rem 0.85rem;font-size:0.8rem;font-weight:600;color:#555;">Geometry <span data-eq="\{R_I, Z_I\}" data-dm="0"></span></div>
</div>
<svg width="14" height="22" viewBox="0 0 14 22" style="display:block;margin:0.1rem auto;"><line x1="7" y1="0" x2="7" y2="13" stroke="#555" stroke-width="2"/><polygon points="2,11 7,20 12,11" fill="#555"/></svg>

<div style="border:1.5px solid #A4CE8B;border-radius:6px;overflow:hidden;margin:0.15rem 0;">
<div style="font-size:0.74rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;padding:0.28rem 0.7rem;background:rgba(164,206,139,0.12);color:#2a5a1a;border-bottom:1px solid #A4CE8B;">MACE Blocks <span style="font-weight:400;text-transform:none;color:#888;">&middot; &times;T &mdash; equivariant message passing</span></div>
<div style="padding:0.4rem 0.6rem;overflow-x:auto;">
<div style="font-size:0.69rem;color:#999;margin-bottom:0.1rem;">Output: equivariant node features per atom</div>
<span data-eq="h_i^{(t)} \in \bigoplus_{l=0}^{L_\text{max}} \mathbb{R}^{K\times(2l+1)}" data-dm="1"></span>
</div>
</div>
<svg width="14" height="22" viewBox="0 0 14 22" style="display:block;margin:0.1rem auto;"><line x1="7" y1="0" x2="7" y2="13" stroke="#555" stroke-width="2"/><polygon points="2,11 7,20 12,11" fill="#555"/></svg>

<div style="border:1.5px solid #86BCBD;border-radius:6px;overflow:hidden;margin:0.15rem 0;">
<div style="font-size:0.74rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;padding:0.28rem 0.7rem;background:rgba(134,188,189,0.12);color:#0f3a3c;border-bottom:1px solid #86BCBD;">Node Degree Expansion (NDE) <span style="font-weight:400;text-transform:none;color:#888;">&middot; elevates l<sub>max</sub> for orbital basis</span></div>
<div style="padding:0.4rem 0.6rem;overflow-x:auto;">
<div style="font-size:0.69rem;color:#999;margin-bottom:0.1rem;">Elevates angular momentum to match AO basis (s, p, d, f &rarr; higher l)</div>
<span data-eq="h_i \;\xrightarrow{\;\text{NDE}\;}\; \tilde{h}_i \;\text{ with }l_\text{max}\text{ matching AO basis}" data-dm="1"></span>
</div>
</div>
<svg width="14" height="22" viewBox="0 0 14 22" style="display:block;margin:0.1rem auto;"><line x1="7" y1="0" x2="7" y2="13" stroke="#555" stroke-width="2"/><polygon points="2,11 7,20 12,11" fill="#555"/></svg>

<div style="border:1.5px solid #C47070;border-radius:6px;overflow:hidden;margin:0.15rem 0;">
<div style="font-size:0.74rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;padding:0.28rem 0.7rem;background:rgba(186,90,90,0.08);color:#5a1a1a;border-bottom:1px solid #C47070;">Edge Update <span style="font-weight:400;text-transform:none;color:#888;">&middot; node features &rarr; atom-pair features e<sub>ij</sub></span></div>
<div style="padding:0.4rem 0.6rem;overflow-x:auto;">
<div style="font-size:0.69rem;color:#999;margin-bottom:0.1rem;">H<sub>IJ</sub> is a pair property &mdash; must live on graph edges, not nodes</div>
<span data-eq="e_{ij} = f\!\left(\tilde{h}_i,\;\tilde{h}_j,\;\hat{\mathbf{r}}_{ij}\right)" data-dm="1"></span>
</div>
</div>
<svg width="14" height="22" viewBox="0 0 14 22" style="display:block;margin:0.1rem auto;"><line x1="7" y1="0" x2="7" y2="13" stroke="#555" stroke-width="2"/><polygon points="2,11 7,20 12,11" fill="#555"/></svg>

<div style="border:1.5px solid #aaa;border-radius:6px;overflow:hidden;margin:0.15rem 0;">
<div style="font-size:0.74rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;padding:0.28rem 0.7rem;background:rgba(170,170,170,0.1);color:#444;border-bottom:1px solid #ccc;">Output <span style="font-weight:400;text-transform:none;color:#888;">&middot; Hamiltonian block &rarr; observables via diagonalisation</span></div>
<div style="padding:0.4rem 0.45rem;">
<div style="display:flex;gap:0.4rem;flex-wrap:wrap;">
<div style="flex:1;min-width:160px;background:#fff;border:1px solid #e5dfd7;border-radius:5px;padding:0.35rem 0.6rem;overflow-x:auto;">
<div style="font-size:0.69rem;font-weight:700;color:#999;margin-bottom:0.1rem;">Hamiltonian block</div>
<span data-eq="H_{IJ}(\mathbf{R}) = \mathrm{Linear}(e_{ij})\in\mathbb{R}^{n_I\times n_J}" data-dm="1"></span>
</div>
<div style="flex:1;min-width:160px;background:#fff;border:1px solid #e5dfd7;border-radius:5px;padding:0.35rem 0.6rem;overflow-x:auto;">
<div style="font-size:0.69rem;font-weight:700;color:#999;margin-bottom:0.1rem;">Observables via diagonalisation</div>
<span data-eq="H\Psi = E\Psi \;\Rightarrow\; \text{DOS, bands, couplings}" data-dm="1"></span>
</div>
</div>
</div>
</div>
</div>
<script>
(function(){
function mhInit(){
  var el=document.getElementById('maceh-widget');
  if(!el||el.dataset.mhInit)return;
  el.dataset.mhInit='1';
  el.querySelectorAll('[data-eq]').forEach(function(e){
    try{var eq=e.getAttribute('data-eq'),dm=e.getAttribute('data-dm')==='1';
    e.innerHTML=katex.renderToString(eq,{throwOnError:false,displayMode:dm});}catch(err){}
  });
}
if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',mhInit);}else{mhInit();}
if(typeof document$!=='undefined'){document$.subscribe(function(){setTimeout(mhInit,80);});}
})();
</script>

---

