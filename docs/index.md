# Machine Learning in Computational Physics

Lecture notes for the Machine Learning in Computational Physics course.

## Table of Contents

### [Chapter 1 — Introduction & Core ML Concepts](chapter-01-introduction-core-ml-concepts.md)

- [1.1 Learning Paradigms](chapter-01-introduction-core-ml-concepts.md#11-learning-paradigms)
- [1.2 Risk Minimization and Loss Functions](chapter-01-introduction-core-ml-concepts.md#12-risk-minimization-and-loss-functions)
- [1.3 Gradient-Based Optimization](chapter-01-introduction-core-ml-concepts.md#13-gradient-based-optimization)
- [1.4 Overfitting, Regularization, and Model Validation](chapter-01-introduction-core-ml-concepts.md#14-overfitting-regularization-and-model-validation)
- [1.5 Data Splitting and Cross-Validation](chapter-01-introduction-core-ml-concepts.md#15-data-splitting-and-cross-validation)

### [Chapter 2 — Descriptors & Featurization](chapter-02-descriptors-featurization.md)

- [2.1 Why Descriptors?](chapter-02-descriptors-featurization.md#21-why-descriptors)
- [2.2 Chronological Overview](chapter-02-descriptors-featurization.md#22-chronological-overview)
- [2.3 Atom-Centered Symmetry Functions (ACSFs) — Behler & Parrinello (2007)](chapter-02-descriptors-featurization.md#23-atom-centered-symmetry-functions-acsfs-behler-parrinello-2007)
- [2.4 Coulomb Matrix — Rupp et al. (2012)](chapter-02-descriptors-featurization.md#24-coulomb-matrix-rupp-et-al-2012)
- [2.5 Smooth Overlap of Atomic Positions (SOAP) — Bartók et al. (2013)](chapter-02-descriptors-featurization.md#25-smooth-overlap-of-atomic-positions-soap-bartok-et-al-2013)
- [2.6 Many-Body Tensor Representation (MBTR) — Huo & Rupp (2022)](chapter-02-descriptors-featurization.md#26-many-body-tensor-representation-mbtr-huo-rupp-2022)
- [2.7 Body Order](chapter-02-descriptors-featurization.md#27-body-order)
- [2.8 Atomic Cluster Expansion (ACE) — Drautz, Phys. Rev. B 99, 014104 (2019)](chapter-02-descriptors-featurization.md#28-atomic-cluster-expansion-ace-drautz-phys-rev-b-99-014104-2019)
- [2.9 Descriptor Properties Summary](chapter-02-descriptors-featurization.md#29-descriptor-properties-summary)

### [Chapter 3 — Dimensionality Reduction & Clustering](chapter-03-dimensionality-reduction-clustering.md)

- [3.1 Principal Component Analysis (PCA)](chapter-03-dimensionality-reduction-clustering.md#31-principal-component-analysis-pca)
- [3.2 Kernel PCA](chapter-03-dimensionality-reduction-clustering.md#32-kernel-pca)
- [3.3 Autoencoders (Nonlinear Dimensionality Reduction)](chapter-03-dimensionality-reduction-clustering.md#33-autoencoders-nonlinear-dimensionality-reduction)
- [3.4 t-SNE (t-distributed Stochastic Neighbor Embedding)](chapter-03-dimensionality-reduction-clustering.md#34-t-sne-t-distributed-stochastic-neighbor-embedding)
- [3.5 UMAP (Uniform Manifold Approximation and Projection)](chapter-03-dimensionality-reduction-clustering.md#35-umap-uniform-manifold-approximation-and-projection)
- [3.6 K-Means Clustering](chapter-03-dimensionality-reduction-clustering.md#36-k-means-clustering)
- [3.7 Hierarchical Clustering](chapter-03-dimensionality-reduction-clustering.md#37-hierarchical-clustering)
- [3.8 Density-Based Clustering: DBSCAN](chapter-03-dimensionality-reduction-clustering.md#38-density-based-clustering-dbscan)
- [3.9 HDBSCAN (Hierarchical DBSCAN)](chapter-03-dimensionality-reduction-clustering.md#39-hdbscan-hierarchical-dbscan)
- [3.10 Clustering Method Summary](chapter-03-dimensionality-reduction-clustering.md#310-clustering-method-summary)

### [Chapter 4 — Regression & Uncertainty Quantification](chapter-04-regression-uncertainty-quantification.md)

- [4.1 Multivariate Linear Regression](chapter-04-regression-uncertainty-quantification.md#41-multivariate-linear-regression)
- [4.2 The Kernel Trick](chapter-04-regression-uncertainty-quantification.md#42-the-kernel-trick)
- [4.3 Kernel Ridge Regression (KRR)](chapter-04-regression-uncertainty-quantification.md#43-kernel-ridge-regression-krr)
- [4.4 Gaussian Process Regression (GPR)](chapter-04-regression-uncertainty-quantification.md#44-gaussian-process-regression-gpr)
- [4.5 Model Selection, Cross-Validation & Ensemble Methods](chapter-04-regression-uncertainty-quantification.md#45-model-selection-cross-validation-ensemble-methods)
- [4.6 Kernel Methods for ML Interatomic Potentials](chapter-04-regression-uncertainty-quantification.md#46-kernel-methods-for-ml-interatomic-potentials)

### [Chapter 5 — Neural Networks, Trees & Regularization](chapter-05-neural-networks-trees-regularization.md)

- [5.1 Decision Stumps and Decision Trees](chapter-05-neural-networks-trees-regularization.md#51-decision-stumps-and-decision-trees)
- [5.2 Boosted Trees](chapter-05-neural-networks-trees-regularization.md#52-boosted-trees)
- [5.3 Bagging and Random Forests](chapter-05-neural-networks-trees-regularization.md#53-bagging-and-random-forests)
- [5.4 Feed-Forward Neural Networks](chapter-05-neural-networks-trees-regularization.md#54-feed-forward-neural-networks)
- [5.5 Activation Functions](chapter-05-neural-networks-trees-regularization.md#55-activation-functions)
- [5.6 Skip Connections and Residual Networks](chapter-05-neural-networks-trees-regularization.md#56-skip-connections-and-residual-networks)
- [5.7 Batch Normalization](chapter-05-neural-networks-trees-regularization.md#57-batch-normalization)
- [5.8 Optimization: Gradient Descent and Adam](chapter-05-neural-networks-trees-regularization.md#58-optimization-gradient-descent-and-adam)
- [5.9 Regularization](chapter-05-neural-networks-trees-regularization.md#59-regularization)

### [Chapter 6 — Graph Neural Networks (GNNs)](chapter-06-graph-neural-networks-gnns.md)

- [6.1 Classical vs. Deep ML: Representation Learning](chapter-06-graph-neural-networks-gnns.md#61-classical-vs-deep-ml-representation-learning)
- [6.2 Graphs: Structure and Notation](chapter-06-graph-neural-networks-gnns.md#62-graphs-structure-and-notation)
- [6.3 Types of Prediction Problems on Graphs](chapter-06-graph-neural-networks-gnns.md#63-types-of-prediction-problems-on-graphs)
- [6.4 Why Not Just Use an MLP on the Adjacency Matrix?](chapter-06-graph-neural-networks-gnns.md#64-why-not-just-use-an-mlp-on-the-adjacency-matrix)
- [6.5 Graph Convolutions: Message, Pool, Update](chapter-06-graph-neural-networks-gnns.md#65-graph-convolutions-message-pool-update)
- [6.6 Receptive Field and Oversmoothing](chapter-06-graph-neural-networks-gnns.md#66-receptive-field-and-oversmoothing)
- [6.7 Graph Convolutional Network (GCN) — Kipf & Welling (2017)](chapter-06-graph-neural-networks-gnns.md#67-graph-convolutional-network-gcn-kipf-welling-2017)
- [6.8 Including Edge Embeddings](chapter-06-graph-neural-networks-gnns.md#68-including-edge-embeddings)
- [6.9 Crystal Graph Convolutional Neural Networks (CGCNN)](chapter-06-graph-neural-networks-gnns.md#69-crystal-graph-convolutional-neural-networks-cgcnn)
- [6.10 Equivariant GNNs](chapter-06-graph-neural-networks-gnns.md#610-equivariant-gnns)

### [Chapter 7 — Machine Learning Interatomic Potentials I](chapter-07-machine-learning-interatomic-potentials-i.md)

- [7.1 The Many-Body Problem and the Schrödinger Equation](chapter-07-machine-learning-interatomic-potentials-i.md#71-the-many-body-problem-and-the-schrodinger-equation)
- [7.2 The Born–Oppenheimer Approximation and the Potential Energy Surface](chapter-07-machine-learning-interatomic-potentials-i.md#72-the-bornoppenheimer-approximation-and-the-potential-energy-surface)
- [7.3 Molecular Dynamics and the Need for MLIPs](chapter-07-machine-learning-interatomic-potentials-i.md#73-molecular-dynamics-and-the-need-for-mlips)
- [7.4 Empirical Force Fields](chapter-07-machine-learning-interatomic-potentials-i.md#74-empirical-force-fields)
- [7.5 MLIP Core Concepts](chapter-07-machine-learning-interatomic-potentials-i.md#75-mlip-core-concepts)
- [7.6 Behler–Parrinello Neural Network Potentials (2007)](chapter-07-machine-learning-interatomic-potentials-i.md#76-behlerparrinello-neural-network-potentials-2007)
- [7.9 Message-Passing MLIPs: SchNet (2018)](chapter-07-machine-learning-interatomic-potentials-i.md#79-message-passing-mlips-schnet-2018)
- [7.10 Euclidean Graph Neural Networks: NequIP (2021)](chapter-07-machine-learning-interatomic-potentials-i.md#710-euclidean-graph-neural-networks-nequip-2021)
- [7.11 MACE Architecture](chapter-07-machine-learning-interatomic-potentials-i.md#711-mace-architecture)

### [Chapter 8 — Machine Learning Interatomic Potentials II](chapter-08-machine-learning-interatomic-potentials-ii.md)

- [8.1 From Test Error to Production-Quality Potentials](chapter-08-machine-learning-interatomic-potentials-ii.md#81-from-test-error-to-production-quality-potentials)
- [8.2 Dynamic Observables and Time-Correlation Functions](chapter-08-machine-learning-interatomic-potentials-ii.md#82-dynamic-observables-and-time-correlation-functions)
- [8.3 Training Data Strategies](chapter-08-machine-learning-interatomic-potentials-ii.md#83-training-data-strategies)
- [8.4 Speed–Accuracy Benchmarks](chapter-08-machine-learning-interatomic-potentials-ii.md#84-speedaccuracy-benchmarks)
- [8.6 Foundation Models](chapter-08-machine-learning-interatomic-potentials-ii.md#86-foundation-models)
- [8.7 Transfer Learning and Fine-Tuning](chapter-08-machine-learning-interatomic-potentials-ii.md#87-transfer-learning-and-fine-tuning)
- [8.8 Long-Range Interactions](chapter-08-machine-learning-interatomic-potentials-ii.md#88-long-range-interactions)
- [8.9 Beyond MLIPs: Electronic Structure ML](chapter-08-machine-learning-interatomic-potentials-ii.md#89-beyond-mlips-electronic-structure-ml)

### [Chapter 9 — Generative Models](chapter-09-generative-models.md)

- [Problem Statement](chapter-09-generative-models.md#problem-statement)
- [Taxonomy of Generative Models](chapter-09-generative-models.md#taxonomy-of-generative-models)
- [Generative Adversarial Networks (GANs)](chapter-09-generative-models.md#generative-adversarial-networks-gans)
- [Variational Autoencoders (VAEs)](chapter-09-generative-models.md#variational-autoencoders-vaes)
- [Autoregressive Models](chapter-09-generative-models.md#autoregressive-models)
- [Large Language Models (LLMs)](chapter-09-generative-models.md#large-language-models-llms)
- [The Transformer Architecture (Vaswani et al. "Attention Is All You Need", 2017)](chapter-09-generative-models.md#the-transformer-architecture-vaswani-et-al-attention-is-all-you-need-2017)
- [The Problem of Sampling Physical Configurations](chapter-09-generative-models.md#the-problem-of-sampling-physical-configurations)
- [Normalizing Flows](chapter-09-generative-models.md#normalizing-flows)
- [Diffusion Models](chapter-09-generative-models.md#diffusion-models)
- [Denoising Diffusion Probabilistic Models (DDPMs)](chapter-09-generative-models.md#denoising-diffusion-probabilistic-models-ddpms)
- [Score-Based Diffusion Models](chapter-09-generative-models.md#score-based-diffusion-models)
- [Flow Matching](chapter-09-generative-models.md#flow-matching)

### [Chapter 10 — ML Methods for Rare Events Sampling](chapter-10-ml-methods-for-rare-events-sampling.md)

- [10.1 Rare Events: Motivation](chapter-10-ml-methods-for-rare-events-sampling.md#101-rare-events-motivation)
- [10.2 Bias Methods](chapter-10-ml-methods-for-rare-events-sampling.md#102-bias-methods)
- [10.3 Transition Path Sampling (TPS)](chapter-10-ml-methods-for-rare-events-sampling.md#103-transition-path-sampling-tps)
- [10.4 The Committor and the Backward Kolmogorov Equation](chapter-10-ml-methods-for-rare-events-sampling.md#104-the-committor-and-the-backward-kolmogorov-equation)
- [10.5 ML for the Committor and AIMMD](chapter-10-ml-methods-for-rare-events-sampling.md#105-ml-for-the-committor-and-aimmd)
