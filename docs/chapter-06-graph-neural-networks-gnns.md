# Chapter 6: Graph Neural Networks (GNNs)

## 6.1 Classical vs. Deep ML: Representation Learning

Classical ML operates on hand-crafted features: given an input $x$ that has already been carefully engineered to encode domain knowledge, a model learns $f(x) = y$. Deep learning is qualitatively different. Instead of requiring a human to specify what the features should be, it learns to construct useful representations from raw or minimally processed data: $f(f(\Theta)) = y$, where the inner $f(\Theta)$ is the learned representation. This is called **representation learning**.

The practical consequence is that deep learning generally requires more data than classical ML: the model must learn both the relevant features and the mapping from features to labels, but it can outperform classical approaches when enough data is available, because the learned representations can capture patterns that hand-crafted features miss. MLPs (also called FFNNs) are composed of stacked **dense layers**, each fully connected to the previous layer. The term "perceptron" refers to a single unit $y = g(\mathbf{w}^T\mathbf{x} + b)$; an MLP is a multilayer stack of such units.

## 6.2 Graphs: Structure and Notation

A **graph** $G = (V, E)$ consists of a set of nodes (also called vertices) $V$ and a set of edges $E$ connecting pairs of nodes. Edges may be undirected or directed. Each node $i \in V$ carries a feature vector $\mathbf{v}_i$ called a **node embedding**, and each edge $(i,j) \in E$ carries a feature vector $\mathbf{e}_{ij}$ called an **edge embedding**. The neighborhood of node $i$ is $\mathcal{N}(i) = \{j : (i,j) \in E\}$.

Graphs are a remarkably general data structure. Images are graphs with a regular grid structure: each pixel is a node connected to its spatial neighbors. A sentence is a directed graph where each word is a node connected to the next. Social networks, biological signaling networks, and financial networks are all naturally represented as graphs. Most relevantly for computational physics, **molecules and crystals are natural graphs**: atoms are nodes (with element type and other properties as features) and bonds or proximity-based connections are edges (with bond distance and direction as features).

Graphs vary along several axes worth knowing. An **undirected** graph has symmetric edges ($(i,j) \in E \Leftrightarrow (j,i) \in E$); a **directed** graph allows one-way connections. Edges may carry scalar **weights** alongside feature vectors. A **multigraph** permits multiple distinct edges between the same pair of nodes: for periodic crystals, the same atom pair can be connected by edges to several different periodic images. **Heterogeneous** graphs have multiple node or edge types (e.g. different chemical species). Finally, **self-loops** $(i,i)$ are sometimes added explicitly so that a node's own embedding contributes to its own updated representation, which is done in the GCN formulation below.

## 6.3 Types of Prediction Problems on Graphs

Three classes of prediction task arise on graphs:

**Graph-level prediction**: a single output for the entire graph, for example, the total energy of a molecule or a material property of a crystal. Requires aggregating information across all nodes into a single global representation.

**Node-level prediction**: an output per node, for example, the oxidation state of each atom, or the charge on each residue in a protein. AlphaFold, which predicts protein 3D structure, uses a node-level GNN where amino acids are nodes linked by sequence and spatial edges.

**Edge-level prediction**: an output per edge, for example, the strength of a bond between two atoms, or whether an interaction exists between two proteins.

## 6.4 Why Not Just Use an MLP on the Adjacency Matrix?

A naive approach to deep learning with graphs is to flatten the adjacency matrix and node features into a vector and feed it to a standard MLP. This fails for two reasons. First, the adjacency matrix has a **fixed size**: a network trained on 5-node graphs cannot accept a 10-node graph without retraining. Second, the result is **sensitive to node ordering**: permuting the node labels produces a different input vector, even though the graph is the same. Any valid model for graphs must be invariant to these relabellings.

## 6.5 Graph Convolutions: Message, Pool, Update

The key insight is to generalize convolutions from regular grids to arbitrary graphs. In a convolutional neural network (CNN) for images, each layer applies a learned filter that combines information from spatially neighboring pixels. On a graph, the analogue is to have each node gather information from its graph neighbors. A single **graph convolution** proceeds in three steps:

**1. Prepare messages.** For each neighbor $j \in \mathcal{N}(i)$, compute a message $M_t(\mathbf{v}_i, \mathbf{v}_j, \mathbf{e}_{ij})$ that encodes the information node $j$ sends to node $i$ at layer $t$. The message function can use the embeddings of both endpoint nodes and the edge between them.

**2. Pool messages.** Aggregate all incoming messages into a single summary vector for node $i$:

$$\mathbf{m}_i = \bigoplus_{j \in \mathcal{N}(i)} M_t(\mathbf{v}_i, \mathbf{v}_j, \mathbf{e}_{ij}).$$

The pooling operator $\bigoplus$ must be **permutation invariant** (the result must not depend on the order in which neighbors are processed) and must handle a variable number of inputs. The three standard choices are Max, Mean, and Sum, and all satisfy both requirements.

**3. Update embedding.** Produce the new node embedding by combining the pooled message with the current embedding:

$$\mathbf{v}_i' = U_t(\mathbf{v}_i, \mathbf{m}_i).$$

Applying this procedure simultaneously to all nodes, then stacking $T$ such layers, constitutes a **graph neural network (GNN)**.

For **graph-level prediction**, a *readout* (or *global pooling*) step collapses all node embeddings into a single fixed-size vector after the final message-passing layer:

$$\mathbf{u} = \text{POOL}\bigl(\{\mathbf{v}_i^{(T)} : i \in V\}\bigr),$$

and a final MLP maps $\mathbf{u}$ to the predicted property. The three standard choices are sum, mean, and max pooling. Sum pooling is sensitive to graph size (larger graphs produce larger sums), making it appropriate when the property scales with the number of atoms (e.g. total energy). Mean pooling is size-invariant and works well for intensive properties. Max pooling captures the presence of the most extreme feature but discards count information.

**Weight sharing** is the defining property of convolutional architectures, and it is worth contrasting with a plain FFNN. In an FFNN, every connection between two units has its own individual weight: if layer $k-1$ has $n$ units and layer $k$ has $m$ units, there are $n \times m$ independent weights. In a GCN, the matrices $\mathbf{W}$ and $\mathbf{B}$ do not belong to any specific node; they define the operation of aggregating neighbours, and the exact same $\mathbf{W}$ and $\mathbf{B}$ are applied at every node. Node 3 and node 47 both use the same weights; their updated embeddings differ only because their neighbourhoods differ. This is directly analogous to image CNNs, where the same $3 \times 3$ filter is slid across every pixel location rather than assigning a separate filter to each pixel.

Weight sharing has two important consequences. First, the number of parameters does not grow with the number of nodes: a GCN on a 5-node graph and a 500-node graph has exactly the same number of trainable parameters. Second, a model trained on small graphs can be evaluated on larger graphs, because the weights encode a local operation rather than anything specific to a particular node's index or position. This is called **inductive** generalization.

## 6.6 Receptive Field and Oversmoothing

After $t$ message-passing layers, node $i$ has received information from all nodes within $t$ hops. This is called the **receptive field** of node $i$ after $t$ layers. A single layer sees only immediate neighbors; two layers see neighbors-of-neighbors, and so on. Stacking more layers increases the receptive field, allowing the network to capture longer-range structural patterns.

However, using too many layers causes **oversmoothing**: repeated averaging pushes all node embeddings toward the same value, erasing the local structural distinctions the network needs to make useful predictions. In practice, 2–4 message-passing layers are used for most molecular and materials applications.

## 6.7 Graph Convolutional Network (GCN)

The first widely adopted GNN architecture, the Graph Convolutional Network (GCN) (Kipf & Welling, *ICLR*, 2017), uses the simplest possible choices for each component:

- **Message function**: pass the neighbor embedding unchanged, $M_t(\mathbf{v}_i, \mathbf{v}_j) = \mathbf{v}_j$ (no processing).
- **Pooling function**: normalized mean, $\mathbf{m}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{v}_j / |\mathcal{N}(i)|$.
- **Update function**: a single affine transformation followed by a nonlinearity, $\mathbf{v}_i' = \sigma(\mathbf{W}\mathbf{m}_i + \mathbf{B}\mathbf{v}_i)$, where $\mathbf{W}$ and $\mathbf{B}$ are weight matrices shared across all nodes.

Stacking $T$ such layers gives the full GCN update rule at layer $t$:

$$\mathbf{v}_i^{(t+1)} = \sigma\!\left(\mathbf{W}^{(t)} \sum_{j \in \mathcal{N}(i)} \frac{\mathbf{v}_j^{(t)}}{|\mathcal{N}(i)|} + \mathbf{B}^{(t)} \mathbf{v}_i^{(t)}\right),$$

where the weight matrices $\mathbf{W}^{(t)}$ and $\mathbf{B}^{(t)}$ are unique to each layer. The final node embeddings are fed to a loss function and an optimizer (e.g. Adam) trains all weight matrices end-to-end via backpropagation.

In the original paper, Kipf & Welling write the GCN update in compact matrix form. Let $\mathbf{H}^{(t)} \in \mathbb{R}^{|V| \times d}$ collect all node embeddings as rows. Adding self-loops defines $\hat{\mathbf{A}} = \mathbf{A} + \mathbf{I}$, and the diagonal degree matrix $\hat{\mathbf{D}}_{ii} = \sum_j \hat{A}_{ij}$ normalizes by node degree. The symmetrically normalized adjacency is

$$\tilde{\mathbf{A}} = \hat{\mathbf{D}}^{-1/2}\,\hat{\mathbf{A}}\,\hat{\mathbf{D}}^{-1/2},$$

and the full-layer update becomes

$$\mathbf{H}^{(t+1)} = \sigma\!\left(\tilde{\mathbf{A}}\,\mathbf{H}^{(t)}\,\mathbf{W}^{(t)}\right).$$

The self-loop in $\hat{\mathbf{A}}$ means node $i$ also aggregates its own embedding, equivalent to including $i \in \mathcal{N}(i)$. The symmetric normalization $\hat{D}_i^{-1/2}\hat{A}_{ij}\hat{D}_j^{-1/2}$ weights the contribution from node $j$ to node $i$ by $1/\sqrt{d_i d_j}$ (where $d_i, d_j$ are their degrees), preventing high-degree hub nodes from dominating aggregation. This matrix form makes the connection to spectral graph convolutions explicit and is efficient to compute via sparse matrix multiplication.

## 6.8 Including Edge Embeddings

The GCN ignores edge features. For physical systems where bond distances, bond types, and directions matter, the message function is extended to include the edge embedding $\mathbf{e}_{ij}$:

$$\mathbf{m}_i = \bigoplus_{j \in \mathcal{N}(i)} M_t(\mathbf{v}_i, \mathbf{v}_j, \mathbf{e}_{ij}).$$

The update function $U_t(\mathbf{v}_i, \mathbf{m}_i)$ stays the same. Message-passing networks offer considerable design flexibility: the update order (nodes before edges, or interleaved in a "weave" pattern that passes messages back and forth), the pooling strategy, and the dimensionality of node and edge embeddings are all choices. The message, pool, and update functions themselves are typically parameterized as MLPs.

## 6.9 Crystal Graph Convolutional Neural Networks (CGCNN)

CGCNN (Xie & Grossman, *Phys. Rev. Lett.* **120**, 145301, 2018) was the first application of graph convolutions to crystalline materials.

**Graph construction.** Each atom in the unit cell is a node. Two atoms are connected by an edge if their distance is below a cutoff radius $r_\text{cut}$. Periodic boundary conditions (PBCs) are respected: the procedure is applied for each atom in the unit cell, and the same pair of atoms may share multiple edges (corresponding to different periodic images of the neighbor), forming a multigraph.

**Initialization.** Node embeddings are initialized by one-hot encoding the element type (dimension 119, one entry per element in the periodic table) and passing through an MLP. Edge embeddings are initialized by expanding the bond distance onto 40 Gaussian basis functions.

**Message function.** The message for edge $(i,j)$ at layer $t$ is the concatenation of both node embeddings and the edge embedding:

$$\mathbf{m}_i^{(t)} = \mathbf{v}_i^{(t)} \oplus \mathbf{v}_j^{(t)} \oplus \mathbf{e}_{ij},$$

where $\oplus$ denotes concatenation.

**Update function.** CGCNN uses a gated, additive update. The message $\mathbf{m}_{ij}^{(t)}$ is passed through two independent linear transforms in parallel:

$$\mathbf{v}_i^{(t+1)} = \underbrace{\mathbf{v}_i^{(t)}}_{\text{skip}} + \sum_{j \in \mathcal{N}(i)} \underbrace{\sigma\!\left(\mathbf{W}_f^{(t)} \mathbf{m}_{ij}^{(t)} + \mathbf{b}_f^{(t)}\right)}_{\text{gate} \in (0,1)} \odot \underbrace{g\!\left(\mathbf{W}_s^{(t)} \mathbf{m}_{ij}^{(t)} + \mathbf{b}_s^{(t)}\right)}_{\text{value}},$$

where $\sigma$ is the sigmoid (gate branch, output in $(0,1)$), $g$ is the softplus (value branch), and $\odot$ is elementwise multiplication. The gate learns per-channel which parts of each neighbor's message are informative and suppresses the rest, analogous to attention or LSTM gating. The additive form $\mathbf{v}_i^{(t)} + [\cdots]$ means the update is incremental: the node accumulates information from neighbors on top of what it already knows, rather than replacing its embedding entirely. The CGCNN paper refers to this as a residual connection, though the primary motivation is simply to prevent nodes from forgetting their own state across message-passing layers.

**Global pooling.** After $T$ message-passing layers, the atom embeddings are collapsed to a single graph-level vector by mean pooling:

$$\mathbf{u}_c = \frac{1}{|G|}\sum_{i \in G} \mathbf{v}_i^{(T)}.$$

**Readout.** A final MLP maps the pooled representation to the predicted property:

$$E = \sigma(\mathbf{W}_r \mathbf{u}_c + \mathbf{b}_r).$$

CGCNN demonstrated good accuracy for crystal property prediction (formation energy MAE $\approx 0.039$ eV/atom) but errors remain too large for reliable quantitative science, motivating more sophisticated architectures such as MEGNet (Chen et al., *Chem. Mater.* **31**, 3564, 2019; adds skip connections and set2set pooling) and M3GNet (Chen & Ong, *Nat. Comput. Sci.* **2**, 718, 2022; includes bond angles and dihedral angles in the message function).

## 6.10 Equivariant GNNs

Scalar properties like total energy must be **invariant** under rotation, reflection, and permutation of equivalent atoms: the energy of a molecule does not change if you rotate the coordinate frame. Standard GNNs using scalar node and edge features satisfy this automatically, as long as edge embeddings encode distances (invariants) rather than displacement vectors.

Vector and tensor properties such as atomic forces and stress tensors must instead be **equivariant**: if the atomic positions are rotated by $R$, the forces must rotate by $R$ as well. Formally, for a group element $g$ with representation $D(g)$:

$$f(D(g)\mathbf{x},\, \omega) = D(g)\, f(\mathbf{x},\, \omega).$$

Building equivariance into the architecture (rather than approximating it via data augmentation) is dramatically more data-efficient: a single training example in one orientation teaches the model about all orientations simultaneously.

Equivariant GNNs use **spherical harmonic** basis functions $Y_l^m(\hat{\mathbf{r}}_{ij})$, where $\hat{\mathbf{r}}_{ij}$ is the unit vector along edge $(i,j)$ and $l = 0, 1, 2, \ldots$ is the angular momentum order. Scalar ($l=0$), vector ($l=1$), and higher-order tensor ($l \geq 2$) channels are maintained and mixed via tensor products that preserve the equivariance. The **e3nn** library implements this algebra. Key equivariant GNN models include NequIP (Batzner et al., *Nat. Commun.* **13**, 2453, 2022; equivariant interatomic potential) and MACE (Batatia et al., *NeurIPS*, 2022; many-body equivariant features, currently among the most accurate and efficient MLIPs available).

---

