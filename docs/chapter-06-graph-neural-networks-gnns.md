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


<div id="mp-widget" style="background:#faf8f5;border:1px solid #e0dbd4;border-radius:8px;padding:1.1rem 1rem 0.9rem;margin:1.8rem 0;">
  <div style="display:flex;align-items:center;justify-content:center;gap:0.9rem;margin-bottom:0.65rem;flex-wrap:wrap;">
    <span style="font-size:0.83rem;font-weight:600;color:#555;">Message Passing &mdash; click any node to trace one step</span>
    <span style="display:inline-flex;border:1px solid #ccc;border-radius:5px;overflow:hidden;font-size:0.76rem;">
      <button id="mp-m-gen" onclick="mpMode('general')" style="padding:2px 11px;border:none;background:#f5f3f0;color:#555;cursor:pointer;">General</button>
      <button id="mp-m-ex"  onclick="mpMode('example')"  style="padding:2px 11px;border:none;background:#86BCBD;color:#fff;cursor:pointer;">Example</button>
    </span>
  </div>
  <div style="display:flex;gap:1rem;align-items:flex-start;flex-wrap:wrap;justify-content:center;">
    <svg id="mp-svg" viewBox="0 0 330 255" width="330" height="255" style="flex-shrink:0;"></svg>
    <div id="mp-panel" style="flex:1;min-width:210px;background:#fff;border:1px solid #e8e2da;border-radius:6px;padding:0.85rem;min-height:185px;font-size:0.81rem;line-height:1.68;color:#333;"></div>
  </div>
  <div style="text-align:center;margin-top:0.65rem;font-size:0.79rem;color:#555;">
    Pooling:&nbsp;
    <button id="mp-b-sum"  onclick="mpPool('sum')"  style="margin:0 3px;padding:3px 11px;border-radius:4px;border:1px solid #b5d4a0;background:#A4CE8B;cursor:pointer;font-size:0.79rem;">Sum</button>
    <button id="mp-b-mean" onclick="mpPool('mean')" style="margin:0 3px;padding:3px 11px;border-radius:4px;border:1px solid #ccc;background:#f5f3f0;cursor:pointer;font-size:0.79rem;">Mean</button>
    <button id="mp-b-max"  onclick="mpPool('max')"  style="margin:0 3px;padding:3px 11px;border-radius:4px;border:1px solid #ccc;background:#f5f3f0;cursor:pointer;font-size:0.79rem;">Max</button>
  </div>
</div>
<script>
(function(){
var NS='http://www.w3.org/2000/svg';
var ND=[{id:0,lbl:'A',x:165,y:32,v:3.0},{id:1,lbl:'B',x:52,y:108,v:1.5},
        {id:2,lbl:'C',x:278,y:108,v:2.8},{id:3,lbl:'D',x:52,y:202,v:4.0},
        {id:4,lbl:'E',x:278,y:202,v:0.5},{id:5,lbl:'F',x:165,y:218,v:2.2}];
var ED=[[0,1],[0,2],[1,2],[1,3],[2,4],[3,5],[4,5],[1,5]];
var ADJ={};ND.forEach(function(n){ADJ[n.id]=[];});
ED.forEach(function(e){ADJ[e[0]].push(e[1]);ADJ[e[1]].push(e[0]);});
var _pool='sum',_sel=null,_mode='example';

function ktx(s){try{return katex.renderToString(s,{throwOnError:false});}catch(e){return '<i style="font-size:0.88em;">'+s+'</i>';}}
function ktxd(s){try{return '<div style="margin:0.3rem 0 0.2rem;text-align:center;">'+katex.renderToString(s,{throwOnError:false,displayMode:true})+'</div>';}catch(e){return '<div style="text-align:center;font-style:italic;">'+s+'</div>';}}

function mk(tag,at,tx){var e=document.createElementNS(NS,tag);for(var k in at)e.setAttribute(k,at[k]);if(tx!==undefined)e.textContent=tx;return e;}
function draw(){
  var svg=document.getElementById('mp-svg');if(!svg)return;svg.innerHTML='';
  ED.forEach(function(e,i){var a=ND[e[0]],b=ND[e[1]];svg.appendChild(mk('line',{id:'mp-e'+i,x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:'#d0cbc4','stroke-width':'2'}));});
  ND.forEach(function(n){
    var g=document.createElementNS(NS,'g');g.setAttribute('style','cursor:pointer;');
    g.addEventListener('click',function(){mpSel(n.id);});
    g.appendChild(mk('circle',{id:'mp-c'+n.id,cx:n.x,cy:n.y,r:'22',fill:'#A4CE8B',stroke:'#fff','stroke-width':'2.5'}));
    var ly=_mode==='example'?n.y-5:n.y+4;
    g.appendChild(mk('text',{x:n.x,y:ly,'text-anchor':'middle','font-size':'11','font-weight':'bold',fill:'#333'},n.lbl));
    if(_mode==='example')g.appendChild(mk('text',{x:n.x,y:n.y+8,'text-anchor':'middle','font-size':'10',fill:'#555'},n.v.toFixed(1)));
    svg.appendChild(g);
  });
  setDefaultPanel();
}
function setDefaultPanel(){
  var p=document.getElementById('mp-panel');if(!p)return;
  if(_mode==='general'){
    p.innerHTML='<div style="color:#888;font-size:0.79rem;font-style:italic;margin-bottom:0.2rem;">Click any node <i>i</i> to see the update equations for that neighbourhood.</div>'+
    '<div style="font-size:0.79rem;color:#666;">Full layer update:</div>'+
    ktxd('v^{\\prime}_i = U_t\\!\\left(v_i,\\;\\bigoplus_{j\\in\\mathcal{N}(i)} M_t(v_i,\\,v_j,\\,e_{ij})\\right)')+
    '<div style="font-size:0.78rem;color:#666;line-height:1.7;margin-top:0.1rem;">'+
    ktx('M_t')+': message function &mdash; what each neighbour sends<br>'+
    ktx('\\bigoplus')+': pooling (Sum / Mean / Max), permutation-invariant<br>'+
    ktx('U_t')+': update &mdash; e.g. '+ktx('\\sigma(\\mathbf{W}\\,m_i + \\mathbf{B}\\,v_i)')+'</div>';
  } else {
    p.innerHTML='<div style="color:#bbb;font-style:italic;font-size:0.8rem;">Click any node to trace the numerical computation.</div>';
  }
}

function mpSel(id){
  _sel=id;
  var nb=ADJ[id],ns=ND[id],lbl=ns.lbl;
  ND.forEach(function(n){var c=document.getElementById('mp-c'+n.id);c.setAttribute('fill','#A4CE8B');c.setAttribute('r','22');});
  ED.forEach(function(e,i){var l=document.getElementById('mp-e'+i);l.setAttribute('stroke','#d0cbc4');l.setAttribute('stroke-width','2');});
  document.getElementById('mp-c'+id).setAttribute('fill','#86BCBD');
  document.getElementById('mp-c'+id).setAttribute('r','25');
  nb.forEach(function(b){var c=document.getElementById('mp-c'+b);c.setAttribute('fill','#F7E49B');c.setAttribute('r','24');});
  ED.forEach(function(e,i){if(e[0]===id||e[1]===id){var l=document.getElementById('mp-e'+i);l.setAttribute('stroke','#86BCBD');l.setAttribute('stroke-width','3');}});

  var nbLbls=nb.map(function(b){return ND[b].lbl;});
  var hr='<hr style="margin:0.45rem 0;border:none;border-top:1px solid #eee;">';
  function hdr(bg,tc,txt){return '<div style="margin-bottom:0.3rem;"><span style="background:'+bg+';color:'+tc+';border-radius:3px;padding:1px 7px;font-size:0.78rem;font-weight:600;">'+txt+'</span></div>';}

  if(_mode==='general'){
    var nbSet = nb.length ? '\\{'+nbLbls.join(',\\,')+'\\}' : '\\emptyset';
    var s1f = ktxd('M_t(v_{'+lbl+'},\\,v_j,\\,e_{'+lbl+'j})\\quad \\text{for each } j \\in \\mathcal{N}('+lbl+')');
    var s2f;
    if(_pool==='sum')
      s2f=ktxd('m_{'+lbl+'} = \\sum_{j\\in '+nbSet+'} M_t(v_{'+lbl+'},\\,v_j,\\,e_{'+lbl+'j})');
    else if(_pool==='mean')
      s2f=ktxd('m_{'+lbl+'} = \\frac{1}{|\\mathcal{N}('+lbl+')|} \\sum_{j\\in '+nbSet+'} M_t(v_{'+lbl+'},\\,v_j,\\,e_{'+lbl+'j})');
    else
      s2f=ktxd('m_{'+lbl+'} = \\max_{j\\in '+nbSet+'} M_t(v_{'+lbl+'},\\,v_j,\\,e_{'+lbl+'j})');
    var s3f=ktxd('v^{\\prime}_{'+lbl+'} = U_t(v_{'+lbl+'},\\,m_{'+lbl+'}) = \\sigma(\\mathbf{W}\\,m_{'+lbl+'} + \\mathbf{B}\\,v_{'+lbl+'})');
    document.getElementById('mp-panel').innerHTML=
      hdr('#A4CE8B','#333','Step 1 — Message')+
      'Node '+ktx('\\mathbf{'+lbl+'}')+'&ensp;has '+ktx('|\\mathcal{N}('+lbl+')|')+' = '+nb.length+' neighbour'+(nb.length!==1?'s':'')+': '+ktx(nbSet.replace(/\\\\/g,'\\'))+
      s1f+hr+
      hdr('#F7E49B','#555','Step 2 — Pool ('+_pool+')')+s2f+hr+
      hdr('#86BCBD','#fff','Step 3 — Update')+s3f+
      '<div style="font-size:0.77rem;color:#aaa;margin-top:0.1rem;text-align:center;">'+ktx('\\mathbf{W}')+'&thinsp;and&thinsp;'+ktx('\\mathbf{B}')+'&ensp;are shared learned weight matrices</div>';
  } else {
    var vals=nb.map(function(b){return ND[b].v;}),msg=0;
    if(_pool==='sum')msg=vals.reduce(function(a,b){return a+b;},0);
    else if(_pool==='mean')msg=vals.length?vals.reduce(function(a,b){return a+b;},0)/vals.length:0;
    else msg=vals.length?Math.max.apply(null,vals):0;
    var upd=msg*0.6+ns.v*0.4;
    var vStr=vals.map(function(v){return v.toFixed(1);}).join(', ');
    var nbStr=nb.map(function(b){return '\\mathbf{'+ND[b].lbl+'}\\!=\\!'+ND[b].v.toFixed(1);}).join(',\\;')||'\\emptyset';
    var s2f2;
    if(_pool==='sum') s2f2=ktxd('m_{'+lbl+'} = '+vals.map(function(v){return v.toFixed(1);}).join(' + ')+' = \\mathbf{'+msg.toFixed(3)+'}');
    else if(_pool==='mean') s2f2=ktxd('m_{'+lbl+'} = \\tfrac{1}{'+nb.length+'}('+vals.map(function(v){return v.toFixed(1);}).join('+')+') = \\mathbf{'+msg.toFixed(3)+'}');
    else s2f2=ktxd('m_{'+lbl+'} = \\max('+vals.map(function(v){return v.toFixed(1);}).join(',\\,')+') = \\mathbf{'+msg.toFixed(3)+'}');
    document.getElementById('mp-panel').innerHTML=
      hdr('#A4CE8B','#333','Step 1 — Message')+
      'Node '+ktx('\\mathbf{'+lbl+'}')+'&ensp;('+ktx('v_{'+lbl+'}')+'&nbsp;=&nbsp;'+ns.v.toFixed(1)+') receives from: '+ktx(nbStr)+
      hr+hdr('#F7E49B','#555','Step 2 — Pool ('+_pool+')')+s2f2+
      hr+hdr('#86BCBD','#fff','Step 3 — Update')+
      ktxd('v^{\\prime}_{'+lbl+'} = \\sigma(\\mathbf{W}\\,m_{'+lbl+'} + \\mathbf{B}\\,v_{'+lbl+'}) \\approx 0.6 \\cdot '+msg.toFixed(2)+' + 0.4 \\cdot '+ns.v.toFixed(1)+' = \\mathbf{'+upd.toFixed(3)+'}')+
      '<div style="font-size:0.77rem;color:#aaa;text-align:center;">W=0.6, B=0.4 for illustration; real weights are learned</div>';
  }
}
window.mpPool=function(p){_pool=p;['sum','mean','max'].forEach(function(pp){var b=document.getElementById('mp-b-'+pp);if(b){b.style.background=pp===p?'#A4CE8B':'#f5f3f0';b.style.borderColor=pp===p?'#b5d4a0':'#ccc';}});if(_sel!==null)mpSel(_sel);};
window.mpMode=function(m){
  _mode=m;
  var gb=document.getElementById('mp-m-gen'),eb=document.getElementById('mp-m-ex');
  if(gb){gb.style.background=m==='general'?'#86BCBD':'#f5f3f0';gb.style.color=m==='general'?'#fff':'#555';}
  if(eb){eb.style.background=m==='example'?'#86BCBD':'#f5f3f0';eb.style.color=m==='example'?'#fff':'#555';}
  draw();if(_sel!==null)mpSel(_sel);
};
window.mpSel=mpSel;
function init(){if(!document.getElementById('mp-svg'))return;_sel=null;draw();}
if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',init);}else{init();}
if(typeof document$!=='undefined'){document$.subscribe(function(){setTimeout(init,50);});}
})();
</script>



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


<div id="cgcnn-widget" style="background:#faf8f5;border:1px solid #e0dbd4;border-radius:8px;padding:1.1rem 1rem 0.9rem;margin:1.8rem 0;">
  <div style="text-align:center;font-size:0.83rem;font-weight:600;color:#555;margin-bottom:0.3rem;">CGCNN: Bond Distance &rarr; Gaussian Basis Features</div>
  <div style="text-align:center;font-size:0.78rem;color:#888;margin-bottom:0.75rem;">Each edge encodes its distance as a 16-dimensional vector. Move the slider to change the bond distance.</div>
  <div style="display:flex;gap:1.2rem;align-items:flex-start;flex-wrap:wrap;justify-content:center;">
    <div style="text-align:center;">
      <div style="font-size:0.75rem;color:#999;margin-bottom:3px;">g<sub>k</sub>(d) = exp(&minus;(d&minus;&mu;<sub>k</sub>)&sup2;&thinsp;/&thinsp;2&sigma;&sup2;)&ensp;&nbsp;&sigma;=0.5&#8239;&Aring;</div>
      <svg id="cgcnn-plot" viewBox="0 0 320 165" width="320" height="165"></svg>
      <div style="margin-top:0.45rem;font-size:0.8rem;color:#555;">
        d&nbsp;=&nbsp;<input type="range" id="cgcnn-sl" min="10" max="79" value="28" oninput="cgcnnUp()" style="width:170px;vertical-align:middle;">&ensp;<span id="cgcnn-dv" style="font-weight:600;color:#86BCBD;min-width:3.5em;display:inline-block;"></span>
      </div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:0.75rem;color:#999;margin-bottom:3px;">Feature vector&ensp;<span style="color:#bbb;">(activations at d)</span></div>
      <svg id="cgcnn-bars" viewBox="0 0 210 165" width="210" height="165"></svg>
    </div>
  </div>
</div>
<script>
(function(){
var NS='http://www.w3.org/2000/svg';
var NK=16,SIG=0.5;
var MU=[];for(var k=0;k<NK;k++)MU.push(0.5+(k*0.5));
function gk(mu,d){return Math.exp(-(d-mu)*(d-mu)/(2*SIG*SIG));}
function mk(tag,at,tx){var e=document.createElementNS(NS,tag);for(var k in at)e.setAttribute(k,at[k]);if(tx!==undefined)e.textContent=tx;return e;}

function drawPlot(d){
  var svg=document.getElementById('cgcnn-plot');if(!svg)return;svg.innerHTML='';
  var W=320,H=165,PL=28,PR=8,PT=10,PB=26,pw=W-PL-PR,ph=H-PT-PB,dmax=8.5;
  svg.appendChild(mk('rect',{x:PL,y:PT,width:pw,height:ph,fill:'#fff',stroke:'#e0dbd4','stroke-width':'0.8'}));
  for(var xi=1;xi<=8;xi++){var px=PL+pw*xi/dmax;svg.appendChild(mk('line',{x1:px,y1:PT+ph,x2:px,y2:PT+ph+4,stroke:'#ddd','stroke-width':'0.8'}));svg.appendChild(mk('text',{x:px,y:PT+ph+13,'text-anchor':'middle','font-size':'8',fill:'#aaa'},xi.toString()));}
  svg.appendChild(mk('text',{x:PL+pw/2,y:H,'text-anchor':'middle','font-size':'8',fill:'#aaa'},'distance (Å)'));
  svg.appendChild(mk('text',{x:5,y:PT+ph/2+3,'text-anchor':'middle','font-size':'8',fill:'#aaa','transform':'rotate(-90,5,'+(PT+ph/2)+')'},'g(d)'));
  for(var ki=0;ki<NK;ki++){
    var pts=[];
    for(var xi2=0;xi2<=200;xi2++){var di=0.0+xi2*(dmax/200);var gi=gk(MU[ki],di);pts.push((PL+pw*di/dmax)+','+(PT+ph*(1-gi)));}
    var act=gk(MU[ki],d);var col=act>0.05?'rgba(134,188,189,'+Math.min(1,act*2).toFixed(2)+')':'rgba(208,203,196,0.35)';
    svg.appendChild(mk('polyline',{points:pts.join(' '),fill:'none',stroke:col,'stroke-width':act>0.05?'1.8':'0.9'}));
  }
  var lx=PL+pw*d/dmax;
  svg.appendChild(mk('line',{x1:lx,y1:PT,x2:lx,y2:PT+ph,stroke:'#BA5A5A','stroke-width':'1.5','stroke-dasharray':'4,3'}));
  svg.appendChild(mk('text',{x:Math.min(lx+3,W-30),y:PT+9,'font-size':'8',fill:'#BA5A5A'},d.toFixed(1)+'Å'));
}

function drawBars(d){
  var svg=document.getElementById('cgcnn-bars');if(!svg)return;svg.innerHTML='';
  var W=210,H=165,PL=8,PR=8,PT=10,PB=26,pw=W-PL-PR,ph=H-PT-PB;
  svg.appendChild(mk('rect',{x:PL,y:PT,width:pw,height:ph,fill:'#fff',stroke:'#e0dbd4','stroke-width':'0.8'}));
  var bw=Math.floor(pw/NK)-2;
  for(var ki=0;ki<NK;ki++){
    var v=gk(MU[ki],d);
    var bh=Math.max(1,v*ph);
    var bx=PL+ki*(pw/NK)+1;
    var by=PT+ph-bh;
    var col=v>0.05?'#86BCBD':'#d5eaeb';
    svg.appendChild(mk('rect',{x:bx,y:by,width:bw,height:bh,fill:col,rx:'1.5'}));
    if(ki%4===0)svg.appendChild(mk('text',{x:bx+bw/2,y:PT+ph+10,'text-anchor':'middle','font-size':'7.5',fill:'#bbb'},(ki+1).toString()));
  }
  svg.appendChild(mk('text',{x:PL+pw/2,y:H,'text-anchor':'middle','font-size':'8',fill:'#aaa'},'basis index'));
}

window.cgcnnUp=function(){
  var sl=document.getElementById('cgcnn-sl');if(!sl)return;
  var d=parseFloat(sl.value)/10;
  document.getElementById('cgcnn-dv').textContent=d.toFixed(1)+'Å';
  drawPlot(d);drawBars(d);
};
function init(){if(!document.getElementById('cgcnn-plot'))return;cgcnnUp();}
if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',init);}else{init();}
if(typeof document$!=='undefined'){document$.subscribe(function(){setTimeout(init,50);});}
})();
</script>

## 6.10 Equivariant GNNs

Scalar properties like total energy must be **invariant** under rotation, reflection, and permutation of equivalent atoms: the energy of a molecule does not change if you rotate the coordinate frame. Standard GNNs using scalar node and edge features satisfy this automatically, as long as edge embeddings encode distances (invariants) rather than displacement vectors.

Vector and tensor properties such as atomic forces and stress tensors must instead be **equivariant**: if the atomic positions are rotated by $R$, the forces must rotate by $R$ as well. Formally, for a group element $g$ with representation $D(g)$:

$$f(D(g)\mathbf{x},\, \omega) = D(g)\, f(\mathbf{x},\, \omega).$$

Building equivariance into the architecture (rather than approximating it via data augmentation) is dramatically more data-efficient: a single training example in one orientation teaches the model about all orientations simultaneously.

Equivariant GNNs use **spherical harmonic** basis functions $Y_l^m(\hat{\mathbf{r}}_{ij})$, where $\hat{\mathbf{r}}_{ij}$ is the unit vector along edge $(i,j)$ and $l = 0, 1, 2, \ldots$ is the angular momentum order. Scalar ($l=0$), vector ($l=1$), and higher-order tensor ($l \geq 2$) channels are maintained and mixed via tensor products that preserve the equivariance. The **e3nn** library implements this algebra. Key equivariant GNN models include NequIP (Batzner et al., *Nat. Commun.* **13**, 2453, 2022; equivariant interatomic potential) and MACE (Batatia et al., *NeurIPS*, 2022; many-body equivariant features, currently among the most accurate and efficient MLIPs available).

---

