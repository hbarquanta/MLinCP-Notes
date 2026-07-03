# Chapter 2 — Descriptors & Featurization

## 2.1 Why Descriptors?

In atomistic machine learning, the inputs $x^{(p)}$ describe atomic structures: a set of $N$ atoms with nuclear charges $\{Z_i\}_{i=1}^N$ and positions $\{\mathbf{R}_i\}_{i=1}^N$. Raw Cartesian coordinates cannot be used directly as ML inputs because the learned function must respect the physical symmetries of the problem. Three terms are worth distinguishing from the outset.

A **structure** is the full specification of an atomic configuration: nuclear charges $\{Z_i\}$, positions $\{\mathbf{R}_i\}$, and the unit cell for periodic systems. A **descriptor** is the algorithm $\phi$ that transforms a structure into a fixed-length numerical vector with the correct symmetry properties. The **feature vector** $\mathbf{x} = \phi(\{Z_i, \mathbf{R}_i\})$ is the actual numerical output that lives in feature space and is passed to the ML model. Descriptors can be **global** — encoding the entire molecule or unit cell as one vector (Coulomb matrix, MBTR) — or **atom-wise** — producing one vector per atom encoding the local chemical environment within a cutoff radius (ACSFs, SOAP, ACE).

### Requirements on descriptors

A valid descriptor $\mathbf{x} = \phi(\{Z_i, \mathbf{R}_i\})$ must satisfy:

1. **Translational invariance**: $\phi$ is unchanged when all positions are shifted by a constant vector $\mathbf{t}$, i.e., $\phi(\{\mathbf{R}_i + \mathbf{t}\}) = \phi(\{\mathbf{R}_i\})$.
2. **Rotational invariance**: $\phi$ is unchanged under any rotation $\mathbf{R} \in SO(3)$ applied to all positions.
3. **Permutation invariance**: $\phi$ is unchanged when identical atoms are relabeled.
4. **Completeness and uniqueness**: the mapping must be **surjective** (every structure maps to some feature vector) *and* **injective** (symmetry-inequivalent structures map to *different* feature vectors). Only when both hold is $\phi$ bijective — a fully invertible encoding. In practice, complete bijective mappings for arbitrary atom configurations are not achievable, so descriptors trade coverage for computational tractability.
5. **Smoothness**: $\phi$ must be smooth and differentiable in atom positions so that forces $F_i = -\partial E / \partial \mathbf{R}_i$ can be computed analytically.

### Invariance, equivariance, and where to put the symmetry

For scalar properties (energy, band gap), the model output must be **invariant** — unchanged under any symmetry operation $\mathcal{R}$:

$$\hat{f}(\mathcal{R}\,\mathbf{X}) = \hat{f}(\mathbf{X})$$

For vectorial properties (forces, dipole moments) or tensorial properties (polarizability, stress), the output must be **equivariant** — it must transform predictably with the symmetry operation:

$$\hat{f}(\mathcal{R}\,\mathbf{X}) = \mathcal{R}\,[\hat{f}(\mathbf{X})]$$

A force vector, for example, must rotate exactly as the molecule rotates. Symmetry can be enforced in three ways: building it into the **descriptor** (the traditional approach for ACSFs, CM, MBTR, SOAP), building it into the **model architecture**, or designing **equivariant models** whose outputs transform correctly under all symmetry operations — the approach of equivariant graph neural networks (Chapter 6), which can directly predict vectorial and tensorial quantities. The choice shapes the inductive bias of the entire ML pipeline.

---

## 2.2 Two-Dimensional Molecular Representations

Before the 3D geometry-aware descriptors covered below, it is useful to note a class of representations based purely on molecular *topology* — the 2D graph of atoms and bonds, without 3D conformation. These are standard in cheminformatics and drug discovery, where 3D structure is often unknown or unnecessary.

### SMILES

**SMILES** (Simplified Molecular Input Line Entry System) encodes a molecule as an ASCII string. Atoms are written by their chemical symbol; bonds are implicit (single) or explicit (`-`, `=`, `#` for single, double, triple); rings are indicated by matching numbers on the opening and closing atoms; branches use parentheses; aromatic atoms are written in lowercase. Example — benzene: `c1ccccc1`. Example — 4-ethylheptane: `CCCC(CC)CCC`.

SMILES is compact and human-readable and defines a composable language for molecules. Its limitations are that it carries no 3D conformation, cannot easily encode radicals or unusual valences, and is not well suited for generative models (small syntax violations produce invalid strings). Variants such as **DeepSMILES** and **SELFIES** (Self-Referencing Embedded Strings) address the validity problem by construction: every string in their grammar decodes to a valid molecule.

### Extended Connectivity Fingerprints (ECFP)

**ECFP** (also known as Morgan or circular fingerprints) represent local chemical environments of each atom at multiple radii and encode the result as a fixed-length bit vector. The algorithm has four stages:

1. **Initial assignment**: each atom receives an integer identifier encoding its local environment (atomic number, degree, formal charge, etc.).
2. **Iterative updating (Morgan algorithm)**: each atom's identifier is updated by combining it with the identifiers of its immediate neighbors, growing the local sphere by one bond per iteration; this is repeated for a fixed number of iterations (the fingerprint radius).
3. **Duplicate removal**: identifiers that appear multiple times are deduplicated.
4. **Bit array formation**: all identifiers are hashed into positions in a fixed-size bit vector (typically 1024 or 2048 bits).

Each active bit in the ECFP vector corresponds to a recognizable chemical substructure, making the fingerprint interpretable. The reference implementation is **RDKit**, an open-source cheminformatics toolkit that also provides hundreds of additional 2D descriptors (logP, TPSA, ring counts, and more).

A fundamental limitation of all 2D descriptors is the absence of *atomic-resolution 3D geometry*: two conformers of the same molecule produce the same SMILES string and ECFP fingerprint. They do not satisfy rotational or translational invariance in the 3D sense and cannot be used for force or energy prediction from atomistic positions.

---

## 2.3 Coulomb Matrix

The Coulomb matrix $M \in \mathbb{R}^{N_\text{at}\times N_\text{at}}$ encodes the electrostatic interactions among all $N_\text{at}$ atoms in a molecule:

$$M_{ij} = \begin{cases} \dfrac{1}{2} Z_i^{2.4} & i = j \\[6pt] \dfrac{Z_i Z_j}{|\mathbf{R}_i - \mathbf{R}_j|} & i \neq j \end{cases}$$

where $Z_i$ is the nuclear charge (atomic number) of atom $i$ and $|\mathbf{R}_i - \mathbf{R}_j|$ is the Euclidean distance between atoms $i$ and $j$. The diagonal element $\frac{1}{2}Z_i^{2.4}$ approximates the free-atom potential energy (Thomas-Fermi model). The off-diagonal element is the classical nuclear repulsion energy.

The Coulomb matrix is a **global** descriptor — it encodes the entire molecule, not the local environment of individual atoms — which makes it size-dependent and inapplicable to periodic systems (it would require padding with zeros). It is not permutation invariant by construction: reordering atoms changes $M$. Common remedies are sorting rows and columns by their $\ell^2$ norm, or using the sorted eigenvalue spectrum of $M$, which is permutation invariant but loses some structural information.

The Coulomb matrix encodes only pairwise distances (2-body) and misses angular information entirely.

---


## 2.4 Many-Body Tensor Representation (MBTR)

The MBTR is a **global** descriptor that encodes the full structure of a molecule by representing many-body interactions as smooth one-dimensional distributions over geometric quantities. Each geometric quantity (e.g. an inverse distance or bond angle) contributes a Gaussian peak weighted by the nuclear charges of the involved atoms, giving a differentiable, continuous fingerprint that is permutation invariant by construction.

Three interaction levels are defined. For each level $k$, the descriptor is the distribution:

$$D(x,\, g_k) = \sum_\text{atom tuples} w_k \cdot \mathcal{N}\!\left(x;\; g_k(i,j,\ldots),\; \sigma^2\right)$$

where $g_k(i,j,\ldots)$ is the geometric function evaluated for the atom tuple, $w_k$ is a nuclear-charge weighting factor, $\sigma$ is a broadening width (hyperparameter), and $\mathcal{N}(x;\mu,\sigma^2)$ denotes a Gaussian with mean $\mu$ and variance $\sigma^2$.

**$k=1$ (1-body)**: $g_1(i) = Z_i$ (nuclear charge); encodes composition.

**$k=2$ (2-body)**: $g_2(i,j) = 1/r_{ij}$ (inverse distance), $w_k = Z_i Z_j$; encodes the distribution of bond distances.

**$k=3$ (3-body)**: $g_3(i,j,k) = \cos\theta_{ijk}$ (cosine of the angle at atom $j$ between atoms $i$ and $k$), $w_k = Z_i Z_j Z_k$; encodes angular structure.

The full MBTR vector is the concatenation of $D(x, g_1)$, $D(x, g_2)$, $D(x, g_3)$ evaluated on fine grids. The 3-body term costs $\mathcal{O}(N_\text{at}^3)$ in the number of atoms. Because the descriptor sums over all atoms globally, it is not directly applicable to large periodic systems or for computing atom-wise (local) energies.

### Interactive Explorer

<div id="mbtr-widget" style="border:1px solid #8884;border-radius:8px;padding:1.2rem;margin:1.4rem 0;background:var(--md-code-bg-color,#f5f5f5);">

<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:1rem;flex-wrap:wrap;">
  <span style="font-weight:600;font-size:0.95rem;color:var(--md-default-fg-color);">MBTR Explorer</span>
  <div style="display:flex;gap:5px;">
    <button id="mbtr-btn-h2o" onclick="mbtrSetMol('h2o')" style="padding:3px 12px;border-radius:20px;border:1.5px solid #86BCBD;background:#86BCBD;color:#fff;font-size:0.8rem;cursor:pointer;font-weight:600;">H₂O</button>
    <button id="mbtr-btn-co2" onclick="mbtrSetMol('co2')" style="padding:3px 12px;border-radius:20px;border:1.5px solid #8886;background:transparent;color:var(--md-default-fg-color);font-size:0.8rem;cursor:pointer;">CO₂</button>
    <button id="mbtr-btn-nh3" onclick="mbtrSetMol('nh3')" style="padding:3px 12px;border-radius:20px;border:1.5px solid #8886;background:transparent;color:var(--md-default-fg-color);font-size:0.8rem;cursor:pointer;">NH₃*</button>
  </div>
  <span style="font-size:0.75rem;color:#999;">* NH₃ approximated as bent H₂X geometry</span>
</div>

<div style="display:flex;gap:1.5rem;flex-wrap:wrap;align-items:flex-start;">

  <!-- Molecule SVG + controls -->
  <div style="flex:0 0 auto;min-width:195px;max-width:220px;">
    <svg id="mbtr-svg" viewBox="0 0 220 185" width="220" height="185" style="display:block;margin:0 auto;">
      <!-- H–H or end-end dashed distance line -->
      <line id="mb-b3" stroke="#aaa" stroke-width="0.8" stroke-dasharray="4,3"/>
      <!-- bonds -->
      <line id="mb-b1" stroke="#777" stroke-width="3.5" stroke-linecap="round"/>
      <line id="mb-b2" stroke="#777" stroke-width="3.5" stroke-linecap="round"/>
      <!-- angle arc -->
      <path id="mb-arc" fill="none" stroke="#F7E49B" stroke-width="1.5"/>
      <!-- atoms (drawn last so they're on top) -->
      <circle id="mb-cn" r="16" stroke-width="1.5"/>
      <circle id="mb-e1" r="13" stroke-width="1.5"/>
      <circle id="mb-e2" r="13" stroke-width="1.5"/>
      <!-- atom labels -->
      <text id="mb-tcn" font-size="12" font-weight="bold" text-anchor="middle" dominant-baseline="middle" fill="white"/>
      <text id="mb-te1" font-size="11" font-weight="bold" text-anchor="middle" dominant-baseline="middle"/>
      <text id="mb-te2" font-size="11" font-weight="bold" text-anchor="middle" dominant-baseline="middle"/>
      <!-- dimension annotations -->
      <text id="mb-lr"  font-size="9" fill="#aaa" text-anchor="middle" font-style="italic"/>
      <text id="mb-lth" font-size="9" fill="#F7E49B" text-anchor="middle"/>
      <text id="mb-lee" font-size="9" fill="#aaa" text-anchor="middle" font-style="italic"/>
    </svg>
    <div id="mb-info" style="font-size:0.77rem;color:#888;text-align:center;margin:0.1rem 0 0.7rem;line-height:1.6;"></div>
    <div id="mbtr-ctrl" style="max-width:220px;"></div>
  </div>

  <!-- k=2 and k=3 plots -->
  <div style="flex:1 1 280px;min-width:250px;">
    <div id="mbtr-k2" style="height:205px;"></div>
    <div id="mbtr-k3" style="height:205px;margin-top:4px;"></div>
  </div>

</div>
</div>

<script>
var _MB={mol:'h2o',r:0.96,theta:104.5,sigma:0.04};

var _MMOLS={
  h2o:{r:0.96,theta:104.5,Zc:8,Ze:1,colc:'#C04040',cole:'#D8D8D8',strokec:'#881818',strokee:'#999',
       lc:'O',le:'H',chk2:['H–H','H–O'],chk3:['H–O–H','O–H–H'],
       rlabel:'r(O–H)',tlabel:'θ(H–O–H)',eelabel:'r(H···H)'},
  co2:{r:1.16,theta:180,Zc:6,Ze:8,colc:'#444',cole:'#C04040',strokec:'#222',strokee:'#881818',
       lc:'C',le:'O',chk2:['C–O','O–O'],chk3:['O–C–O','C–O–O'],
       rlabel:'r(C=O)',tlabel:'θ(O–C–O)',eelabel:'r(O···O)'},
  nh3:{r:1.01,theta:107.8,Zc:7,Ze:1,colc:'#3366AA',cole:'#D8D8D8',strokec:'#224488',strokee:'#999',
       lc:'N',le:'H',chk2:['H–H','H–N'],chk3:['H–N–H','N–H–H'],
       rlabel:'r(N–H)',tlabel:'θ(H–N–H)',eelabel:'r(H···H)'},
};

var _MK2COL=['#86BCBD','#BA5A5A','#A4CE8B'];
var _MK3COL=['#BA5A5A','#86BCBD','#A4CE8B'];

function _mbDark(){return document.body.getAttribute('data-md-color-scheme')==='slate';}
function _mbClr(){var d=_mbDark();return{fg:d?'#ddd':'#333',grid:d?'rgba(255,255,255,0.08)':'rgba(0,0,0,0.08)'}}
function _mbLs(a,b,n){return Array.from({length:n},function(_,i){return a+(b-a)*i/(n-1);});}
function _mbG(x,mu,s){return Math.exp(-0.5*Math.pow((x-mu)/s,2));}
function _mbLayout(xl,yl,title){
  var c=_mbClr();
  return{title:{text:title,font:{size:11,color:c.fg},x:0.5,xanchor:'center',pad:{t:2}},
    xaxis:{title:{text:xl,font:{size:11}},color:c.fg,gridcolor:c.grid,zeroline:false},
    yaxis:{title:{text:yl,font:{size:11}},color:c.fg,gridcolor:c.grid,zeroline:false},
    paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:c.fg,size:11},
    showlegend:true,legend:{font:{color:c.fg,size:10},bgcolor:'rgba(0,0,0,0)',x:1,xanchor:'right',y:1,yanchor:'top'},
    margin:{l:48,r:12,t:30,b:44}};
}

/* geometry helpers */
function _mbRee(){return 2*_MB.r*Math.sin(_MB.theta*Math.PI/360);}
function _mbCosCenter(){return Math.cos(_MB.theta*Math.PI/180);}
function _mbCosEnd(){return Math.sin(_MB.theta*Math.PI/360);}  /* = sin(θ/2) */

function mbtrPlotK2(){
  if(!document.getElementById('mbtr-k2')||!window.Plotly) return;
  var mol=_MMOLS[_MB.mol], s=_MB.sigma, xs=_mbLs(0,1.8,500), traces=[];
  var inv_r=1/_MB.r, inv_ree=1/_mbRee();
  /* channel 0: end–end pair (H-H or O-O or H-H) */
  traces.push({x:xs,y:xs.map(function(x){return _mbG(x,inv_ree,s);}),
    mode:'lines',name:mol.chk2[0],line:{color:_MK2COL[0],width:2.5}});
  /* channel 1: center–end pair (2 equivalent pairs → amplitude 2) */
  traces.push({x:xs,y:xs.map(function(x){return 2*_mbG(x,inv_r,s);}),
    mode:'lines',name:mol.chk2[1],line:{color:_MK2COL[1],width:2.5}});
  /* vertical markers at peak positions */
  var shapes=[
    {type:'line',x0:inv_ree,x1:inv_ree,y0:0,y1:1,yref:'paper',layer:'below',line:{color:_MK2COL[0],width:1,dash:'dot'}},
    {type:'line',x0:inv_r,x1:inv_r,y0:0,y1:1,yref:'paper',layer:'below',line:{color:_MK2COL[1],width:1,dash:'dot'}},
  ];
  var L=_mbLayout('1/r  (Å⁻¹)','D(x, g₂)  (a.u.)','k = 2  —  pairwise inverse distances');
  L.xaxis.range=[0,1.8]; L.shapes=shapes;
  Plotly.react('mbtr-k2',traces,L,{responsive:true,displayModeBar:false});
}

function mbtrPlotK3(){
  if(!document.getElementById('mbtr-k3')||!window.Plotly) return;
  var mol=_MMOLS[_MB.mol], s=_MB.sigma*2.5, xs=_mbLs(-1,1,500), traces=[];
  var cosC=_mbCosCenter(), cosE=_mbCosEnd();
  var linearMol=Math.abs(_MB.theta-180)<0.5;
  /* channel 0: end–center–end (angle at center) */
  traces.push({x:xs,y:xs.map(function(x){return _mbG(x,cosC,s);}),
    mode:'lines',name:mol.chk3[0],line:{color:_MK3COL[0],width:2.5}});
  /* channel 1: center–end–end (2 equivalent triples) */
  traces.push({x:xs,y:xs.map(function(x){return 2*_mbG(x,cosE,s);}),
    mode:'lines',name:mol.chk3[1],line:{color:_MK3COL[1],width:2.5,dash:linearMol?'solid':'solid'}});
  var shapes=[
    {type:'line',x0:cosC,x1:cosC,y0:0,y1:1,yref:'paper',layer:'below',line:{color:_MK3COL[0],width:1,dash:'dot'}},
    {type:'line',x0:cosE,x1:cosE,y0:0,y1:1,yref:'paper',layer:'below',line:{color:_MK3COL[1],width:1,dash:'dot'}},
  ];
  var L=_mbLayout('cos θ','D(x, g₃)  (a.u.)','k = 3  —  bond angles');
  L.xaxis.range=[-1,1];
  L.xaxis.tickvals=[-1,-0.5,0,0.5,1];
  L.xaxis.ticktext=['-1','−½','0','½','1'];
  L.shapes=shapes;
  Plotly.react('mbtr-k3',traces,L,{responsive:true,displayModeBar:false});
}

function mbtrUpdateSVG(){
  var mol=_MMOLS[_MB.mol],th=_MB.theta*Math.PI/180,r=_MB.r;
  var cx=110,cy=118,sc=66; /* scale: 66 px / Å */
  var x1=cx+sc*r*Math.sin(th/2), y1=cy-sc*r*Math.cos(th/2);
  var x2=cx-sc*r*Math.sin(th/2), y2=y1;
  var svg=document.getElementById('mbtr-svg');
  if(!svg) return;
  function sa(id,attrs){var e=svg.getElementById(id);if(!e) return;
    Object.keys(attrs).forEach(function(k){e.setAttribute(k,attrs[k]);});}
  /* bonds */
  sa('mb-b1',{x1:cx,y1:cy,x2:x1,y2:y1});
  sa('mb-b2',{x1:cx,y1:cy,x2:x2,y2:y2});
  sa('mb-b3',{x1:x1,y1:y1,x2:x2,y2:y2});
  /* atoms */
  sa('mb-cn',{cx:cx,cy:cy,fill:mol.colc,stroke:mol.strokec});
  sa('mb-e1',{cx:x1,cy:y1,fill:mol.cole,stroke:mol.strokee});
  sa('mb-e2',{cx:x2,cy:y2,fill:mol.cole,stroke:mol.strokee});
  /* atom labels */
  var tcn=svg.getElementById('mb-tcn');if(tcn){tcn.setAttribute('x',cx);tcn.setAttribute('y',cy);tcn.textContent=mol.lc;}
  var te1=svg.getElementById('mb-te1');if(te1){te1.setAttribute('x',x1);te1.setAttribute('y',y1);te1.textContent=mol.le;}
  var te2=svg.getElementById('mb-te2');if(te2){te2.setAttribute('x',x2);te2.setAttribute('y',y2);te2.textContent=mol.le;}
  /* r bond label: midpoint of bond 1, offset perpendicular */
  var mxr=(cx+x1)/2, myr=(cy+y1)/2;
  var dx=x1-cx,dy=y1-cy,len=Math.sqrt(dx*dx+dy*dy)||1;
  sa('mb-lr',{x:mxr+8*dy/len,y:myr-8*dx/len});
  var lr=svg.getElementById('mb-lr');if(lr) lr.textContent='r';
  /* angle arc */
  var aR=22;
  var ax1=cx+aR*Math.sin(th/2),ay1=cy-aR*Math.cos(th/2);
  var ax2=cx-aR*Math.sin(th/2),ay2=ay1;
  var showArc=_MB.theta<175;
  var arc=svg.getElementById('mb-arc');
  if(arc){
    if(showArc) arc.setAttribute('d','M '+ax1.toFixed(1)+' '+ay1.toFixed(1)+' A '+aR+' '+aR+' 0 0 0 '+ax2.toFixed(1)+' '+ay2.toFixed(1));
    else arc.setAttribute('d','');
  }
  /* θ label */
  var lth=svg.getElementById('mb-lth');
  if(lth){lth.setAttribute('x',cx);lth.setAttribute('y',cy-aR-6);lth.textContent=showArc?'θ':'';}
  /* end-end dashed label */
  var midex=(x1+x2)/2, midey=Math.min(y1,y2)-9;
  var lee=svg.getElementById('mb-lee');
  if(lee){lee.setAttribute('x',midex);lee.setAttribute('y',midey);
    lee.textContent=_mbRee().toFixed(2)+' Å';}
  /* info text */
  var m=_MMOLS[_MB.mol];
  var info=document.getElementById('mb-info');
  if(info) info.innerHTML=m.rlabel+'&nbsp;=&nbsp;<strong>'+_MB.r.toFixed(2)+'</strong>&nbsp;Å&nbsp;&nbsp;·&nbsp;&nbsp;'+
    m.tlabel+'&nbsp;=&nbsp;<strong>'+_MB.theta.toFixed(1)+'°</strong><br>'+
    m.eelabel+'&nbsp;=&nbsp;<strong>'+_mbRee().toFixed(2)+'</strong>&nbsp;Å';
}

function mbtrUpdate(k,v){
  _MB[k]=+v;
  var fmts={r:2,theta:1,sigma:3};
  var el=document.getElementById('mbv-'+k);
  if(el) el.textContent=(+_MB[k]).toFixed(fmts[k]);
  mbtrUpdateSVG(); mbtrPlotK2(); mbtrPlotK3();
}

function mbtrSetMol(mol){
  _MB.mol=mol;
  var m=_MMOLS[mol];
  _MB.r=m.r; _MB.theta=m.theta;
  ['h2o','co2','nh3'].forEach(function(id){
    var btn=document.getElementById('mbtr-btn-'+id);
    if(!btn) return;
    if(id===mol){btn.style.background='#86BCBD';btn.style.color='#fff';
      btn.style.border='1.5px solid #86BCBD';btn.style.fontWeight='600';}
    else{btn.style.background='transparent';btn.style.color='var(--md-default-fg-color)';
      btn.style.border='1.5px solid #8886';btn.style.fontWeight='normal';}
  });
  _renderMbtrCtrl(); mbtrUpdateSVG(); mbtrPlotK2(); mbtrPlotK3();
}

function _mbSlider(k,label,min,max,step,dp){
  return '<div style="margin-bottom:0.5rem;">'
    +'<label style="font-size:0.82rem;color:var(--md-default-fg-color);">'+label
    +' = <span id="mbv-'+k+'">'+(+_MB[k]).toFixed(dp)+'</span></label><br>'
    +'<input type="range" min="'+min+'" max="'+max+'" step="'+step+'" value="'+_MB[k]
    +'" style="width:100%;accent-color:#86BCBD;" oninput="mbtrUpdate(\''+k+'\',this.value)">'
    +'</div>';
}

function _renderMbtrCtrl(){
  var h='';
  h+=_mbSlider('r','r<sub>bond</sub> (Å)',0.80,1.60,0.01,2);
  h+=_mbSlider('theta','θ (°)',60,180,0.5,1);
  h+=_mbSlider('sigma','σ (Å⁻¹)',0.01,0.10,0.005,3);
  var el=document.getElementById('mbtr-ctrl');
  if(el) el.innerHTML=h;
}

function _mbtrInit(){
  if(!document.getElementById('mbtr-k2')) return;
  function _run(){_renderMbtrCtrl();mbtrUpdateSVG();mbtrPlotK2();mbtrPlotK3();}
  if(window.Plotly){_run();return;}
  var s=document.createElement('script');
  s.src='https://cdn.plot.ly/plotly-2.27.0.min.js';
  s.onload=_run; document.head.appendChild(s);
}
if(typeof document$!=='undefined'){document$.subscribe(_mbtrInit);}
else if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',_mbtrInit);}
else{_mbtrInit();}
</script>



---


## 2.5 Atom-Centered Symmetry Functions (ACSFs)

ACSFs encode the local chemical environment of atom $i$ by summing contributions from all neighbors within a cutoff radius $R_c$. All functions are constructed to be invariant under translation, rotation, and permutation of like atoms.

### Cutoff Function

Every ACSF is multiplied by a smooth cutoff that forces all functions and their derivatives to zero at $R_c$:

$$f_c(r) = \begin{cases} \dfrac{1}{2}\!\left[\cos\!\left(\dfrac{\pi r}{R_c}\right) + 1\right] & r \le R_c \\ 0 & r > R_c \end{cases}$$

This ensures the descriptor is smooth and that forces (derivatives of the energy with respect to atomic positions) are continuous.

### Radial Symmetry Functions (2-body)

Three radial functions probe the distribution of neighbor distances $r_{ij} = |\mathbf{R}_j - \mathbf{R}_i|$.

**$G^{(1)}$ — neighbor count.** The simplest function, with no free parameters beyond the cutoff:

$$G_i^{(1)} = \sum_{j \neq i} f_c(r_{ij})$$

It counts the effective number of neighbors within $R_c$, weighted by their distance. This gives a rough measure of coordination number.

**$G^{(2)}$ — Gaussian radial.** The most widely used radial function:

$$G_i^{(2)} = \sum_{j \neq i} \exp\!\left[-\eta\left(r_{ij} - \mu_s\right)^2\right] f_c(r_{ij})$$

The parameter $\eta > 0$ controls the width of the Gaussian peak and $\mu_s \ge 0$ shifts its center along $r$. By using many $(η, \mu_s)$ pairs, the full radial distribution of neighbors can be probed at different distances and resolutions.

**$G^{(3)}$ — cosine radial.** An oscillatory alternative to the Gaussian:

$$G_i^{(3)} = \sum_{j \neq i} \cos(\kappa\, r_{ij})\, f_c(r_{ij})$$

The frequency $\kappa$ controls how rapidly the function oscillates with distance. Less commonly used than $G^{(2)}$ in practice.

### Angular Symmetry Functions (3-body)

Angular functions additionally encode the angle $\theta_{ijk}$ at atom $i$ subtended by neighbor pair $(j, k)$, giving access to 3-body correlations.

**$G^{(4)}$ — angular with $r_{jk}$.** The full angular function:

$$G_i^{(4)} = 2^{1-\zeta}\sum_{\substack{j,k \neq i \\ j < k}}\left(1 + \lambda\cos\theta_{ijk}\right)^\zeta \exp\!\left[-\eta\left(r_{ij}^2 + r_{ik}^2 + r_{jk}^2\right)\right] f_c(r_{ij})\,f_c(r_{ik})\,f_c(r_{jk})$$

The parameter $\zeta > 0$ controls angular resolution (larger $\zeta$ gives sharper angular peaks), $\lambda \in \{-1, +1\}$ shifts the peak to $\theta = 0$ or $\theta = \pi$, and $\eta$ controls the radial decay. Including $r_{jk}$ makes the function sensitive to the distance between the two neighbors as well.

**$G^{(5)}$ — angular without $r_{jk}$.** Identical to $G^{(4)}$ but drops the $j$–$k$ distance term:

$$G_i^{(5)} = 2^{1-\zeta}\sum_{\substack{j,k \neq i \\ j < k}}\left(1 + \lambda\cos\theta_{ijk}\right)^\zeta \exp\!\left[-\eta\left(r_{ij}^2 + r_{ik}^2\right)\right] f_c(r_{ij})\,f_c(r_{ik})$$

Omitting $r_{jk}$ and $f_c(r_{jk})$ makes the function less sensitive to the separation between the two neighbors and somewhat cheaper to evaluate.

### The Descriptor Vector

The full ACSF fingerprint of atom $i$ is the concatenation of many symmetry function values evaluated at different parameter combinations — e.g. many $(\eta, \mu_s)$ pairs for $G^{(2)}$ and many $(\eta, \zeta, \lambda)$ combinations for $G^{(4)}$ or $G^{(5)}$. This gives a fixed-length vector that encodes the local chemical environment. One vector is computed per atom and per element species, then fed into an element-specific feedforward neural network whose output is the atomic energy contribution $\varepsilon_i$. The total energy is $E = \sum_i \varepsilon_i$.

### Interactive Explorer

<div id="acsf-widget" style="border:1px solid #8884;border-radius:8px;padding:1.2rem;margin:1.4rem 0;background:var(--md-code-bg-color,#f5f5f5);">

<!-- Title + function toggle buttons -->
<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:1rem;flex-wrap:wrap;">
  <span style="font-weight:600;font-size:0.95rem;color:var(--md-default-fg-color);">ACSF Explorer</span>
  <div style="display:flex;gap:5px;flex-wrap:wrap;">
    <button id="acsf-btn-g1" onclick="acsfToggleFn('g1')" style="padding:3px 12px;border-radius:20px;border:1.5px solid #8886;background:transparent;color:var(--md-default-fg-color);font-size:0.8rem;cursor:pointer;">G¹</button>
    <button id="acsf-btn-g2" onclick="acsfToggleFn('g2')" style="padding:3px 12px;border-radius:20px;border:1.5px solid #86BCBD;background:#86BCBD;color:#fff;font-size:0.8rem;cursor:pointer;font-weight:600;">G²</button>
    <button id="acsf-btn-g3" onclick="acsfToggleFn('g3')" style="padding:3px 12px;border-radius:20px;border:1.5px solid #8886;background:transparent;color:var(--md-default-fg-color);font-size:0.8rem;cursor:pointer;">G³</button>
    <button id="acsf-btn-g4" onclick="acsfToggleFn('g4')" style="padding:3px 12px;border-radius:20px;border:1.5px solid #8886;background:transparent;color:var(--md-default-fg-color);font-size:0.8rem;cursor:pointer;">G⁴</button>
    <button id="acsf-btn-g5" onclick="acsfToggleFn('g5')" style="padding:3px 12px;border-radius:20px;border:1.5px solid #8886;background:transparent;color:var(--md-default-fg-color);font-size:0.8rem;cursor:pointer;">G⁵</button>
  </div>
</div>

<!-- Function explorer: parameters + single plot -->
<div style="display:flex;gap:1.5rem;flex-wrap:wrap;align-items:flex-start;">
  <div id="acsf-ctrl" style="flex:0 0 auto;min-width:190px;max-width:215px;"></div>
  <div id="acsf-plot" style="flex:1 1 280px;min-width:250px;height:290px;"></div>
</div>

<!-- Descriptor basis section -->
<hr style="margin:1rem 0;border:none;border-top:1px solid #8882;">
<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.7rem;flex-wrap:wrap;">
  <span style="font-size:0.85rem;font-weight:600;color:var(--md-default-fg-color);">Descriptor basis</span>
  <span style="font-size:0.79rem;color:#888;">— N copies of each type tile parameter space; the full descriptor concatenates all these values</span>
  <div style="display:flex;align-items:center;gap:0.5rem;margin-left:auto;">
    <label style="font-size:0.82rem;white-space:nowrap;color:var(--md-default-fg-color);">N = <span id="av-nbasis">5</span></label>
    <input type="range" min="2" max="8" step="1" value="5" style="width:90px;accent-color:#86BCBD;" oninput="acsfUpdate('nbasis',this.value)">
  </div>
</div>
<div id="acsf-plot-b" style="height:260px;"></div>

</div>

<script>
var _A={Rc:6,eta:0.5,mus:2.0,kappa:1.0,zeta:4,lam:1,theta:60,nbasis:5,
        fns:{g1:false,g2:true,g3:false,g4:false,g5:false}};
var _XMAX=8;
var _FNCOL={g1:'#A4CE8B',g2:'#86BCBD',g3:'#F7E49B',g4:'#BA5A5A',g5:'#C07AB5'};
var _FNLAB={g1:'G¹',g2:'G²',g3:'G³',g4:'G⁴',g5:'G⁵'};
var _BPAL=['#0077BB','#33BBEE','#009988','#EE7733','#CC3311','#EE3377','#AA3377','#BBBB22'];

function _fc(r){return r>=_A.Rc?0:0.5*(Math.cos(Math.PI*r/_A.Rc)+1);}
function _ls(a,b,n){return Array.from({length:n},function(_,i){return a+(b-a)*i/(n-1);});}
function _dark(){return document.body.getAttribute('data-md-color-scheme')==='slate';}
function _clr(){var d=_dark();return{fg:d?'#ddd':'#333',grid:d?'rgba(255,255,255,0.08)':'rgba(0,0,0,0.08)'};}

function _layout(xl,yl){
  var c=_clr();
  return{
    xaxis:{title:{text:xl,font:{size:12}},color:c.fg,gridcolor:c.grid,zeroline:false},
    yaxis:{title:{text:yl,font:{size:12}},color:c.fg,gridcolor:c.grid,zeroline:true,zerolinecolor:c.grid},
    paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
    font:{color:c.fg,size:12},showlegend:true,
    legend:{font:{color:c.fg,size:10},bgcolor:'rgba(0,0,0,0)',x:1,xanchor:'right',y:1,yanchor:'top'},
    margin:{l:55,r:15,t:10,b:50}
  };
}

function _evalFn(fn,r,th_deg){
  var fc=_fc(r);
  var th=(th_deg!==undefined?th_deg:_A.theta)*Math.PI/180;
  var ang=Math.pow(Math.max(0,1+_A.lam*Math.cos(th)),_A.zeta);
  if(fn==='g1') return fc;
  if(fn==='g2') return Math.exp(-_A.eta*Math.pow(r-_A.mus,2))*fc;
  if(fn==='g3') return Math.cos(_A.kappa*r)*fc;
  var rjk=2*r*Math.abs(Math.sin(th/2));
  var prefac=Math.pow(2,1-_A.zeta)*ang;
  if(fn==='g4') return prefac*Math.exp(-_A.eta*(2*r*r+rjk*rjk))*fc*fc*_fc(rjk);
  if(fn==='g5') return prefac*Math.exp(-_A.eta*2*r*r)*fc*fc;
  return 0;
}

function _selFns(){return['g1','g2','g3','g4','g5'].filter(function(f){return _A.fns[f];});}

function acsfPlot(){
  if(!document.getElementById('acsf-plot')||!window.Plotly) return;
  var rs=_ls(0,_XMAX,400),sel=_selFns(),traces=[];
  traces.push({x:rs,y:rs.map(_fc),mode:'lines',name:'f<sub>c</sub>',
    line:{color:'#888',dash:'dot',width:1.2},opacity:0.5});
  sel.forEach(function(fn){
    traces.push({x:rs,y:rs.map(function(r){return _evalFn(fn,r);}),
      mode:'lines',name:_FNLAB[fn],line:{color:_FNCOL[fn],width:2.5}});
  });
  var hasAng=sel.indexOf('g4')!==-1||sel.indexOf('g5')!==-1;
  var yl=hasAng?'value   (θ = '+_A.theta+'°)':'value';
  var L=_layout('r (Å)',yl);
  L.xaxis.range=[0,Math.min(_A.Rc+0.5,_XMAX)];
  Plotly.react('acsf-plot',traces,L,{responsive:true,displayModeBar:false});
}

function acsfPlotBasis(){
  if(!document.getElementById('acsf-plot-b')||!window.Plotly) return;
  var rs=_ls(0,_XMAX,400),sel=_selFns(),n=_A.nbasis,traces=[];
  sel.forEach(function(fn){
    if(fn==='g1'){
      traces.push({x:rs,y:rs.map(_fc),mode:'lines',name:'G¹',
        line:{color:_FNCOL.g1,width:2}});
    } else if(fn==='g2'){
      _ls(0.5,Math.min(_A.Rc-0.5,_XMAX-0.5),n).forEach(function(mu,i){
        traces.push({x:rs,y:rs.map(function(r){return Math.exp(-_A.eta*Math.pow(r-mu,2))*_fc(r);}),
          mode:'lines',name:'G²  μ<sub>s</sub>='+mu.toFixed(1)+' Å',
          line:{color:_BPAL[i%_BPAL.length],width:2}});
      });
    } else if(fn==='g3'){
      _ls(0.5,3.0,Math.min(n,6)).forEach(function(kap,i){
        traces.push({x:rs,y:rs.map(function(r){return Math.cos(kap*r)*_fc(r);}),
          mode:'lines',name:'G³  κ='+kap.toFixed(1)+' Å⁻¹',
          line:{color:_BPAL[i%_BPAL.length],width:2}});
      });
    } else if(fn==='g4'||fn==='g5'){
      _ls(0,180,n).forEach(function(th,i){
        traces.push({x:rs,y:rs.map(function(r){return _evalFn(fn,r,th);}),
          mode:'lines',name:_FNLAB[fn]+'  θ='+th.toFixed(0)+'°',
          line:{color:_BPAL[i%_BPAL.length],width:2,dash:fn==='g5'?'dash':'solid'}});
      });
    }
  });
  if(!traces.length){
    Plotly.react('acsf-plot-b',[],_layout('r (Å)','descriptor components'),{responsive:true,displayModeBar:false});
    return;
  }
  var L=_layout('r (Å)','descriptor components');
  L.xaxis.range=[0,Math.min(_A.Rc+0.5,_XMAX)];
  L.legend.font.size=9;
  Plotly.react('acsf-plot-b',traces,L,{responsive:true,displayModeBar:false});
}

function acsfUpdate(k,v){
  _A[k]=k==='nbasis'?Math.round(+v):(+v);
  var fmts={Rc:1,eta:2,mus:1,kappa:1,zeta:1,theta:0,nbasis:0};
  var el=document.getElementById('av-'+k);
  if(el) el.textContent=(+_A[k]).toFixed(fmts[k]!==undefined?fmts[k]:1);
  acsfPlot(); acsfPlotBasis();
}

function acsfToggleFn(fn){
  _A.fns[fn]=!_A.fns[fn];
  var btn=document.getElementById('acsf-btn-'+fn);
  if(btn){
    if(_A.fns[fn]){
      btn.style.background=_FNCOL[fn];btn.style.color='#fff';
      btn.style.border='1.5px solid '+_FNCOL[fn];btn.style.fontWeight='600';
    } else {
      btn.style.background='transparent';btn.style.color='var(--md-default-fg-color)';
      btn.style.border='1.5px solid #8886';btn.style.fontWeight='normal';
    }
  }
  _renderCtrl(); acsfPlot(); acsfPlotBasis();
}

function _sl(k,label,min,max,step,val,dp){
  return '<div style="margin-bottom:0.5rem;">'
    +'<label style="font-size:0.82rem;color:var(--md-default-fg-color);">'+label
    +' = <span id="av-'+k+'">'+(+val).toFixed(dp)+'</span></label><br>'
    +'<input type="range" min="'+min+'" max="'+max+'" step="'+step+'" value="'+val
    +'" style="width:100%;accent-color:#86BCBD;" oninput="acsfUpdate(\''+k+'\',this.value)">'
    +'</div>';
}

function _renderCtrl(){
  var sel=_selFns();
  var hasG2=sel.indexOf('g2')!==-1;
  var hasG3=sel.indexOf('g3')!==-1;
  var hasG45=sel.indexOf('g4')!==-1||sel.indexOf('g5')!==-1;
  var h='';
  h+=_sl('Rc','R<sub>c</sub> (Å)',2,10,0.5,_A.Rc,1);
  if(hasG2||hasG45) h+=_sl('eta','η',0.01,2,0.01,_A.eta,2);
  if(hasG2) h+=_sl('mus','μ<sub>s</sub> (Å)',0,8,0.1,_A.mus,1);
  if(hasG3) h+=_sl('kappa','κ (Å⁻¹)',0.1,4,0.1,_A.kappa,1);
  if(hasG45){
    h+=_sl('theta','θ (°)',0,180,5,_A.theta,0);
    h+=_sl('zeta','ζ',0.5,16,0.5,_A.zeta,1);
    h+='<div style="margin-bottom:0.5rem;">'
      +'<label style="font-size:0.82rem;color:var(--md-default-fg-color);">λ = <span id="av-lam">'+_A.lam+'</span></label><br>'
      +'<button onclick="_A.lam*=-1;document.getElementById(\'av-lam\').textContent=_A.lam;acsfPlot();acsfPlotBasis();"'
      +' style="padding:2px 10px;border-radius:4px;border:1px solid #8886;background:var(--md-default-bg-color);'
      +'color:var(--md-default-fg-color);cursor:pointer;font-size:0.8rem;">Toggle ±1</button>'
      +'</div>';
  }
  if(!sel.length) h='<p style="font-size:0.82rem;color:#999;font-style:italic;">Select a function above.</p>';
  document.getElementById('acsf-ctrl').innerHTML=h;
}

function _init(){
  if(!document.getElementById('acsf-plot')) return;
  function _run(){ _renderCtrl(); acsfPlot(); acsfPlotBasis(); }
  if(window.Plotly){ _run(); return; }
  var s=document.createElement('script');
  s.src='https://cdn.plot.ly/plotly-2.27.0.min.js';
  s.onload=_run; document.head.appendChild(s);
}

if(typeof document$!=='undefined'){ document$.subscribe(_init); }
else if(document.readyState==='loading'){ document.addEventListener('DOMContentLoaded',_init); }
else { _init(); }
</script>

---


## 2.6 Weighted Atom-Centered Symmetry Functions (wACSFs)

Standard ACSFs handle multi-element systems by constructing *separate* symmetry functions for each element combination: one set of radial functions per element pair $(Z_i, Z_j)$ and one set of angular functions per triple $(Z_i, Z_j, Z_k)$. For a system with $S$ distinct elements this requires $\mathcal{O}(S)$ radial function sets and $\mathcal{O}(S^2)$ angular sets — parameter proliferation that becomes unwieldy for chemically diverse systems.

**Weighted ACSFs (wACSFs)** replace the element-specific function sets with a single unified set, weighting each neighbor's contribution by its nuclear charge $Z_j$. The weighted radial function is:

$$wG_i^{(2)} = \sum_{j \neq i} Z_j \exp\!\left[-\eta\left(r_{ij} - \mu_s\right)^2\right] f_c(r_{ij})$$

and the weighted angular function is:

$$wG_i^{(4)} = 2^{1-\zeta}\sum_{\substack{j,k \neq i \\ j < k}} Z_j Z_k \left(1 + \lambda\cos\theta_{ijk}\right)^\zeta \exp\!\left[-\eta\left(r_{ij}^2 + r_{ik}^2 + r_{jk}^2\right)\right] f_c(r_{ij})\,f_c(r_{ik})\,f_c(r_{jk})$$

The nuclear charge factors $Z_j$ (2-body) and $Z_j Z_k$ (3-body) act as a continuous chemical weighting: heavier elements contribute more strongly than lighter ones. This implicitly encodes the chemical identity of each neighbor without requiring separate functions per element pair, reducing the total number of descriptor parameters by a factor of order $S$.

The trade-off is that the $Z$-weighting is fixed and may not optimally distinguish all chemical environments — standard ACSFs with element-specific parameters remain more expressive when the element set is small and fixed. wACSFs generalize better across compositionally diverse systems and are a natural choice for MLIPs designed to span large regions of chemical space.

---

## 2.7 Smooth Overlap of Atomic Positions (SOAP)

SOAP constructs a rotationally invariant descriptor by representing the local atomic environment as a smooth density and computing its power spectrum under the rotation group. It is well suited for use with kernel methods.

**Step 1 — Density smearing with Gaussians.** The local environment of atom $i$ is represented as a sum of Gaussians centered on each neighbor $j$ within the cutoff:

$$\rho_i(\mathbf{r}) = \sum_{j \in \mathcal{N}(i)} \exp\!\left(-\alpha\,|\mathbf{r} - \mathbf{r}_{ij}|^2\right) f_c(r_{ij})$$

where $\mathbf{r}_{ij} = \mathbf{R}_j - \mathbf{R}_i$ is the displacement vector from atom $i$ to neighbor $j$, $\alpha > 0$ controls the Gaussian width, and $f_c$ is a smooth cutoff function. Here $\mathcal{N}(i)$ denotes the set of atoms within cutoff $R_c$.

**Step 2 — Representation via radial functions and spherical harmonics.** The density $\rho_i(\mathbf{r})$ is expanded in a product basis of radial functions $g_n(r)$ and real spherical harmonics $Y_l^m(\hat{\mathbf{r}})$:

$$\rho_i(\mathbf{r}) = \sum_{n,l,m} c_{nlm}^i\, g_n(r)\, Y_l^m(\hat{\mathbf{r}})$$

where $n = 1,\ldots,N_\text{max}$ indexes the radial basis, $l = 0,\ldots,L_\text{max}$ is the angular momentum quantum number, $m = -l,\ldots,l$ is the magnetic quantum number, and $\hat{\mathbf{r}} = \mathbf{r}/|\mathbf{r}|$ is the unit direction vector. The coefficients $c_{nlm}^i$ are obtained by projection onto the basis.

**Step 3 — Distance metric by averaging over rotations.** The power spectrum is obtained by contracting the $m$-components, yielding a descriptor that is invariant to rotations of the local environment:

$$p_{nn'l} = \sum_m \left(c_{nlm}\right)^* c_{n'lm}$$

(atom index $i$ suppressed for clarity).

**SOAP kernel**: the similarity between two environments is

$$k(\rho, \rho') = \left|\sum_{n,n',l} p_{nn'l}\, p'_{nn'l}\right|^\zeta$$

where $\zeta \ge 1$ controls the nonlinearity. Combined with Gaussian Process Regression (GPR), SOAP forms the basis of the GAP (Gaussian Approximation Potential) framework. As shown in the ACE connection below, SOAP corresponds to ACE at body order 3 (correlation order 2).

**Completeness failure.** Despite satisfying all the symmetry requirements, SOAP is *not* a complete descriptor. The power spectrum contraction $\sum_m c_{nlm}^* c_{n'lm}$ discards phase information, meaning two physically distinct environments can produce identical SOAP descriptors. This was proven rigorously by Pozdnyakov et al. (*Phys. Rev. Lett.* **125**, 166001, 2020), who constructed explicit pairs of inequivalent environments that are indistinguishable by SOAP. Any model built on SOAP cannot in principle distinguish such environment pairs. In practice these failures are rare for real materials, but they represent a fundamental theoretical limitation that ACE resolves by providing a provably complete basis at arbitrary body order.

Despite this limitation, SOAP is a powerful structural similarity metric. Mapping 15\,869 distinct ice polymorphs using the SOAP kernel distance reveals the structural diversity and clustering of ice phases — a demonstration of its practical utility for materials-space exploration.

---


## 2.8 Atomic Cluster Expansion (ACE)

ACE is a complete and systematic framework for constructing many-body descriptors and potentials. Its central achievement is showing that the local atomic energy $\varepsilon_i$ can be expressed exactly as a **linear combination of a rotationally invariant basis** $\{B_{i,\mathbf{v}}\}$, where the basis is built from products of one-particle functions summed over neighbors — making computation $\mathcal{O}(N_\text{neigh})$ rather than $\mathcal{O}(N_\text{neigh}^\nu)$.

### Step 1 — Many-body cluster expansion

By the generalized cluster expansion theorem, any local potential energy $\varepsilon_i$ depending only on the chemical environment within a cutoff can be written as a sum over body-order contributions:

$$\varepsilon_i = V^{(0)}(Z_i) + \sum_{j_1 \in \mathcal{N}(i)} V^{(1)}(x_{ij_1}) + \sum_{\substack{j_1, j_2 \in \mathcal{N}(i) \\ j_1 < j_2}} V^{(2)}(x_{ij_1}, x_{ij_2}) + \cdots + \sum_{\substack{j_1 < \cdots < j_\nu \\ j_t \in \mathcal{N}(i)}} V^{(\nu)}(x_{ij_1},\ldots,x_{ij_\nu})$$

where $x_{ij} = (\mathbf{r}_{ij}, Z_j)$ collects the displacement vector $\mathbf{r}_{ij} = \mathbf{R}_j - \mathbf{R}_i$ and the species $Z_j$ of neighbor $j$, and $\mathcal{N}(i)$ is the neighbor set of atom $i$ within cutoff $R_c$. The term $V^{(0)}(Z_i)$ is a species-dependent constant (free-atom energy). Naive evaluation of the $\nu$-body term costs $\mathcal{O}(N_\text{neigh}^\nu)$, which becomes intractable for $\nu \ge 4$.

Equivalently, by expanding the ordered sum into an unrestricted sum with a combinatorial prefactor:

$$\varepsilon_i = V^{(0)}(Z_i) + \sum_{j_1} U^{(1)}(x_{ij_1}) + \frac{1}{2!}\sum_{j_1,j_2} U^{(2)}(x_{ij_1},x_{ij_2}) + \cdots + \frac{1}{\nu!}\sum_{j_1,\ldots,j_\nu} U^{(\nu)}(x_{ij_1},\ldots,x_{ij_\nu})$$

### Step 2 — Expand each body-order term in a product basis

The key structural assumption is that each interaction function $U^{(\nu)}$ can be expanded in a **separable product** of one-particle basis functions $\phi_k$:

$$U^{(\nu)}(x_{ij_1},\ldots,x_{ij_\nu}) = \sum_{k_1,\ldots,k_\nu} c^{(Z_i)}_{k_1\cdots k_\nu}\, \phi_{k_1}(x_{ij_1})\cdots\phi_{k_\nu}(x_{ij_\nu})$$

where $c^{(Z_i)}_{k_1\cdots k_\nu}$ are coefficients that depend on the species $Z_i$ of the central atom, and $k_1,\ldots,k_\nu$ are basis indices. This separability is what enables the efficient factorization in Step 4.

### Step 3 — Atomic one-particle basis: radial functions and spherical harmonics

The one-particle basis functions are products of radial functions $R_{nl}$ and spherical harmonics $Y_l^m$, with a Kronecker delta selecting the correct species:

$$\phi_{znlm}(\mathbf{r}_{ij}, Z_j) = R_{nl}(r_{ij})\, Y_l^m(\hat{\mathbf{r}}_{ij})\, \delta_{z Z_j}$$

where $r_{ij} = |\mathbf{r}_{ij}|$ is the scalar distance, $\hat{\mathbf{r}}_{ij} = \mathbf{r}_{ij}/r_{ij}$ is the unit direction vector, $n$ is a radial quantum number, $l$ is the angular momentum quantum number, $m \in \{-l,\ldots,l\}$ is the magnetic quantum number, and $z$ is a species label. The Kronecker delta $\delta_{z Z_j}$ enforces that only neighbors of species $z$ contribute to the $z$-channel.

Summing over all neighbors gives the **one-particle basis matrix**:

$$A^i_{znlm} = \sum_{j \in \mathcal{N}(i)} \phi_{znlm}(\mathbf{r}_{ij}, Z_j)$$

This sum costs $\mathcal{O}(N_\text{neigh})$ and needs to be computed only once, regardless of the desired body order $\nu$.

### Step 4 — Form body-order products

Products of $A$ values directly give the $\nu$-body interaction terms. For a multi-index $\boldsymbol{v} = (z_1 n_1 l_1 m_1, \ldots, z_\nu n_\nu l_\nu m_\nu)$, the $\nu$-particle basis element is:

$$\tilde{A}^i_{\boldsymbol{v}} = \prod_{t=1}^\nu A^i_{z_t n_t l_t m_t}$$

This product is formed at negligible cost after the $\mathcal{O}(N_\text{neigh})$ neighbor loop. The cost of evaluating body order $\nu$ is therefore $\mathcal{O}(N_\text{neigh})$, not $\mathcal{O}(N_\text{neigh}^\nu)$ — the central computational advantage of ACE.

### Step 5 — Contract with Clebsch-Gordan coefficients to enforce rotational invariance

The products $\tilde{A}^i_{\boldsymbol{v}}$ are not rotationally invariant because they carry free magnetic quantum numbers $m_1,\ldots,m_\nu$. To project onto the rotationally invariant subspace, one contracts with generalized Clebsch-Gordan (Wigner 3j) coefficients. The resulting **symmetrized basis functions** $B$ are guaranteed to transform as scalars under $SO(3)$ rotations:

At correlation order 1 (2-body):

$$B^{(1)}_{in} = A^i_{n00}$$

(The $l=0$, $m=0$ component of $A$ is already a scalar.)

At correlation order 2 (3-body):

$$B^{(2)}_{in_1 n_2 l} = \sum_{m=-l}^{l} (-1)^m A^i_{n_1 lm}\, A^i_{n_2 l\text{,}-m}$$

(Contraction of two $l$-components via the trivial Clebsch-Gordan rule for coupling $l \otimes l \to 0$.)

At correlation order 3 (4-body):

$$B^{(3)}_{in_1 n_2 n_3, l_1 l_2 l_3} = \sum_{m_1,m_2,m_3} \begin{pmatrix} l_1 & l_2 & l_3 \\ m_1 & m_2 & m_3 \end{pmatrix} A^i_{n_1 l_1 m_1}\, A^i_{n_2 l_2 m_2}\, A^i_{n_3 l_3 m_3}$$

where $\left(\begin{smallmatrix} l_1 & l_2 & l_3 \\ m_1 & m_2 & m_3 \end{smallmatrix}\right)$ is the Wigner 3j symbol, which is nonzero only when $m_1 + m_2 + m_3 = 0$ and the triangle inequality $|l_1 - l_2| \le l_3 \le l_1 + l_2$ is satisfied.

### Final linear model

The local atomic energy is expressed as a linear combination of the symmetrized basis:

$$\varepsilon_i = \sum_{\boldsymbol{v}} c_{\boldsymbol{v}}\, B^i_{\boldsymbol{v}}$$

where $\boldsymbol{v}$ runs over all valid multi-indices and $c_{\boldsymbol{v}}$ are the fitting coefficients. The total energy is $E = \sum_i \varepsilon_i$. Since the energy is **linear in the coefficients** $c_{\boldsymbol{v}}$, fitting reduces to a standard linear least-squares or ridge regression problem, which is fast and avoids local minima. Forces are obtained analytically by differentiating $B^i_{\boldsymbol{v}}$ with respect to atomic positions.

### Scalar, vectorial, and tensorial outputs

The ACE framework is not limited to scalar output. By choosing different coupling schemes in the Clebsch-Gordan contraction step, one can construct basis functions $B^i_{\boldsymbol{v}}$ that transform as vectors, rank-2 tensors, or higher under $SO(3)$ rotations. This allows ACE to predict force vectors, dipole moments, polarizability tensors, and Born effective charges, with the correct transformation properties guaranteed by construction and without any additional symmetry engineering.


### What ACE does differently

| Challenge | Earlier approaches | ACE solution |
|-----------|------------------|--------------|
| $\mathcal{O}(N^\nu)$ cost of many-body sum | Truncate at 3-body (ACSFs angular) | Factorize into $A$ basis: $\mathcal{O}(N)$ |
| Rotational invariance is ad hoc | Hand-craft functions (ACSFs, SOAP) | Systematic Clebsch-Gordan coupling |
| Nonlinear model needed for accuracy | GAP: kernel regression | $B$ basis is linear → fast, no local minima |
| Completeness unclear | SOAP/ACSFs: heuristic coverage | ACE is provably complete (bijective) |

### ACE as a unifying framework

ACE subsumes all major hand-crafted descriptors as special cases:
- **ACSFs** (Behler) correspond to specific choices of radial and angular basis functions in the ACE framework.
- **SOAP** is equivalent to ACE at correlation order 2 (body order 3), using a particular radial basis and the specific Clebsch-Gordan contraction that produces the power spectrum.
- **Steinhardt bond-order parameters** are also a special case.

Crucially, ACE at correlation order 3 (body order 4) is **complete (bijective)** for environments with up to 7 neighbors: no two distinct environments map to the same descriptor. For practical systems with more neighbors, higher correlation orders are required.

---


## 2.9 Body Order

A descriptor or potential has **body order** $\nu$ if it depends on $\nu$-tuples of atoms — that is, if it cannot be written as a sum of contributions from $(\nu-1)$-tuples. 2-body = pairs only; 3-body = triplets (encodes angles); 4-body = quadruplets (encodes dihedral angles). Higher body order captures richer geometric information but increases computational cost.

| Descriptor | Body order | Locality |
|-----------|-----------|---------|
| Coulomb matrix | 2 (pairs) | Global |
| ACSFs — radial $G^{(2)}$ | 2 | Local |
| ACSFs — angular $G^{(4)}$ | 3 | Local |
| SOAP power spectrum | 3 (via contraction) | Local |
| MBTR $k=2$ | 2 | Global |
| MBTR $k=3$ | 3 | Global |
| ACE (correlation order $\nu$) | $\nu + 1$ | Local |

---


## 2.10 Descriptor Properties Summary

A useful descriptor must satisfy several formal properties. **Translational invariance** means the descriptor value is unchanged when all atomic positions are shifted by a constant vector; **rotational invariance** means it is unchanged under global rotations; **permutational invariance** means it is unchanged when identical atoms are relabeled. Beyond symmetry, a descriptor should be **unique** (injective): two physically distinct environments must map to distinct descriptor values. A stronger requirement is **completeness**: the descriptor space must form a complete basis for all invariant functions of the environment, so that in principle any property can be represented. Finally, **smoothness** requires that small changes in atomic positions produce small, continuous changes in the descriptor — a prerequisite for differentiable force evaluation.

| Descriptor | Trans. | Rot. | Perm. | Unique | Complete | Smooth | Key failure |
|-----------|:------:|:----:|:-----:|:------:|:--------:|:------:|-------------|
| Coulomb matrix | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | Atom-order dependent; eigenspectrum non-unique; sorting introduces discontinuities |
| MBTR | ✓ | ✓ | ✓ | ~ | ~ | ✓ | Unique for molecules; not unique for periodic systems (cell-shape dependence); truncated at $k \le 3$ |
| ACSF | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | Formally proven non-injective: distinct environments yield identical descriptors (Pozdnyakov et al., PRL 2020) |
| SOAP | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | Power spectrum fails completeness condition; phase information lost by $m$-contraction |
| ACE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Formally complete basis (Drautz 2019); no known failures |
| MACE | ✓ | ≈ | ✓ | ✓ | ✓ | ✓ | Internal node features are $E(3)$-equivariant (not invariant); energy output is invariant; complete in the limit of high body order |

Here ✓ = satisfied, ✗ = fails, ~ = conditional, ≈ = equivariant (energy is invariant). The central result motivating ACE over its predecessors is the Pozdnyakov et al. (2020) proof that any model using only 3-body or 4-body features — including ACSF and SOAP — will assign identical descriptors to certain pairs of physically distinct environments. ACE resolves this by providing a provably complete basis at arbitrary body order.

---


## 2.11 Historical Overview

| Year | Descriptor | Authors | Key Idea |
|------|-----------|---------|----------|
| 2007 | ACSFs | Behler & Parrinello | Symmetry functions, local, differentiable |
| 2012 | Coulomb Matrix | Rupp et al. | Electrostatic encoding of molecular geometry |
| 2013 | SOAP | Bartók et al. | Smooth density overlap, designed for kernels |
| 2017 | MBTR | Huo & Rupp | Many-body tensor, global, smooth distributions |
| 2018 | wACSFs | Gastegger et al. | Charge-weighted ACSFs for multi-element systems |
| 2019 | ACE | Drautz | Systematic many-body expansion, complete, linear |

---


## References

<a id="ref-bp07"></a>
(Behler & Parrinello, 2007) Behler, J. & Parrinello, M. *Generalized Neural-Network Representation of High-Dimensional Potential-Energy Surfaces.* Phys. Rev. Lett. **98**, 146401 (2007). [DOI](https://doi.org/10.1103/PhysRevLett.98.146401)

<a id="ref-bp11"></a>
(Behler, 2011) Behler, J. *Atom-centered symmetry functions for constructing high-dimensional neural network potentials.* J. Chem. Phys. **134**, 074106 (2011). [DOI](https://doi.org/10.1063/1.3553717)

<a id="ref-rup12"></a>
(Rupp et al., 2012) Rupp, M. et al. *Fast and Accurate Modeling of Molecular Atomization Energies with Machine Learning.* Phys. Rev. Lett. **108**, 058301 (2012). [DOI](https://doi.org/10.1103/PhysRevLett.108.058301)

<a id="ref-bar13"></a>
(Bartók et al., 2013) Bartók, A. P. et al. *On representing chemical environments.* Phys. Rev. B **87**, 184115 (2013). [DOI](https://doi.org/10.1103/PhysRevB.87.184115)

<a id="ref-wacsf18"></a>
(Gastegger et al., 2018) Gastegger, M. et al. *wACSF — Weighted atom-centered symmetry functions as descriptors in machine learning potentials.* J. Chem. Phys. **148**, 241709 (2018). [DOI](https://doi.org/10.1063/1.5019667)

<a id="ref-huo22"></a>
(Huo & Rupp, 2022) Huo, H. & Rupp, M. *Unified Representation of Molecules and Crystals for Machine Learning.* Mach. Learn.: Sci. Technol. **3**, 045017 (2022). [DOI](https://doi.org/10.1088/2632-2153/aca005)

<a id="ref-drautz19"></a>
(Drautz, 2019) Drautz, R. *Atomic cluster expansion for accurate and transferable interatomic potentials.* Phys. Rev. B **99**, 014104 (2019). [DOI](https://doi.org/10.1103/PhysRevB.99.014104)
