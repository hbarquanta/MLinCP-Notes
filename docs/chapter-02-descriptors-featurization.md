# Chapter 2 — Descriptors & Featurization

## 2.1 Why Descriptors?

In atomistic machine learning, the inputs $x^{(p)}$ describe atomic structures: a set of $N$ atoms with nuclear charges $\{Z_i\}_{i=1}^N$ and positions $\{\mathbf{R}_i\}_{i=1}^N$. Raw Cartesian coordinates cannot be used directly as ML inputs, because the learned function must respect the physical symmetries of the problem. A valid descriptor $\mathbf{x} = \phi(\{Z_i, \mathbf{R}_i\})$ must satisfy:

1. **Translational invariance**: $\phi$ is unchanged when all positions are shifted by a constant vector $\mathbf{t}$, i.e., $\phi(\{\mathbf{R}_i + \mathbf{t}\}) = \phi(\{\mathbf{R}_i\})$.
2. **Rotational invariance**: $\phi$ is unchanged under any rotation $\mathbf{R} \in SO(3)$ applied to all positions.
3. **Permutation invariance**: $\phi$ is unchanged when identical atoms are relabeled.
4. **Uniqueness**: distinct atomic environments should map to distinct descriptors (injectivity up to symmetry).
5. **Differentiability**: $\phi$ must be smooth and differentiable in atom positions so that forces $F_i = -\partial E / \partial \mathbf{R}_i$ can be computed analytically.

## 2.2 Chronological Overview

| Year | Descriptor | Authors | Key Idea |
|------|-----------|---------|----------|
| 2007 | ACSFs | Behler & Parrinello | Symmetry functions, local, differentiable |
| 2012 | Coulomb Matrix | Rupp et al. | Electrostatic encoding of molecular geometry |
| 2013 | SOAP | Bartók et al. | Smooth density overlap, designed for kernels |
| 2017 | MBTR | Huo & Rupp | Many-body tensor, global, smooth distributions |
| 2019 | ACE | Drautz | Systematic many-body expansion, complete, linear |

---

## 2.3 Atom-Centered Symmetry Functions (ACSFs)

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
<div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.9rem;flex-wrap:wrap;">
  <span style="font-weight:600;font-size:0.95rem;color:var(--md-default-fg-color);">ACSF Explorer</span>
  <div style="display:flex;gap:6px;">
    <button id="acsf-tab-s" onclick="acsfMode('single')" style="padding:3px 14px;border-radius:20px;border:1.5px solid #86BCBD;background:#86BCBD;color:#fff;font-size:0.8rem;cursor:pointer;font-weight:600;">Single</button>
    <button id="acsf-tab-b" onclick="acsfMode('basis')" style="padding:3px 14px;border-radius:20px;border:1.5px solid #8886;background:transparent;color:var(--md-default-fg-color);font-size:0.8rem;cursor:pointer;">Basis view</button>
  </div>
</div>
<div style="display:flex;gap:1.5rem;flex-wrap:wrap;align-items:flex-start;">
  <div id="acsf-ctrl" style="flex:0 0 auto;min-width:190px;max-width:215px;"></div>
  <div id="acsf-plot" style="flex:1 1 280px;min-width:250px;height:300px;"></div>
</div>
</div>

<script>
var _A = {
  mode:'single', fn:'g2',
  Rc:6, eta:0.5, mus:2.0, kappa:1.0, zeta:4, lam:1, nbasis:5
};
var _PAL = ['#BA5A5A','#A4CE8B','#86BCBD','#F7E49B','#BA5A5A','#A4CE8B','#86BCBD','#F7E49B'];
var _XMAX = 8;

function _fc(r){ return r>=_A.Rc?0:0.5*(Math.cos(Math.PI*r/_A.Rc)+1); }
function _ls(a,b,n){ return Array.from({length:n},function(_,i){return a+(b-a)*i/(n-1);}); }
function _dark(){ return document.body.getAttribute('data-md-color-scheme')==='slate'; }
function _c(){
  var d=_dark();
  return {fg:d?'#ddd':'#333', grid:d?'rgba(255,255,255,0.08)':'rgba(0,0,0,0.08)'};
}
function _layout(xl,yl){
  var c=_c();
  return {
    xaxis:{title:{text:xl,font:{size:12}},color:c.fg,gridcolor:c.grid,zeroline:false},
    yaxis:{title:{text:yl,font:{size:12}},color:c.fg,gridcolor:c.grid,zeroline:true,zerolinecolor:c.grid},
    paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
    font:{color:c.fg,size:12},showlegend:true,
    legend:{font:{color:c.fg,size:11},bgcolor:'rgba(0,0,0,0)',x:0.98,xanchor:'right'},
    margin:{l:55,r:15,t:15,b:50}
  };
}

function _plotSingle(){
  var rs=_ls(0,_XMAX,400), traces=[], L;
  if(['g1','g2','g3'].includes(_A.fn)){
    traces.push({x:rs,y:rs.map(_fc),mode:'lines',name:'f<sub>c</sub>',
      line:{color:'#A4CE8B',dash:'dot',width:1.5}});
    var yg=rs.map(function(r){
      if(_A.fn==='g1') return _fc(r);
      if(_A.fn==='g2') return Math.exp(-_A.eta*Math.pow(r-_A.mus,2))*_fc(r);
      if(_A.fn==='g3') return Math.cos(_A.kappa*r)*_fc(r);
    });
    traces.push({x:rs,y:yg,mode:'lines',name:_A.fn==='g1'?'G¹':_A.fn==='g2'?'G²':'G³',
      line:{color:'#86BCBD',width:2.5}});
    L=_layout('r (Å)','value'); L.xaxis.range=[0,_XMAX];
  } else {
    var th=_ls(0,Math.PI,400), deg=th.map(function(t){return t*180/Math.PI;});
    var ya=th.map(function(t){return Math.pow(Math.max(0,1+_A.lam*Math.cos(t)),_A.zeta);});
    traces.push({x:deg,y:ya,mode:'lines',name:_A.fn==='g4'?'G⁴':'G⁵',
      line:{color:'#86BCBD',width:2.5}});
    L=_layout('θ (°)','(1 + λ cos θ)^ζ');
    L.xaxis.range=[0,180]; L.xaxis.tickvals=[0,30,60,90,120,150,180];
  }
  Plotly.react('acsf-plot',traces,L,{responsive:true,displayModeBar:false});
}

function _plotBasis(){
  var rs=_ls(0,_XMAX,400), traces=[], n=Math.round(_A.nbasis);
  var centers=_ls(0.5,Math.min(_A.Rc-0.5,_XMAX-0.5),n);
  centers.forEach(function(mu,i){
    traces.push({x:rs,
      y:rs.map(function(r){return Math.exp(-_A.eta*Math.pow(r-mu,2))*_fc(r);}),
      mode:'lines',name:'μ<sub>s</sub>='+mu.toFixed(1),
      line:{color:_PAL[i%_PAL.length],width:2}});
  });
  var L=_layout('r (Å)','G² value'); L.xaxis.range=[0,_XMAX];
  Plotly.react('acsf-plot',traces,L,{responsive:true,displayModeBar:false});
}

function acsfPlot(){
  if(!document.getElementById('acsf-plot')||!window.Plotly) return;
  if(_A.mode==='single') _plotSingle(); else _plotBasis();
}

function acsfUpdate(k,v){
  _A[k]=k==='nbasis'?Math.round(+v):(+v);
  var fmts={Rc:1,eta:2,mus:1,kappa:1,zeta:1,nbasis:0};
  var el=document.getElementById('av-'+k);
  if(el) el.textContent=(+_A[k]).toFixed(fmts[k]!==undefined?fmts[k]:1);
  acsfPlot();
}

function _sl(k,label,min,max,step,val,dp){
  return '<div style="margin-bottom:0.55rem;">'
    +'<label style="font-size:0.82rem;color:var(--md-default-fg-color);">'+label
    +' = <span id="av-'+k+'">'+(+val).toFixed(dp)+'</span></label><br>'
    +'<input type="range" min="'+min+'" max="'+max+'" step="'+step+'" value="'+val
    +'" style="width:100%;accent-color:#86BCBD;"'
    +' oninput="acsfUpdate(\''+k+'\',this.value)">'
    +'</div>';
}

function _lamBtn(){
  return '<div style="margin-bottom:0.55rem;">'
    +'<label style="font-size:0.82rem;color:var(--md-default-fg-color);">λ = <span id="av-lam">'+_A.lam+'</span></label><br>'
    +'<button onclick="_A.lam*=-1;document.getElementById(\'av-lam\').textContent=_A.lam;acsfPlot();"'
    +' style="padding:2px 10px;border-radius:4px;border:1px solid #8886;background:var(--md-default-bg-color);'
    +'color:var(--md-default-fg-color);cursor:pointer;font-size:0.8rem;">Toggle ±1</button>'
    +'</div>';
}

function _renderCtrl(){
  var h='', fn=_A.fn;
  if(_A.mode==='single'){
    h+='<div style="margin-bottom:0.8rem;">'
      +'<label style="font-size:0.82rem;font-weight:600;">Function</label><br>'
      +'<select onchange="acsfSetFn(this.value)" style="margin:0.25rem 0;width:100%;padding:5px;'
      +'border-radius:4px;border:1px solid #8886;background:var(--md-default-bg-color);'
      +'color:var(--md-default-fg-color);font-size:0.82rem;">'
      +['g1','g2','g3','g4','g5'].map(function(f){
        var n={g1:'G¹',g2:'G²',g3:'G³',g4:'G⁴',g5:'G⁵'};
        return '<option value="'+f+'"'+(f===fn?' selected':'')+'>'+n[f]+'</option>';
      }).join('')+'</select></div>';
    h+=_sl('Rc','R<sub>c</sub> (Å)',2,10,0.5,_A.Rc,1);
    if(fn==='g2'){h+=_sl('eta','η',0.05,3,0.05,_A.eta,2);h+=_sl('mus','μ<sub>s</sub> (Å)',0,7,0.1,_A.mus,1);}
    if(fn==='g3') h+=_sl('kappa','κ (Å⁻¹)',0.1,4,0.1,_A.kappa,1);
    if(fn==='g4'||fn==='g5'){h+=_sl('eta','η',0.01,1,0.01,_A.eta,2);h+=_sl('zeta','ζ',0.5,16,0.5,_A.zeta,1);h+=_lamBtn();}
  } else {
    h+='<p style="font-size:0.8rem;color:var(--md-default-fg-color);margin:0 0 0.6rem 0;">G² functions with evenly spaced μ<sub>s</sub>, showing how they cover radial space to form the descriptor.</p>';
    h+=_sl('Rc','R<sub>c</sub> (Å)',2,10,0.5,_A.Rc,1);
    h+=_sl('eta','η',0.05,3,0.05,_A.eta,2);
    h+=_sl('nbasis','N functions',2,8,1,_A.nbasis,0);
  }
  document.getElementById('acsf-ctrl').innerHTML=h;
}

function acsfSetFn(fn){ _A.fn=fn; acsfPlot(); }

function acsfMode(m){
  _A.mode=m;
  var s=document.getElementById('acsf-tab-s'), b=document.getElementById('acsf-tab-b');
  if(m==='single'){
    s.style.background='#86BCBD';s.style.color='#fff';s.style.border='1.5px solid #86BCBD';s.style.fontWeight='600';
    b.style.background='transparent';b.style.color='var(--md-default-fg-color)';b.style.border='1.5px solid #8886';b.style.fontWeight='normal';
  } else {
    b.style.background='#86BCBD';b.style.color='#fff';b.style.border='1.5px solid #86BCBD';b.style.fontWeight='600';
    s.style.background='transparent';s.style.color='var(--md-default-fg-color)';s.style.border='1.5px solid #8886';s.style.fontWeight='normal';
  }
  _renderCtrl(); acsfPlot();
}

function _init(){
  if(!document.getElementById('acsf-plot')) return;
  function _run(){ _renderCtrl(); acsfPlot(); }
  if(window.Plotly){ _run(); return; }
  var s=document.createElement('script');
  s.src='https://cdn.plot.ly/plotly-2.27.0.min.js';
  s.onload=_run;
  document.head.appendChild(s);
}

if(typeof document$!=='undefined'){ document$.subscribe(_init); }
else if(document.readyState==='loading'){ document.addEventListener('DOMContentLoaded',_init); }
else { _init(); }
</script>
---

## 2.4 Coulomb Matrix

The Coulomb matrix $M \in \mathbb{R}^{N_\text{at}\times N_\text{at}}$ encodes the electrostatic interactions among all $N_\text{at}$ atoms in a molecule:

$$M_{ij} = \begin{cases} \dfrac{1}{2} Z_i^{2.4} & i = j \\[6pt] \dfrac{Z_i Z_j}{|\mathbf{R}_i - \mathbf{R}_j|} & i \neq j \end{cases}$$

where $Z_i$ is the nuclear charge (atomic number) of atom $i$ and $|\mathbf{R}_i - \mathbf{R}_j|$ is the Euclidean distance between atoms $i$ and $j$. The diagonal element $\frac{1}{2}Z_i^{2.4}$ approximates the free-atom potential energy (Thomas-Fermi model). The off-diagonal element is the classical nuclear repulsion energy.

The Coulomb matrix is a **global** descriptor — it encodes the entire molecule, not the local environment of individual atoms — which makes it size-dependent and inapplicable to periodic systems (it would require padding with zeros). It is not permutation invariant by construction: reordering atoms changes $M$. Common remedies are sorting rows and columns by their $\ell^2$ norm, or using the sorted eigenvalue spectrum of $M$, which is permutation invariant but loses some structural information.

The Coulomb matrix encodes only pairwise distances (2-body) and misses angular information entirely.

---

## 2.5 Smooth Overlap of Atomic Positions (SOAP)

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

---

## 2.6 Many-Body Tensor Representation (MBTR)

The MBTR is a **global** descriptor that encodes the full structure of a molecule by representing many-body interactions as smooth one-dimensional distributions over geometric quantities. Each geometric quantity (e.g. an inverse distance or bond angle) contributes a Gaussian peak weighted by the nuclear charges of the involved atoms, giving a differentiable, continuous fingerprint that is permutation invariant by construction.

Three interaction levels are defined. For each level $k$, the descriptor is the distribution:

$$D(x,\, g_k) = \sum_\text{atom tuples} w_k \cdot \mathcal{N}\!\left(x;\; g_k(i,j,\ldots),\; \sigma^2\right)$$

where $g_k(i,j,\ldots)$ is the geometric function evaluated for the atom tuple, $w_k$ is a nuclear-charge weighting factor, $\sigma$ is a broadening width (hyperparameter), and $\mathcal{N}(x;\mu,\sigma^2)$ denotes a Gaussian with mean $\mu$ and variance $\sigma^2$.

**$k=1$ (1-body)**: $g_1(i) = Z_i$ (nuclear charge); encodes composition.

**$k=2$ (2-body)**: $g_2(i,j) = 1/r_{ij}$ (inverse distance), $w_k = Z_i Z_j$; encodes the distribution of bond distances.

**$k=3$ (3-body)**: $g_3(i,j,k) = \cos\theta_{ijk}$ (cosine of the angle at atom $j$ between atoms $i$ and $k$), $w_k = Z_i Z_j Z_k$; encodes angular structure.

The full MBTR vector is the concatenation of $D(x, g_1)$, $D(x, g_2)$, $D(x, g_3)$ evaluated on fine grids. The 3-body term costs $\mathcal{O}(N_\text{at}^3)$ in the number of atoms. Because the descriptor sums over all atoms globally, it is not directly applicable to large periodic systems or for computing atom-wise (local) energies.

---

## 2.7 Body Order

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

## 2.9 Descriptor Properties Summary

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

## References

<a id="ref-bp07"></a>
[BP07] Behler, J. & Parrinello, M. *Generalized Neural-Network Representation of High-Dimensional Potential-Energy Surfaces.* Phys. Rev. Lett. **98**, 146401 (2007). [DOI](https://doi.org/10.1103/PhysRevLett.98.146401)

<a id="ref-bp11"></a>
[BP11] Behler, J. *Atom-centered symmetry functions for constructing high-dimensional neural network potentials.* J. Chem. Phys. **134**, 074106 (2011). [DOI](https://doi.org/10.1063/1.3553717)

<a id="ref-rup12"></a>
[Rup12] Rupp, M. et al. *Fast and Accurate Modeling of Molecular Atomization Energies with Machine Learning.* Phys. Rev. Lett. **108**, 058301 (2012). [DOI](https://doi.org/10.1103/PhysRevLett.108.058301)

<a id="ref-bar13"></a>
[Bar13] Bartók, A. P. et al. *On representing chemical environments.* Phys. Rev. B **87**, 184115 (2013). [DOI](https://doi.org/10.1103/PhysRevB.87.184115)

<a id="ref-huo22"></a>
[HR22] Huo, H. & Rupp, M. *Unified Representation of Molecules and Crystals for Machine Learning.* Mach. Learn.: Sci. Technol. **3**, 045017 (2022). [DOI](https://doi.org/10.1088/2632-2153/aca005)

<a id="ref-drautz19"></a>
[Dra19] Drautz, R. *Atomic cluster expansion for accurate and transferable interatomic potentials.* Phys. Rev. B **99**, 014104 (2019). [DOI](https://doi.org/10.1103/PhysRevB.99.014104)
