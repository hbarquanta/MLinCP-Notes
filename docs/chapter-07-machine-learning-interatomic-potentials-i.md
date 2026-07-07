# Chapter 7: Machine Learning Interatomic Potentials I

## 7.1 The Many-Body Problem and the Schrödinger Equation

The fundamental equation governing all of chemistry and materials science is the time-independent Schrödinger equation,

$$\hat{H}\psi = E\psi,$$

where $\psi$ is the many-body wavefunction of a system of $N$ electrons and $M$ nuclei. The full Hamiltonian contains five terms (in atomic units):

$$\hat{H} = -\sum_{I=1}^{M}\frac{\nabla_I^2}{2M_I} - \sum_{i=1}^{N}\frac{\nabla_i^2}{2} + \sum_{I<J}\frac{q_I q_J}{r_{IJ}} - \sum_{i=1}^{N}\sum_{J=1}^{M}\frac{q_J}{r_{iJ}} + \sum_{i<j}\frac{1}{r_{ij}},$$

representing, in order: the kinetic energy of the nuclei, the kinetic energy of the electrons, nucleus–nucleus repulsion, nucleus–electron attraction, and electron–electron repulsion. The wavefunction $\psi(\mathbf{r}_i, \mathbf{R}_I)$ depends on the coordinates of all electrons and all nuclei simultaneously, making the exact solution intractable for any system beyond hydrogen.

## 7.2 The Born–Oppenheimer Approximation and the Potential Energy Surface

The key simplification comes from the large mass difference between electrons and nuclei: a proton is roughly 1836 times heavier than an electron, so $M_I \approx 1836\, m_e$ for a hydrogen nucleus and much more for heavier atoms. Because of this, nuclei move orders of magnitude more slowly than electrons. From the perspective of the electrons, the nuclei are essentially stationary; from the perspective of the nuclei, the electrons adjust instantaneously to any change in nuclear configuration.

This motivates the **Born–Oppenheimer (BO) approximation**: assume the total wavefunction factorizes into a product of an electronic part (depending parametrically on nuclear positions $\{\mathbf{R}_I\}$) and a nuclear part:

$$\psi(\mathbf{r}_i, \mathbf{R}_I) = \psi_e(\mathbf{r}_i;\, \mathbf{R}_I)\cdot \psi_N(\mathbf{R}_I).$$

Substituting this into the full Schrödinger equation and treating the nuclear kinetic energy term as negligible for the electronic problem, one arrives at the **electronic Schrödinger equation**:

$$\hat{H}_e\,\psi_e = E_e\,\psi_e, \qquad \hat{H}_e = -\sum_i \frac{\nabla_i^2}{2} - \sum_{i,J}\frac{q_J}{r_{iJ}} + \sum_{i<j}\frac{1}{r_{ij}} + \underbrace{\sum_{I<J}\frac{q_Iq_J}{r_{IJ}}}_{\text{const. for fixed }\{\mathbf{R}_I\}}.$$

Solving this equation for a given set of nuclear positions $\{\mathbf{R}_I\}$ yields the electronic energy $E_e(\{\mathbf{R}_I\})$. Repeating this for all possible nuclear configurations defines the **Potential Energy Surface (PES)**, a scalar function over the $3M$-dimensional space of nuclear coordinates that encodes all the chemistry of the system: bond lengths, reaction barriers, phase transitions, and spectra.

The PES has a rich geometry that encodes all chemistry of the system. Its key special points are:

**Minima** ($\nabla E = 0$, all curvatures positive): correspond to stable molecular geometries. The system at rest sits in one of these basins.

**First-order saddle points** ($\nabla E = 0$, exactly one negative curvature): correspond to **transition states**, the highest-energy point along the minimum-energy path connecting two minima. They separate reactant and product basins and control reaction kinetics.

The rate of thermally activated passage over a barrier is given by **transition state theory (Eyring equation)**:

$$k = \frac{k_BT}{h}\,e^{-\Delta G^\ddagger/RT},$$

where $\Delta G^\ddagger$ is the free energy difference between the transition state and the reactant minimum. The exponential dependence on $\Delta G^\ddagger$ means even small changes in barrier height dramatically affect the rate; a 0.06 eV reduction in barrier at room temperature increases the rate by roughly a factor of 10.

<div id="pes-widget" style="background:#faf8f5;border:1px solid #e0dbd4;border-radius:8px;padding:1.2rem 1rem 1rem;margin:1.8rem 0;font-family:inherit;">
<div style="text-align:center;font-size:0.84rem;font-weight:700;color:#555;margin-bottom:0.8rem;letter-spacing:0.01em;">Potential Energy Surface &amp; Eyring Equation</div>
<div style="font-size:0.77rem;text-align:center;color:#888;margin-bottom:0.7rem;">V(&xi;) = a(&xi;&sup2;&minus;1)&sup2; + b&xi; &nbsp;|&nbsp; adjust sliders to see how barrier height controls reaction rate</div>
<div style="display:flex;gap:1rem;align-items:flex-start;flex-wrap:wrap;">
<svg id="pes-svg" style="flex:2;min-width:260px;height:270px;display:block;flex-shrink:0;"></svg>
<div id="pes-info" style="flex:1;min-width:180px;padding-top:0.1rem;"></div>
</div>
<div style="margin-top:0.6rem;font-size:0.74rem;color:#555;display:grid;grid-template-columns:auto 1fr 56px;gap:0.15rem 0.45rem;align-items:center;">
<label>Barrier <em>a</em></label>
<input type="range" id="pes-a" min="5" max="50" value="25" style="width:100%;height:16px;" oninput="pesUp()">
<span id="pes-av" style="text-align:right;font-variant-numeric:tabular-nums;">0.25 eV</span>
<label>Asymmetry <em>b</em></label>
<input type="range" id="pes-b" min="-15" max="15" value="0" style="width:100%;height:16px;" oninput="pesUp()">
<span id="pes-bv" style="text-align:right;font-variant-numeric:tabular-nums;">+0.00 eV</span>
<label>Temperature</label>
<input type="range" id="pes-T" min="100" max="700" value="300" style="width:100%;height:16px;" oninput="pesUp()">
<span id="pes-Tv" style="text-align:right;font-variant-numeric:tabular-nums;">300 K</span>
</div>
</div>
<script>
(function(){
var NS='http://www.w3.org/2000/svg';
function V(x,a,b){return a*(x*x-1)*(x*x-1)+b*x;}
function dV(x,a,b){return 4*a*x*(x*x-1)+b;}
function d2V(x,a){return 12*a*x*x-4*a;}
function findCrit(a,b){
  var res={R:null,TS:null,P:null},N=797,xlo=-1.8,xhi=1.8,dp=null,xp=null;
  for(var i=0;i<=N;i++){
    var x=xlo+i*(xhi-xlo)/N,d=dV(x,a,b);
    if(dp!==null&&dp*d<0){
      var xc=xp-dp*(x-xp)/(d-dp),vc=V(xc,a,b);
      if(d2V(xc,a)>0){if(!res.R)res.R={x:xc,v:vc};else res.P={x:xc,v:vc};}
      else res.TS={x:xc,v:vc};
    }
    dp=d;xp=x;
  }
  return res;
}
function mk(tag,at,tx){var e=document.createElementNS(NS,tag);for(var k in at)e.setAttribute(k,at[k]);if(tx!=null)e.textContent=tx;return e;}
function kx(s){try{return katex.renderToString(s,{throwOnError:false,displayMode:false});}catch(e){return s;}}
window.pesUp=function(){
  var a=parseFloat(document.getElementById('pes-a').value)/100;
  var b=parseFloat(document.getElementById('pes-b').value)/100;
  var T=parseFloat(document.getElementById('pes-T').value);
  document.getElementById('pes-av').textContent=a.toFixed(2)+' eV';
  document.getElementById('pes-bv').textContent=(b>=0?'+':'')+b.toFixed(2)+' eV';
  document.getElementById('pes-Tv').textContent=Math.round(T)+' K';
  var svg=document.getElementById('pes-svg');
  if(!svg)return;
  var bbox=svg.getBoundingClientRect();
  var W=bbox.width>20?Math.floor(bbox.width):320;
  var H=bbox.height>20?Math.floor(bbox.height):270;
  svg.setAttribute('viewBox','0 0 '+W+' '+H);
  svg.innerHTML='';
  var cr=findCrit(a,b);
  if(!cr.R||!cr.TS||!cr.P)return;
  var vR=cr.R.v,vTS=cr.TS.v,vP=cr.P.v;
  var vlo=Math.min(vR,vP)-0.04,vhi=vTS+0.08;
  var PL=52,PR=14,PT=18,PB=32,PW=W-PL-PR,PH=H-PT-PB;
  function px(x){return PL+PW*(x+1.8)/3.6;}
  function py(v){return PT+PH*(1-(v-vlo)/(vhi-vlo));}
  svg.appendChild(mk('rect',{x:PL,y:PT,width:PW,height:PH,fill:'#fff',stroke:'#e0dbd4','stroke-width':1}));
  if(vlo<0&&vhi>0){svg.appendChild(mk('line',{x1:PL,y1:py(0),x2:PL+PW,y2:py(0),stroke:'#f0ece8','stroke-width':1}));}
  [[vR,'#A4CE8B'],[vTS,'#BA5A5A'],[vP,'#86BCBD']].forEach(function(p){
    svg.appendChild(mk('line',{x1:PL,y1:py(p[0]),x2:PL+PW,y2:py(p[0]),stroke:p[1],'stroke-width':0.9,'stroke-dasharray':'4,3',opacity:0.55}));
  });
  var pts=[];
  for(var i=0;i<=450;i++){var xi=-1.8+i*3.6/450,vi=V(xi,a,b);if(vi<=vhi+0.3&&vi>=vlo-0.3)pts.push(px(xi).toFixed(1)+','+py(vi).toFixed(1));}
  svg.appendChild(mk('polyline',{points:pts.join(' '),fill:'none',stroke:'#333','stroke-width':2.3,'stroke-linejoin':'round','stroke-linecap':'round'}));
  var bfwd=vTS-vR,brv=vTS-vP,dG=vP-vR;
  var xa=px(cr.R.x)+14;
  svg.appendChild(mk('line',{x1:xa,y1:py(vR),x2:xa,y2:py(vTS),stroke:'#aaa','stroke-width':1,'stroke-dasharray':'2,2'}));
  svg.appendChild(mk('text',{x:xa+3,y:(py(vR)+py(vTS))/2+3,'font-size':9.5,'fill':'#888','font-family':'inherit'},'ΔG‡'));
  var ya=py(Math.max(vR,vP))-10;
  svg.appendChild(mk('line',{x1:px(cr.R.x),y1:ya,x2:px(cr.P.x),y2:ya,stroke:'#ccc','stroke-width':0.9,'stroke-dasharray':'2,2'}));
  svg.appendChild(mk('text',{x:(px(cr.R.x)+px(cr.P.x))/2,y:ya-3,'text-anchor':'middle','font-size':8.5,'fill':'#ccc','font-family':'inherit'},'ΔG'));
  svg.appendChild(mk('circle',{cx:px(cr.R.x),cy:py(vR),r:5.5,fill:'#A4CE8B',stroke:'#fff','stroke-width':1.5}));
  svg.appendChild(mk('circle',{cx:px(cr.TS.x),cy:py(vTS),r:5.5,fill:'#BA5A5A',stroke:'#fff','stroke-width':1.5}));
  svg.appendChild(mk('circle',{cx:px(cr.P.x),cy:py(vP),r:5.5,fill:'#86BCBD',stroke:'#fff','stroke-width':1.5}));
  svg.appendChild(mk('text',{x:px(cr.R.x),y:py(vR)+17,'text-anchor':'middle','font-size':9.5,'fill':'#444','font-weight':'600','font-family':'inherit'},'R'));
  svg.appendChild(mk('text',{x:px(cr.TS.x),y:py(vTS)-10,'text-anchor':'middle','font-size':9.5,'fill':'#BA5A5A','font-weight':'600','font-family':'inherit'},'TS'));
  svg.appendChild(mk('text',{x:px(cr.P.x),y:py(vP)+17,'text-anchor':'middle','font-size':9.5,'fill':'#444','font-weight':'600','font-family':'inherit'},'P'));
  svg.appendChild(mk('text',{x:PL+PW/2,y:H-4,'text-anchor':'middle','font-size':9,'fill':'#bbb','font-family':'inherit'},'Reaction coordinate ξ'));
  svg.appendChild(mk('text',{x:11,y:PT+PH/2,'text-anchor':'middle','font-size':9,'fill':'#bbb',transform:'rotate(-90,11,'+(PT+PH/2)+')',  'font-family':'inherit'},'V / eV'));
  var vstep=(vhi-vlo)>0.3?0.1:0.05;
  for(var vi2=Math.ceil(vlo/vstep)*vstep;vi2<=vhi+0.001;vi2+=vstep){
    var yy=py(vi2);if(yy<PT-2||yy>PT+PH+2)continue;
    svg.appendChild(mk('line',{x1:PL-3,y1:yy,x2:PL,y2:yy,stroke:'#ccc','stroke-width':0.8}));
    svg.appendChild(mk('text',{x:PL-5,y:yy+3,'text-anchor':'end','font-size':8,'fill':'#bbb','font-family':'inherit'},vi2.toFixed(2)));
  }
  var kBT=8.617e-5*T,krat=Math.exp(-dG/kBT);
  var kfmt=krat>1e5?krat.toExponential(1):krat>99?Math.round(krat).toString():krat.toFixed(krat<0.01?4:2);
  var info=document.getElementById('pes-info');
  if(!info)return;
  info.innerHTML=
    '<div style="background:#f5f2ee;border-radius:6px;padding:0.5rem 0.65rem;border:1px solid #e5dfd7;">'+
    '<div style="font-size:0.79rem;font-weight:700;color:#555;margin-bottom:0.3rem;">Energetics</div>'+
    '<table style="font-size:0.8rem;border-collapse:collapse;width:100%;"><tbody>'+
    '<tr><td style="color:#A4CE8B;font-weight:600;padding-right:0.5rem;padding-bottom:0.2rem;">'+kx('\\Delta G^\\ddagger_{\\mathrm{fwd}}')+'</td><td style="padding-bottom:0.2rem;">'+bfwd.toFixed(3)+' eV</td></tr>'+
    '<tr><td style="color:#86BCBD;font-weight:600;padding-right:0.5rem;padding-bottom:0.2rem;">'+kx('\\Delta G^\\ddagger_{\\mathrm{rev}}')+'</td><td style="padding-bottom:0.2rem;">'+brv.toFixed(3)+' eV</td></tr>'+
    '<tr><td style="font-weight:600;padding-right:0.5rem;">'+kx('\\Delta G_{\\mathrm{rxn}}')+'</td><td>'+(dG>=0?'+':'')+dG.toFixed(3)+' eV</td></tr>'+
    '</tbody></table>'+
    '<hr style="margin:0.35rem 0;border:none;border-top:1px solid #e0dbd4;">'+
    '<div style="font-size:0.78rem;color:#888;">'+kx('k_{\\mathrm{B}}T')+' = '+(kBT*1000).toFixed(1)+' meV</div>'+
    '<div style="font-size:0.78rem;margin-top:0.12rem;">'+kx('\\dfrac{k_{\\mathrm{fwd}}}{k_{\\mathrm{rev}}} = e^{-\\Delta G / k_{\\mathrm{B}}T}')+'</div>'+
    '<div style="font-size:0.97rem;font-weight:700;color:#333;margin-top:0.1rem;">= '+kfmt+'</div>'+
    '<div style="font-size:0.7rem;color:#bbb;margin-top:0.25rem;line-height:1.4;">0.06 eV barrier &darr; at 300 K<br>&rArr; rate &times;10</div>'+
    '</div>';
};
function init(){if(!document.getElementById('pes-svg'))return;pesUp();}
if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',init);}else{init();}
if(typeof document$!=='undefined'){document$.subscribe(function(){setTimeout(init,50);});}
})();
</script>

## 7.3 Molecular Dynamics and the Need for MLIPs

Given the PES, the classical nuclear motion is governed by Newton's second law. The force on nucleus $i$ is the negative gradient of the PES with respect to its position:

$$\mathbf{F}_i = -\frac{\partial V(\{\mathbf{r}\})}{\partial \mathbf{r}_i} = m_i\mathbf{a}_i.$$

Integrating these equations of motion numerically propagates the system through phase space and gives access to dynamical observables: diffusion coefficients, reaction rates, vibrational spectra, and transport properties. A standard and stable numerical integration scheme is the **Verlet integrator**, which updates positions using the previous and current positions and the current force:

$$\mathbf{r}_i(t+\Delta t) = 2\mathbf{r}_i(t) - \mathbf{r}_i(t-\Delta t) + \frac{\mathbf{F}_i(t)}{m_i}\Delta t^2.$$

The time step $\Delta t$ must be much smaller than the fastest nuclear motion (typically $\sim 1\,\mathrm{fs}$ for bond vibrations). A simulation of physically meaningful duration (nanoseconds to microseconds) therefore requires $10^6$ to $10^9$ force evaluations.

In **Ab Initio Molecular Dynamics (AIMD)**, each force evaluation requires solving the electronic Schrödinger equation from scratch at the current nuclear geometry. This is the gold standard for accuracy but is prohibitively expensive: roughly 99.99% of the compute time is spent on the quantum mechanics, and AIMD is typically limited to a few hundred atoms over a few tens of picoseconds. **Machine Learning Interatomic Potentials (MLIPs)** learn the mapping from nuclear geometry to energy and forces, replacing the quantum mechanical solver with a fast surrogate that achieves speedups of $10^3$–$10^6$ over AIMD while retaining near-DFT accuracy.

## 7.4 Empirical Force Fields

Before MLIPs, the standard approach to cheap force evaluation was empirical force fields (FFs): "charged balls connected with springs." The total energy is split into **short-range bonded** and **long-range non-bonded** contributions:

$$E_\text{total} = \underbrace{E_\text{bonds} + E_\text{angle} + E_\text{dihedral}}_{\text{bonded (short range)}} + \underbrace{E_\text{vdW} + E_\text{Coulomb}}_{\text{non-bonded (long range)}}$$

Expanding each term explicitly:

$$E_\text{total} = \underbrace{\sum_\text{bonds} K_r(r - r_\text{eq})^2 + \sum_\text{angles} K_\theta(\theta - \theta_\text{eq})^2 + \sum_\text{dihedrals} \frac{V_n}{2}[1 + \cos(n\phi - \gamma)]}_{\text{short range}}$$

$$+ \underbrace{\sum_{i<j}\!\left(\frac{A_{ij}}{R_{ij}^{12}} - \frac{B_{ij}}{R_{ij}^6}\right) + \sum_{i<j} \frac{q_i q_j}{\varepsilon R_{ij}}}_{\text{long range}}$$

The bonded terms encode increasingly complex geometry: bonds are **2-body** (depend on a single distance $r$), angles are **3-body** (depend on the angle $\theta$ between two bonds), and dihedrals are **4-body** (depend on the torsion angle $\phi$ around a bond). The non-bonded terms act between all pairs: the Lennard-Jones $r^{-12}$ captures short-range Pauli repulsion and $r^{-6}$ the van der Waals attraction, while the Coulomb term handles electrostatics between partial charges.

Standard FFs already have 3-body (angle) and 4-body (dihedral) terms, but these are defined along explicit *bonded* connections; they only apply between atoms that share a bond topology. This makes them unsuitable for metals, where bonding is delocalized and there is no fixed bond graph.

**Embedded Atom Model (EAM).** EAM handles metals by replacing the bond topology with a continuous local electron density: it introduces a nonlinear embedding function $F$ representing the energy of placing atom $i$ into a density $\rho_i$ built up from all neighbors within a cutoff, regardless of bonding:

$$E = \sum_i F(\rho_i) + \sum_{i<j} V(r_{ij}), \qquad \rho_i = \sum_{j \neq i} \phi_j(r_{ij}).$$

The nonlinearity of $F$ provides effective many-body character without needing a bond graph.

**Tersoff potential.** For covalent materials like silicon, the **Tersoff potential** addresses a different problem: silicon forms directional $sp^3$ bonds (tetrahedral geometry), and bond strength depends on coordination: a silicon atom with fewer neighbors forms stronger individual bonds than one surrounded by many. Tersoff captures this via a bond-order term $b_{ij}$ that modifies the attractive part of the potential:

$$E = \sum_{i<j} \left[V_R(r_{ij}) + b_{ij}\, V_A(r_{ij})\right],$$

where $V_R$ is a repulsive term and $V_A$ is an attractive term. The bond-order $b_{ij} < 1$ decreases as the local coordination increases, and it depends explicitly on the angles $\theta_{ijk}$ that neighboring bonds make with bond $(i,j)$. This makes Tersoff a true 3-body potential for non-metallic covalent systems, in contrast to EAM which achieves many-body character through the density $\rho_i$ without explicit angular terms.

Despite these extensions, empirical FFs share several concrete limitations. First, the bond topology is fixed at the start of a simulation; atoms connected by bonds stay connected, so bond breaking and formation (chemical reactions) are impossible. Second, angular and dihedral terms only fire along the bond graph: two atoms that are spatially close but not bonded have only an isotropic LJ/Coulomb interaction; their relative orientation is completely ignored. This means non-bonded angular correlations (e.g., the preferred angle between a water molecule and a nearby hydrophobic group) are entirely missing. Third, atomic charges are fixed; there is no charge redistribution as the electronic structure changes. Finally, parameters must be re-fitted for every new chemical system and often fail to transfer even between different phases of the same material (e.g., a water FF fitted to the liquid may not reproduce ice).

## 7.5 MLIP Core Concepts

The conceptual shift from force fields to MLIPs can be stated simply. Classical force fields sum over bonded interactions, tracking which pairs, triples, and quadruplets of atoms share a bond and accumulating separate contributions from each term. MLIPs instead sum over atomic environments: for each atom $i$, a descriptor of its local chemical environment within $R_\text{cut}$ enters a machine learning model that outputs $\varepsilon_i$. The bond topology disappears entirely, and the same model handles bond stretching, breaking, and formation because nothing is hard-coded about connectivity. This shift from topology-based to environment-based energy decomposition is what makes MLIPs transferable to new structures and compositions.

Every MLIP regardless of method follows the same three-stage pipeline: (1) compute an invariant local descriptor $\mathbf{q}_i$ for each atom from its neighborhood geometry, (2) pass $\mathbf{q}_i$ through a regression model to predict $\varepsilon_i$, and (3) sum atomic energies and differentiate to get forces. The methods in Sections 7.6--7.11 differ primarily in how steps (1) and (2) are implemented, as shown in the diagram below.

MLIPs replace the fixed analytical form with a flexible ML model, but retain four physically motivated design principles:

**Additivity.** The total energy is decomposed into a sum of atomic contributions $\varepsilon_i$:

$$E_\text{tot} = \sum_i \varepsilon_i.$$

This is an approximation (in reality atomic energies are not uniquely defined), but it is a very good one for short-range interactions and makes the model scale linearly with system size.

**Locality.** Each atomic energy depends only on the chemical environment within a cutoff radius $R_\text{cut}$:

$$\varepsilon_i = \varepsilon_i\!\left(\{r_{ij} : r_{ij} < R_\text{cut}\}\right).$$

Long-range effects (electrostatics, dispersion) require separate treatment, but for many covalent and metallic systems a cutoff of 4–6 Å captures the dominant interactions.

**Free atom reference.** Atomic energies are defined relative to the energy of the isolated atom of the same species, $\varepsilon_s^0$:

$$\tilde{\varepsilon}_i = \varepsilon_i - \varepsilon_s^0.$$

This removes the large species-dependent offset and makes the model transferable across different compositions: a model trained on carbon–hydrogen systems can predict $\tilde{\varepsilon}$ for a new C–H ratio without relearning baseline energies.

**Conservative forces.** Forces are never predicted directly; they are always obtained as the negative gradient of the predicted total energy:

$$\mathbf{F}_i = -\frac{\partial E_\text{tot}}{\partial \mathbf{r}_i}.$$

This guarantees energy conservation in MD (no energy drift) and ensures force equivariance automatically: if you rotate the structure, the forces rotate the same way because they are derived from a scalar energy.

### Development Timeline
| Era | Method | Key Feature |
|-----|--------|-------------|
| pre-2007 | Custom FFs / Tersoff / EAM | Analytical, limited transferability |
| >2007 | Behler-Parrinello NNPs | ACSFs + NNs per element |
| >2010 | GAP / Linear methods | SOAP+GPR, SNAP, MTP, ACE |
| >2018 | MPNNs | SchNet, DimeNet |
| >2021 | Equivariant GNNs | NequIP, MACE |
| >2023 | Foundation models | MACE-MP-0, UMA |

The diagram below shows the general anatomy shared by all MLIPs, and how the specific methods covered in Sections 7.6--7.11 instantiate each stage.

<div style="background:#faf8f5;border:1px solid #e0dbd4;border-radius:8px;padding:1.2rem 1.2rem 1rem;margin:1.2rem 0 1.8rem;">
<div style="text-align:center;font-size:0.84rem;font-weight:700;color:#555;margin-bottom:0.9rem;letter-spacing:0.01em;">General Anatomy of an MLIP</div>
<div style="text-align:center;margin-bottom:0.4rem;">
<span style="display:inline-block;background:#f0ece6;border:1.5px solid #c8c2ba;border-radius:6px;padding:0.3rem 1.3rem;font-size:0.82rem;font-weight:600;color:#555;">
Atomic positions &amp; species &nbsp;{<b>r</b><sub>i</sub>, Z<sub>i</sub>}
</span></div>
<div style="text-align:center;color:#bbb;font-size:1.15rem;line-height:1.2;margin:0.2rem 0;">&#8595;</div>
<div style="text-align:center;font-size:0.74rem;font-weight:700;color:#888;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:0.3rem;">(1) Invariant local environment descriptor <b>q</b><sub>i</sub></div>
<div style="display:flex;justify-content:center;gap:0.4rem;flex-wrap:wrap;margin-bottom:0.4rem;">
<span style="background:rgba(164,206,139,0.2);border:1px solid #A4CE8B;border-radius:4px;padding:0.22rem 0.55rem;font-size:0.77rem;">ACSF <span style="color:#888;">(BP)</span></span>
<span style="background:rgba(164,206,139,0.2);border:1px solid #A4CE8B;border-radius:4px;padding:0.22rem 0.55rem;font-size:0.77rem;">SOAP <span style="color:#888;">(GAP)</span></span>
<span style="background:rgba(134,188,189,0.2);border:1px solid #86BCBD;border-radius:4px;padding:0.22rem 0.55rem;font-size:0.77rem;"><i>B</i>-basis <span style="color:#888;">(ACE)</span></span>
<span style="background:rgba(186,90,90,0.12);border:1px solid #BA5A5A;border-radius:4px;padding:0.22rem 0.55rem;font-size:0.77rem;">Spherical harmonics + learned <span style="color:#888;">(MACE)</span></span>
</div>
<div style="text-align:center;color:#bbb;font-size:1.15rem;line-height:1.2;margin:0.2rem 0;">&#8595;</div>
<div style="text-align:center;font-size:0.74rem;font-weight:700;color:#888;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:0.3rem;">(2) ML regression: &epsilon;<sub>i</sub> = f(<b>q</b><sub>i</sub>)</div>
<div style="display:flex;justify-content:center;gap:0.4rem;flex-wrap:wrap;margin-bottom:0.4rem;">
<span style="background:rgba(164,206,139,0.2);border:1px solid #A4CE8B;border-radius:4px;padding:0.22rem 0.55rem;font-size:0.77rem;">MLP per element <span style="color:#888;">(BP)</span></span>
<span style="background:rgba(164,206,139,0.2);border:1px solid #A4CE8B;border-radius:4px;padding:0.22rem 0.55rem;font-size:0.77rem;">GPR / KRR <span style="color:#888;">(GAP)</span></span>
<span style="background:rgba(134,188,189,0.2);border:1px solid #86BCBD;border-radius:4px;padding:0.22rem 0.55rem;font-size:0.77rem;">Linear regression <span style="color:#888;">(ACE)</span></span>
<span style="background:rgba(186,90,90,0.12);border:1px solid #BA5A5A;border-radius:4px;padding:0.22rem 0.55rem;font-size:0.77rem;">Equivariant MPNN <span style="color:#888;">(MACE)</span></span>
</div>
<div style="text-align:center;color:#bbb;font-size:1.15rem;line-height:1.2;margin:0.2rem 0;">&#8595;</div>
<div style="text-align:center;margin-top:0.4rem;">
<span style="display:inline-block;background:#f0ece6;border:1.5px solid #c8c2ba;border-radius:6px;padding:0.3rem 1.3rem;font-size:0.82rem;font-weight:600;color:#555;">
E<sub>tot</sub> = &sum;<sub>i</sub>&thinsp;&epsilon;<sub>i</sub>&ensp;&nbsp;&ensp;<b>F</b><sub>i</sub> = &minus;&part;E<sub>tot</sub>/&part;<b>r</b><sub>i</sub>
</span></div>
<div style="display:flex;justify-content:center;gap:1.2rem;margin-top:0.7rem;font-size:0.73rem;color:#aaa;">
<span><span style="color:#A4CE8B;font-weight:600;">&#9632;</span> 2007&ndash;2017 era</span>
<span><span style="color:#86BCBD;font-weight:600;">&#9632;</span> 2019 (ACE)</span>
<span><span style="color:#BA5A5A;font-weight:600;">&#9632;</span> 2022+ equivariant</span>
</div>
</div>

## 7.6 Behler–Parrinello Neural Network Potentials

The first MLIP era to use neural networks was launched by Behler & Parrinello (*Phys. Rev. Lett.* **98**, 146401, 2007). The key idea: replace hand-crafted analytical functions with element-specific neural networks, but keep the hand-crafted descriptor step.

**Architecture.** For each atom $i$ of element $s$, compute a vector of **Atom-Centered Symmetry Functions (ACSFs)**, manually designed rotationally, translationally, and permutationally invariant functions of the local neighborhood. ACSFs come in two types: radial functions that encode the distribution of neighbor distances, and angular functions that encode the distribution of bond angles. These feed into an element-specific feedforward neural network (one NN per element type) that outputs the atomic energy contribution $\varepsilon_i$. The total energy is then $E = \sum_i \varepsilon_i$ as usual.

**What was new.** The regression step is now a neural network rather than a linear model (like ACE) or a kernel method (like GAP). This gives far greater flexibility to fit complex, high-dimensional PES regions, but the descriptor step (ACSFs) is still hand-crafted and fixed. Behler-Parrinello networks are thus classical ML in the sense that representation learning does not occur: the features are fixed, only the regression weights are learned.

**Limitation.** The quality of the potential is bottlenecked by the expressiveness of the ACSFs. Choosing good symmetry functions for a new system requires domain expertise. This motivated the move toward architectures that learn their own representations directly from atomic positions.

## 7.7 GAP (Gaussian Approximation Potentials)

GAP (Bartók et al., *Phys. Rev. Lett.* **104**, 136403, 2010) is not a new method on top of SOAP and GPR: it *is* SOAP combined with GPR, packaged as a complete MLIP. The three ingredients and how they connect:

**Descriptor (SOAP):** for each atom $i$, compute a rotationally and permutationally invariant descriptor $\mathbf{q}_i$ of its local chemical environment within a cutoff $r_\text{cut}$. The SOAP kernel $k(\mathbf{q}_i, \mathbf{q}_j)$ measures environment similarity.

**Regression (GPR/KRR):** treat the unknown atomic energy function as a Gaussian process with the SOAP kernel as covariance. Training on a dataset of $P$ DFT reference energies (and forces) reduces to solving a linear system $(K + \lambda I)\boldsymbol{\alpha} = \mathbf{y}$, exactly as in KRR ([Section 4.3](chapter-04-regression-uncertainty-quantification.md#43-kernel-ridge-regression-krr)). The solution $\boldsymbol{\alpha}$ gives the GPR weights; GPR additionally provides an analytic uncertainty $\sigma^2(\mathbf{q})$ at no extra cost ([Section 4.4](chapter-04-regression-uncertainty-quantification.md#44-gaussian-process-regression-gpr)).

**Prediction:** the energy of a new structure is a sum of atomic contributions,

$$E = \sum_i \varepsilon_i, \qquad \varepsilon_i = \sum_s \alpha_s\, k(\mathbf{q}_i, \mathbf{q}_s),$$

where the sum over $s$ runs over training (or sparse representative) points. Forces are obtained analytically as $F_i = -\partial E/\partial \mathbf{r}_i$, which requires differentiating the SOAP descriptor with respect to atomic positions.

GAP inherits all the properties of GPR: uncertainty estimates, exact kernel interpolation in the limit $\lambda \to 0$, and hyperparameter tuning via log-marginal likelihood. It also inherits the $O(P^3)$ scaling bottleneck of KRR; in practice, a sparse approximation (selecting $M \ll P$ representative environments) reduces this to $O(PM^2)$, but GAP remains limited to datasets of order $10^4$ structures, compared to $10^6$ for deep learning MLIPs like MACE.

## 7.8 ACE (Atomic Cluster Expansion as MLIP)

ACE (Drautz, *Phys. Rev. B* **99**, 014104, 2019) provides a systematic way to expand the atomic energy $\varepsilon_i$ in a complete, symmetry-adapted basis of many-body functions, without explicitly looping over all pairs, triplets, and quadruplets of neighbors as empirical FFs do.

**One-particle basis.** For each neighbor $j$ of atom $i$, define a one-particle basis function that encodes both distance and direction:

$$\phi_{nlm}(\mathbf{r}_{ij}) = R_n(r_{ij})\, Y_l^m(\hat{\mathbf{r}}_{ij}),$$

where $R_n$ are radial basis functions and $Y_l^m$ are spherical harmonics. Summing over all neighbors gives the **$A$-basis** (atomic density projection):

$$A_{i,nlm} = \sum_{j \in \mathcal{N}(i)} \phi_{nlm}(\mathbf{r}_{ij}).$$

This is permutation-invariant by construction; the sum treats all neighbors equally regardless of order.

**Many-body correlations via products.** A single $A_{i,nlm}$ captures only 1-body information (the smeared density of neighbors). To capture 2-body correlations (pairs), take a product of two $A$ terms; for 3-body correlations (triplets), a product of three, and so on:

$$\nu\text{-body}: \quad A_{i,\nu_1} \cdot A_{i,\nu_2} \cdots A_{i,\nu_\nu}.$$

Crucially, each $A$ is already a sum over all neighbors, so the product implicitly encodes correlations between all $\nu$-tuples of neighbors without explicitly enumerating them. This is far more efficient than the FF approach of looping over all pairs, all angles, all dihedrals separately.

**$B$-basis (symmetrized products).** The raw products $A^{\otimes\nu}$ still carry the magnetic quantum numbers $m$ from the spherical harmonics and are not yet rotationally invariant. Coupling the angular momentum indices using Clebsch–Gordan coefficients yields the rotationally and permutationally invariant **$B$-basis**:

$$B_{i,\mathbf{n}\mathbf{l}} = \sum_{\mathbf{m}} C_{\mathbf{lm}}\prod_{\xi=1}^\nu A_{i,n_\xi l_\xi m_\xi}.$$

**Linear model.** The atomic energy is then a linear combination of $B$ basis functions:

$$E_i = \sum_{\mathbf{n},\mathbf{l}} c_{\mathbf{n}\mathbf{l}}\, B_{i,\mathbf{n}\mathbf{l}},$$

which can be written compactly as

$$E = \sum_{i,v} c_v B_{iv}.$$

Because the model is linear in the coefficients $\{c_v\}$, fitting reduces to simple linear regression; no iterative gradient descent is required.

**Relation to GAP and MACE.** The SOAP power spectrum used in GAP is exactly the $\nu=2$ ACE basis: GAP is a nonlinear (GPR) model on top of 2-body ACE features. ACE generalizes this to arbitrary body order with a linear model. MACE takes ACE one step further: instead of fitting linear coefficients on the ACE basis, it learns nonlinear transformations of the ACE features through neural network layers and stacks multiple message-passing iterations to extend the effective receptive field beyond a single cutoff sphere.

## 7.9 Message-Passing MLIPs: SchNet

The 2018 generation of MLIPs replaced fixed descriptors with message-passing GNNs that learn their own atomic representations end-to-end. The prototype is **SchNet** (Schütt et al., *J. Chem. Phys.* **148**, 241722, 2018).

**Chemical species embedding.** Discrete element types (C, H, O, …) cannot be fed directly into a neural network. The standard solution is a two-step approach. First, each element is represented as a **one-hot vector** of length 119 (one entry per element in the periodic table); it is sparse, binary, and carries no notion of chemical similarity. This is then mapped to a dense, continuous **learned embedding** vector via a trainable linear layer $\mathbf{x}_i = W_\text{emb}\, \mathbf{z}_i$, where $\mathbf{z}_i$ is the one-hot vector. The embedding vectors are optimized end-to-end during training, so chemically similar elements naturally end up close in embedding space.

**Continuous filter convolution.** Unlike GCN (which uses fixed normalized averaging) SchNet introduces a **continuous filter convolution** (cfconv): the interaction between atoms $i$ and $j$ is weighted by a filter $W(r_{ij})$ that depends continuously on the interatomic distance via radial basis functions, not a discrete adjacency matrix entry. This lets the network smoothly interpolate interactions across all distances up to the cutoff. The interaction block computes:

$$\mathbf{x}_i^{(t+1)} = \mathbf{x}_i^{(t)} + \sum_{j \in \mathcal{N}(i)} W^{(t)}(r_{ij}) \odot \mathbf{x}_j^{(t)},$$

where $W^{(t)}(r_{ij})$ is a learned filter that expands the distance $r_{ij}$ onto radial basis functions and passes it through an MLP. After $T$ interaction blocks, atom-wise MLPs map the node embeddings to scalar energy contributions, which are summed to give $E$.

**What SchNet achieved.** By learning the representation rather than hand-crafting it, SchNet and the subsequent MPNNs (DimeNet, PaiNN, …) substantially outperformed Behler-Parrinello potentials, but they used only scalar (distance-based) features, discarding directional information. This limits their data efficiency and accuracy for systems with strong angular dependencies.

## 7.10 Euclidean Graph Neural Networks: NequIP

Scalar MPNNs like SchNet use only interatomic distances as edge features, discarding the direction $\hat{\mathbf{r}}_{ij}$. This makes them invariant under rotation by construction, but it also means they cannot directly encode angular information; they must learn it implicitly through many layers.

**Euclidean neural networks** (NequIP: Batzner et al., *Nat. Commun.* **13**, 2453, 2022) solve this by combining message-passing with **representation theory of the rotation group**. The key principle is that all operations in the network are built to commute with the action of the **Euclidean group E(3)**, the group of all rotations, translations, and inversions in three-dimensional space:

$$f(D(g)\mathbf{x},\, \omega) = D(g)\, f(\mathbf{x},\, \omega),$$

where $g \in \mathrm{E}(3)$ is a symmetry operation and $D(g)$ is its matrix representation (a rotation matrix for $g \in \mathrm{SO}(3)$, or combined with inversion for $\mathrm{O}(3)$). Node features are no longer plain vectors: they are collections of **steerable tensors**, indexed by angular momentum $l = 0, 1, 2, \ldots$: scalars ($l=0$), vectors ($l=1$), and higher-rank tensors ($l \geq 2$). Each channel transforms correctly under rotations via the Wigner-$D$ matrices.

**Edge features** are constructed from spherical harmonics $Y_l^m(\hat{\mathbf{r}}_{ij})$ evaluated on the unit direction vector, capturing angular information explicitly. **Tensor products** (implemented via Clebsch–Gordan coupling, handled by the e3nn library) mix features of different angular momenta while preserving equivariance. The result is a network that sees both distances and directions, respects all symmetries exactly by architecture, and is dramatically more data-efficient than scalar MPNNs; a single training structure in one orientation teaches the model about all orientations simultaneously.

## 7.11 MACE Architecture

MACE (Batatia et al., *NeurIPS* **35**, 11423, 2022) combines the equivariant message-passing framework of NequIP with the ACE product basis, giving a model that reaches high body order efficiently with few interaction layers. The overall pipeline is: **Embedding → [Interaction + Product + Update] × $S$ → Readout**, where the bracketed block is repeated $S$ times.

**Step 1: Embedding (computed once).** Three parallel initializations happen at the start:

At the *node level*, each atom $i$ of element type $m$ (an integer from 1 to 119) is initialized with a learned feature vector. The one-hot vector $\delta_{sm}$ is 1 when $s = m$ and 0 otherwise; it selects the column of the weight matrix $W$ corresponding to element $m$:

$$h_{i,k00}^{(0)} = \sum_{s=1}^{119} W_{ks}\,\delta_{sm} = W_{km}.$$

Here $k = 1,\ldots,K$ indexes the feature channel (how many parallel features each atom carries), and the subscript $00$ on $h$ denotes $l=0, m_l=0$: a scalar irreducible representation, meaning this initial embedding is rotationally invariant. In practice this equation just says: look up the $k$-th row of the $K \times 119$ weight matrix $W$ at column $m$; every atom of the same element starts with the same learned vector, and those vectors are updated during training. This is the same one-hot → embedding step as in SchNet and other MPNNs.

At the *edge level*, two quantities are computed for each pair $(i,j)$ within $R_\text{cut}$. The **radial embedding** expands the distance $r_{ij}$ onto $N_\text{max}$ sinc-like radial basis functions with a smooth cutoff envelope:

$$\tilde{f}_n(r_{ij}) = \sqrt{\frac{2}{r_\text{cut}}}\frac{\sin\!\left(\frac{n\pi r_{ij}}{r_\text{cut}}\right)}{r_{ij}}\,f_\text{cut}(r_{ij}).$$

The **angular embedding** encodes the direction $\hat{\mathbf{r}}_{ij}$ via spherical harmonics $Y_l^{m_l}(\hat{\mathbf{r}}_{ij})$ up to maximum order $L_\text{max}$. This is what NequIP added over SchNet/CGCNN: the radial part is shared, the angular part is the equivariant extension.

**Step 2: Interaction (pooling across neighbors).** Each interaction block has four sequential sub-steps.

*Linear channel mixing.* Every atom carries $K$ parallel channels, where each channel $k$ is one independent copy of the full irrep feature structure (one scalar at $l=0$, one vector at $l=1$, etc.). Think of $K$ as the width of the network: $K$ parallel streams that each develop their own representation of the local environment. Every atom has the same $K$ channels regardless of species; species information lives in the channel *values* set by the initial embedding, not in which channels exist. Before the tensor product, the $K$ streams are linearly mixed at fixed angular momentum $(l_2, m_2)$ via a learned $K \times K$ matrix:

$$\tilde{h}_{i,kl_2m_2}^{(s)} = \sum_{k'} W_{kk'l_2}^{(s)}\, h_{i,k'l_2m_2}^{(s)}.$$

This lets the model learn useful combinations of channels (e.g., a weighted mix of "coordination" and "local density" channels) before the expensive tensor product, at the cost of only a cheap matrix multiply. $W^{(s)}$ is learned per layer and per $l$.

*Radial MLP.* In parallel, the radial basis functions $\tilde{f}_n(r_{ij})$ for each edge are passed through a small MLP to produce learned radial weights:

$$R_{k\eta_1 l_1 l_2 l}^{(s)}(r_{ij}) = \text{MLP}\!\left(\{\tilde{f}_n(r_{ij})\}_n\right).$$

This is the generalization of SchNet's cfconv to equivariant features: instead of a fixed radial filter, the network learns how much weight to give each distance for each combination of angular momenta $(l_1, l_2 \to l)$.

*One-particle basis (tensor product).* The linearly mixed node features, learned radial weights, and spherical harmonics are combined via a CG tensor product (implemented in e3nn):

$$\varphi_{i,kl,m_l}^{(s)} = \sum_{l_1,l_2,m_1,m_2} C_{l_1m_1,l_2m_2}^{l,m_l}\, R_{k\eta_1 l_1 l_2 l}^{(s)}(r_{ij})\, \tilde{h}_{j,kl_1,m_1}^{(s)}\, Y_{l_2}^{m_2}(\hat{\mathbf{r}}_{ij}).$$

The CG coefficients $C_{l_1m_1,l_2m_2}^{l,m_l}$ are fixed numerical constants (not learned) that couple two angular momentum channels $(l_1, l_2)$ into a single output channel $l$, guaranteeing the result transforms correctly under rotation, which is what enforces equivariance. Crucially, $\varphi$ itself is **equivariant** (it transforms as an irrep of the rotation group), unlike the one-particle basis in ACE which is invariant (scalar). This means MACE carries directional tensor information through the network, not just scalar summaries; the $l>0$ channels propagate vector and higher-tensor features between layers, and only collapse to scalars at the readout step.

*Neighbor sum → $A$-basis.* Finally, $\varphi$ is summed over all neighbors and linearly mixed across channels to give the **$A$-basis**:

$$A_{i,kl,m_l}^{(s)} = \sum_{k'} W_{kk'}^{(s)}\sum_{j \in \mathcal{N}(i)} \varphi_{i,k'l,m_l}^{(s)}.$$

The sum over $j$ makes the result permutation invariant. In pure ACE, this neighbor sum is the entire story; the additional learned weights $W_{kk'}^{(s)}$ are what make MACE a representation *learning* model rather than a fixed basis expansion.

**Step 3: Product (higher body order).** The $A$-basis encodes 1-body correlations (single neighbors). MACE achieves $\nu$-body correlations by taking tensor products of $\nu$ copies of the $A$-basis with itself, coupled via CG coefficients to give the rotationally invariant **$B$-basis**:

$$B_{i,\eta,kLM}^{(s)} = \sum_{lm} C_{lm}^{LM}\prod_{\xi=1}^{\nu} A_{i,k_\xi l_\xi m_\xi}^{(s)}.$$

This is exactly the ACE $B$-basis construction from [Section 7.8](#78-ace-atomic-cluster-expansion-as-mlip), but computed inside a learned message-passing layer rather than over a fixed linear model. A learnable message is then formed as a linear combination over the $B$-basis channels:

$$m_{i,kLM}^{(s)} = \sum_{\eta,k'} W_{\eta,kk'}^{(s)}\, B_{i,\eta,k'LM}^{(s)},$$

and the node features are updated additively (combining the message with the previous features):

$$h_{i,kLM}^{(s+1)} = \sum_{k'} W_{kk',L}^{(s)}\, m_{i,k'LM}^{(s)} + \sum_{k'} W_{kk',L}^{(s)}\, h_{i,k'LM}^{(s)}.$$

The key advantage over NequIP ([§7.10](#710-euclidean-graph-neural-networks-nequip)): NequIP takes one tensor product per layer, effectively reaching body order $S+1$ after $S$ layers. MACE takes $\nu$ tensor products *within* each product block, reaching body order $(\nu+1) \times S + 1$, much higher body order for the same number of layers.

**Step 4: Readout.** After each interaction block $s$, the scalar channels ($l=0$) of the node features are read out to a per-atom energy contribution. For intermediate layers a linear readout is used; for the final layer ($s=S$) an MLP:

$$\mathcal{R}^{(s)}\!\left(h_i^{(s)}\right) = \begin{cases} \displaystyle\sum_k W_k^{(s)}\, h_{i,k00}^{(s)} & 1 \leq s < S \\ \text{MLP}\!\left(\left\{h_{i,k00}^{(S)}\right\}_k\right) & s = S \end{cases}$$

The total predicted atomic energy is

$$E_i = \sum_s \mathcal{R}^{(s)}\!\left(h_i^{(s)}\right).$$

Forces are obtained by automatic differentiation through the entire graph:

$$\mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{r}_i}.$$

**Training loss.** The model is trained jointly on energies and forces, normalized per atom and per force component:

$$\mathcal{L} = \frac{\lambda_E}{B}\sum_{b=1}^B\!\left(\frac{E_b - \hat{E}_b}{N_b}\right)^{\!2} + \frac{\lambda_F}{3B}\sum_{b=1}^B\sum_{i,\alpha}^{N_b,3}\!\left(-\frac{\partial E_b}{\partial r_{i,\alpha}} - \hat{F}_{i,\alpha}\right)^{\!2}.$$

Including forces in the loss (which give $3N$ additional training signals per structure) is the main reason GNN-based MLIPs are data-efficient.

**Key hyperparameters and their effect:**

**$R_\text{cut}$ (cutoff radius).** Determines the size of the local neighbourhood. Larger $R_\text{cut}$ means more neighbors per atom, more compute per interaction block, but access to longer-range interactions. Typical values: 4–6 Å for covalent/metallic systems.

**$L_\text{max}$ (maximum angular momentum order).** Controls the angular resolution of the edge features: how many spherical harmonic orders $l = 0, 1, \ldots, L_\text{max}$ are included. $L_\text{max}=1$ gives scalars + vectors; $L_\text{max}=2$ adds rank-2 tensors; $L_\text{max}=3$ adds rank-3 tensors. Higher $L_\text{max}$ captures more complex angular geometry but the CG tensor product cost grows roughly as $L_\text{max}^3$. In practice $L_\text{max}=2$ is the standard balance point.

**$N_\text{max}$ (radial basis size).** Number of sinc-like radial basis functions used to expand distances. Controls radial resolution. Typical value: 8–10.

**$S$ (number of interaction layers).** Controls the total field-of-view: each layer extends the receptive field by one $R_\text{cut}$ sphere, so after $S$ layers the model sees up to $R_\text{cut} \times S$. Almost always set to $S=2$ in practice; this already gives a 10 Å field-of-view for $R_\text{cut}=5$ Å, sufficient for most short-range systems.

**$K$ (channels per irrep).** The width of the network: how many parallel streams run through each irrep type. Typical values: 64–256. Doubling $K$ roughly doubles the parameter count and compute.

**$\nu$ (correlation order).** How many $A$-basis tensors are multiplied in the product block. Determines body order per layer; the total body order across all $S$ layers is $(\nu+1)\times S + 1$. $\nu=2$ for speed (triangles of atoms), $\nu=3$ for accuracy (tetrahedra). For $\nu=3, S=2$: total body order = $(3+1)\times 2 + 1 = 9$.

**e3nn irrep notation.** The feature space of each atom is described as a direct sum of irreducible representations using the notation `NxLp`, where $N$ is the number of channels, $L$ is the angular momentum order, and `p` is parity (`e` = even/gerade, `o` = odd/ungerade) under spatial inversion. Parity is an independent quantum number from $L$: it is *not* redundant with the angular momentum label. Raw spherical harmonics built from a displacement vector have the "natural" parity $(-1)^L$ ($l=0$ even, $l=1$ odd, $l=2$ even, …), which is why the very first vector features in the network are `1o`. However, once features are combined via Clebsch–Gordan tensor products, the parity of an output irrep is the *product* of the input parities, independent of which output $L$ it lands on: coupling two `1o` vectors decomposes into $L=0,1,2$, but every output carries parity $(-1)\times(-1)=+1$. The $L=1$ output is therefore `1e`, not `1o`: this is precisely the cross product of the two input vectors (a pseudovector), distinct from the ordinary `1o` (polar-vector) channel. Deeper layers of MACE/NequIP therefore carry both `1e` and `1o` (and potentially `0o`, `2o`, etc.) simultaneously; this is required to retain all information produced by the tensor product algebra, and it is what lets the network represent axial/pseudo geometric quantities (e.g. chirality-sensitive angular arrangements) in addition to ordinary polar ones. Each `NxLp` block contributes $N \times (2L+1)$ numbers to the feature vector per atom. Examples:

- `32x0e`: 32 scalar channels → $32 \times 1 = 32$ numbers per atom
- `32x1o`: 32 vector channels → $32 \times 3 = 96$ numbers per atom
- `32x2e`: 32 rank-2 tensor channels → $32 \times 5 = 160$ numbers per atom
- `32x0e + 32x1o`: scalars + vectors → $32 + 96 = 128$ numbers per atom ($L_\text{max}=1$)
- `32x0e + 32x1o + 32x2e`: scalars + vectors + rank-2 tensors → $32 + 96 + 160 = 288$ numbers per atom ($L_\text{max}=2$)

The initial node embedding $h_{i,k00}^{(0)}$ is `Kx0e` only: pure scalars, since no directional information has been seen yet. The $l>0$ channels are populated in the first interaction block when the spherical harmonics $Y_l^m(\hat{\mathbf{r}}_{ij})$ enter via the tensor product.

**Representative model configurations:**

| Model size | $R_\text{cut}$ | $L_\text{max}$ | $K$ | $S$ | $\nu$ | Feature irreps | Body order |
|---|---|---|---|---|---|---|---|
| Small (fast) | 5.0 Å | 1 | 128 | 2 | 2 | `128x0e + 128x1o` | 7 |
| Medium | 5.0 Å | 2 | 128 | 2 | 3 | `128x0e + 128x1o + 128x2e` | 9 |
| Large | 6.0 Å | 2 | 256 | 2 | 3 | `256x0e + 256x1o + 256x2e` | 9 |

The small model is suitable for fast MD on well-defined systems; the medium model is the typical research choice; the large model (similar to MACE-MP-0) is used for universal/foundation models trained on millions of structures across the periodic table.

<div id="mace-fc" style="background:#faf8f5;border:1px solid #e0dbd4;border-radius:8px;padding:1.1rem 0.9rem;margin:1.8rem 0;font-family:inherit;position:relative;padding-right:44px;">
<div style="text-align:center;font-size:0.84rem;font-weight:700;color:#444;margin-bottom:0.75rem;letter-spacing:0.01em;">MACE Architecture &mdash; Embedding &rarr; [Interaction + Product + Readout] &times; <em>S</em> &rarr; Output</div>

<div style="display:flex;justify-content:center;gap:0.7rem;margin-bottom:0.15rem;flex-wrap:wrap;">
<div style="background:#f0ece6;border:1.5px solid #c8c2ba;border-radius:5px;padding:0.22rem 0.85rem;font-size:0.8rem;font-weight:600;color:#555;">Species Z<sub>i</sub></div>
<div style="background:#f0ece6;border:1.5px solid #c8c2ba;border-radius:5px;padding:0.22rem 0.85rem;font-size:0.8rem;font-weight:600;color:#555;">Edges: r<sub>ij</sub>, &nbsp;r&#770;<sub>ij</sub></div>
</div>
<svg width="14" height="22" viewBox="0 0 14 22" style="display:block;margin:0.15rem auto;"><line x1="7" y1="0" x2="7" y2="13" stroke="#555" stroke-width="2"/><polygon points="2,11 7,20 12,11" fill="#555"/></svg>
<div style="border:1.5px solid #A4CE8B;border-radius:6px;overflow:hidden;margin:0.15rem 0;">
<div style="font-size:0.74rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;padding:0.28rem 0.7rem;background:rgba(164,206,139,0.12);color:#2a5a1a;border-bottom:1px solid #A4CE8B;">Embedding <span style="font-weight:400;text-transform:none;color:#888;">&middot; computed once</span></div>
<div style="display:flex;gap:0.4rem;padding:0.45rem;flex-wrap:wrap;">
<div id="mace-emb-node" style="flex:1;min-width:150px;background:#fff;border:1px solid #e5dfd7;border-left:3px solid #A4CE8B;border-radius:5px;padding:0.35rem 0.6rem;overflow-x:auto;">
<div style="font-size:0.69rem;font-weight:700;color:#999;margin-bottom:0.1rem;">Node <span style="color:#A4CE8B;">&#9679;</span> from species Z<sub>i</sub></div>
<span data-eq="h_{i,k00}^{(0)} = W_{km}" data-dm="1"></span></div>
<div id="mace-emb-radial" style="flex:1.2;min-width:200px;background:#fff;border:1px solid #e5dfd7;border-left:3px solid #86BCBD;border-radius:5px;padding:0.35rem 0.6rem;overflow-x:auto;">
<div style="font-size:0.69rem;font-weight:700;color:#999;margin-bottom:0.1rem;">Radial basis <span style="color:#86BCBD;">&#9679;</span> from edge distances r<sub>ij</sub></div>
<span data-eq="\tilde{f}_n(r_{ij}) = \sqrt{\tfrac{2}{r_c}}\,\frac{\sin\!\left(\frac{n\pi r_{ij}}{r_c}\right)}{r_{ij}}\,f_c(r_{ij})" data-dm="1"></span></div>
<div id="mace-emb-angular" style="flex:1;min-width:150px;background:#fff;border:1px solid #e5dfd7;border-left:3px solid #86BCBD;border-radius:5px;padding:0.35rem 0.6rem;overflow-x:auto;">
<div style="font-size:0.69rem;font-weight:700;color:#999;margin-bottom:0.1rem;">Angular basis <span style="color:#86BCBD;">&#9679;</span> from directions r&#770;<sub>ij</sub></div>
<span data-eq="Y_l^{m_l}(\hat{\mathbf{r}}_{ij}),\; l = 0,\ldots,L_{\max}" data-dm="1"></span></div>
</div>
</div>
<svg width="14" height="22" viewBox="0 0 14 22" style="display:block;margin:0.15rem auto;"><line x1="7" y1="0" x2="7" y2="13" stroke="#555" stroke-width="2"/><polygon points="2,11 7,20 12,11" fill="#555"/></svg>
<div style="border:1.5px dashed #86BCBD;border-radius:8px;padding:0.45rem;background:rgba(134,188,189,0.03);">
<div style="text-align:center;font-size:0.73rem;font-weight:700;color:#86BCBD;letter-spacing:0.03em;margin-bottom:0.3rem;">&#10227;&thinsp;&times;S&thinsp; interaction layers</div>

<div style="border:1.5px solid #86BCBD;border-radius:6px;overflow:hidden;margin-bottom:0.3rem;">
<div style="font-size:0.74rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;padding:0.28rem 0.7rem;background:rgba(134,188,189,0.12);color:#0f3a3c;border-bottom:1px solid #86BCBD;">Interaction <span style="font-weight:400;text-transform:none;color:#888;">&middot; equivariant message passing &rarr; A-basis</span></div>
<div style="padding:0.4rem 0.45rem;">
<div style="display:flex;gap:0.4rem;flex-wrap:wrap;">
<div id="mace-int-linear" style="flex:1;min-width:175px;background:#fff;border:1px solid #e5dfd7;border-left:3px solid #A4CE8B;border-radius:5px;padding:0.35rem 0.6rem;overflow-x:auto;">
<div style="font-size:0.69rem;font-weight:700;color:#999;margin-bottom:0.1rem;">1 &middot; Linear mix &nbsp;<span style="color:#A4CE8B;">&#9679;</span><span style="font-weight:400;"> node features h</span></div>
<span data-eq="\tilde{h}_{i,kl_2m_2}^{(s)} = \textstyle\sum_{k'} W_{kk'l_2}^{(s)}\,h_{i,k'l_2m_2}^{(s)}" data-dm="1"></span></div>
<div id="mace-int-radial" style="flex:1;min-width:175px;background:#fff;border:1px solid #e5dfd7;border-left:3px solid #86BCBD;border-radius:5px;padding:0.35rem 0.6rem;overflow-x:auto;">
<div style="font-size:0.69rem;font-weight:700;color:#999;margin-bottom:0.1rem;">2 &middot; Radial MLP &nbsp;<span style="color:#86BCBD;">&#9679;</span><span style="font-weight:400;"> edge features f&#771;</span></div>
<span data-eq="R_{kl_1l_2l}^{(s)}(r_{ij}) = \mathrm{MLP}\!\left(\{\tilde{f}_n(r_{ij})\}_n\right)" data-dm="1"></span></div>
</div>
<div style="display:flex;align-items:center;justify-content:center;gap:0.5rem;padding:0.18rem 0;font-size:0.69rem;color:#bbb;">
<span style="color:#A4CE8B;font-weight:600;">h&#771;</span><span>+</span><span style="color:#86BCBD;font-weight:600;">R</span><span>+</span><span style="color:#86BCBD;font-weight:600;">Y<sub>l</sub><sup>m</sup></span><span>&darr;</span>
</div>
<div id="mace-int-tp" style="background:#fff;border:1px solid #e5dfd7;border-radius:5px;padding:0.35rem 0.6rem;overflow-x:auto;margin:0.08rem 0;">
<div style="font-size:0.69rem;font-weight:700;color:#999;margin-bottom:0.1rem;">3 &middot; Tensor product (Clebsch&ndash;Gordan) &nbsp;<span style="font-weight:400;color:#bbb;">combines h&#771; + R + Y</span></div>
<span data-eq="\varphi_{i,klm}^{(s)} = \textstyle\sum_{l_1l_2m_1m_2} C_{l_1m_1,l_2m_2}^{l,m}\,R_{kl_1l_2l}^{(s)}\,\tilde{h}_{j,kl_1m_1}^{(s)}\,Y_{l_2}^{m_2}(\hat{\mathbf{r}}_{ij})" data-dm="1"></span></div>
<svg width="10" height="18" viewBox="0 0 10 18" style="display:block;margin:0.1rem auto;"><line x1="5" y1="0" x2="5" y2="11" stroke="#888" stroke-width="1.5"/><polygon points="1,9 5,17 9,9" fill="#888"/></svg>
<div style="background:#fff;border:1px solid #e5dfd7;border-radius:5px;padding:0.35rem 0.6rem;overflow-x:auto;margin:0.08rem 0;">
<div style="font-size:0.69rem;font-weight:700;color:#999;margin-bottom:0.1rem;">4 &middot; A-basis (neighbour sum + linear)</div>
<span data-eq="A_{i,klm}^{(s)} = \textstyle\sum_{k'} W_{kk'}^{(s)}\sum_{j\in\mathcal{N}(i)}\varphi_{i,k'lm}^{(s)}" data-dm="1"></span></div>
</div>
</div>
<svg width="14" height="22" viewBox="0 0 14 22" style="display:block;margin:0.15rem auto;"><line x1="7" y1="0" x2="7" y2="13" stroke="#555" stroke-width="2"/><polygon points="2,11 7,20 12,11" fill="#555"/></svg>
<div style="border:1.5px solid #C47070;border-radius:6px;overflow:hidden;margin-bottom:0.3rem;">
<div style="font-size:0.74rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;padding:0.28rem 0.7rem;background:rgba(186,90,90,0.08);color:#5a1a1a;border-bottom:1px solid #C47070;">Product <span style="font-weight:400;text-transform:none;color:#888;">&middot; &nu;-body ACE correlations &rarr; B-basis &rarr; update h</span></div>
<div style="padding:0.4rem 0.45rem;">
<div style="display:flex;gap:0.4rem;flex-wrap:wrap;">
<div style="flex:1;min-width:175px;background:#fff;border:1px solid #e5dfd7;border-radius:5px;padding:0.35rem 0.6rem;overflow-x:auto;">
<div style="font-size:0.69rem;font-weight:700;color:#999;margin-bottom:0.1rem;">B-basis &nbsp;<span style="font-weight:400;color:#bbb;">&nu;-fold product of A</span></div>
<span data-eq="B_{i,\eta,kLM}^{(s)} = \textstyle\sum_{lm} C_{lm}^{LM}\prod_{\xi=1}^{\nu} A_{i,k_\xi l_\xi m_\xi}^{(s)}" data-dm="1"></span></div>
<div style="flex:1;min-width:175px;background:#fff;border:1px solid #e5dfd7;border-radius:5px;padding:0.35rem 0.6rem;overflow-x:auto;">
<div style="font-size:0.69rem;font-weight:700;color:#999;margin-bottom:0.1rem;">Message &nbsp;<span style="font-weight:400;color:#bbb;">linear over B channels</span></div>
<span data-eq="m_{i,kLM}^{(s)} = \textstyle\sum_{\eta,k'} W_{\eta,kk'}^{(s)}\,B_{i,\eta,k'LM}^{(s)}" data-dm="1"></span></div>
</div>
<svg width="10" height="18" viewBox="0 0 10 18" style="display:block;margin:0.1rem auto;"><line x1="5" y1="0" x2="5" y2="11" stroke="#888" stroke-width="1.5"/><polygon points="1,9 5,17 9,9" fill="#888"/></svg>
<div style="background:#fff;border:1px solid #e5dfd7;border-radius:5px;padding:0.35rem 0.6rem;overflow-x:auto;margin:0.08rem 0;">
<div style="font-size:0.69rem;font-weight:700;color:#999;margin-bottom:0.1rem;">Update (residual) &nbsp;<span style="font-weight:400;color:#bbb;">m + skip from h<sup>(s)</sup></span></div>
<span data-eq="h_{i,kLM}^{(s+1)} = \textstyle\sum_{k'} W_{kk'L}^{(s)}\,m_{i,k'LM}^{(s)} + \sum_{k'} U_{kk'L}^{(s)}\,h_{i,k'LM}^{(s)}" data-dm="1"></span></div>
</div>
</div>
<svg width="14" height="22" viewBox="0 0 14 22" style="display:block;margin:0.15rem auto;"><line x1="7" y1="0" x2="7" y2="13" stroke="#555" stroke-width="2"/><polygon points="2,11 7,20 12,11" fill="#555"/></svg>
<div style="border:1.5px solid #c8ad3a;border-radius:6px;overflow:hidden;">
<div style="font-size:0.74rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;padding:0.28rem 0.7rem;background:rgba(247,228,155,0.25);color:#5a4c00;border-bottom:1px solid #c8ad3a;">Readout <span style="font-weight:400;text-transform:none;color:#888;">&middot; l=0 scalars &rarr; atomic energy &epsilon;<sub>i</sub><sup>(s)</sup></span></div>
<div style="display:flex;gap:0.4rem;padding:0.4rem 0.45rem;flex-wrap:wrap;">
<div style="flex:1;min-width:155px;background:#fff;border:1px solid #e5dfd7;border-radius:5px;padding:0.35rem 0.6rem;overflow-x:auto;">
<div style="font-size:0.69rem;font-weight:700;color:#999;margin-bottom:0.1rem;">s &lt; S &nbsp;(linear)</div>
<span data-eq="\varepsilon_i^{(s)} = \textstyle\sum_k W_k^{(s)}\,h_{i,k00}^{(s)}" data-dm="1"></span></div>
<div style="flex:1;min-width:155px;background:#fff;border:1px solid #e5dfd7;border-radius:5px;padding:0.35rem 0.6rem;overflow-x:auto;">
<div style="font-size:0.69rem;font-weight:700;color:#999;margin-bottom:0.1rem;">s = S &nbsp;(MLP)</div>
<span data-eq="\varepsilon_i^{(S)} = \mathrm{MLP}\!\left(\{h_{i,k00}^{(S)}\}_k\right)" data-dm="1"></span></div>
</div>
</div>

</div>
<svg width="14" height="22" viewBox="0 0 14 22" style="display:block;margin:0.15rem auto;"><line x1="7" y1="0" x2="7" y2="13" stroke="#555" stroke-width="2"/><polygon points="2,11 7,20 12,11" fill="#555"/></svg>
<div style="border:1.5px solid #aaa;border-radius:6px;overflow:hidden;margin:0.15rem 0;">
<div style="font-size:0.74rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;padding:0.28rem 0.7rem;background:rgba(170,170,170,0.1);color:#444;border-bottom:1px solid #ccc;">Output <span style="font-weight:400;text-transform:none;color:#888;">&middot; sum over atoms &amp; layers; forces via autodiff</span></div>
<div style="padding:0.4rem 0.45rem;">
<div style="display:flex;gap:0.4rem;flex-wrap:wrap;margin-bottom:0.3rem;">
<div style="flex:1;min-width:175px;background:#fff;border:1px solid #e5dfd7;border-radius:5px;padding:0.35rem 0.6rem;overflow-x:auto;">
<div style="font-size:0.69rem;font-weight:700;color:#999;margin-bottom:0.1rem;">Total energy</div>
<span data-eq="E_i = \textstyle\sum_s \varepsilon_i^{(s)},\quad E_{\mathrm{tot}} = \sum_i E_i" data-dm="1"></span></div>
<div style="flex:1;min-width:175px;background:#fff;border:1px solid #e5dfd7;border-radius:5px;padding:0.35rem 0.6rem;overflow-x:auto;">
<div style="font-size:0.69rem;font-weight:700;color:#999;margin-bottom:0.1rem;">Forces (conservative)</div>
<span data-eq="\mathbf{F}_i = -\dfrac{\partial E_{\mathrm{tot}}}{\partial\mathbf{r}_i}" data-dm="1"></span></div>
</div>
<div style="background:#fff;border:1px solid #e5dfd7;border-radius:5px;padding:0.35rem 0.6rem;overflow-x:auto;">
<div style="font-size:0.69rem;font-weight:700;color:#999;margin-bottom:0.1rem;">Training loss</div>
<span data-eq="\mathcal{L} = \frac{\lambda_E}{B}\sum_b\!\left(\frac{E_b-\hat{E}_b}{N_b}\right)^{\!2} + \frac{\lambda_F}{3B}\sum_{b,i,\alpha}\!\left(\hat{F}_{i\alpha}+\frac{\partial E_b}{\partial r_{i\alpha}}\right)^{\!2}" data-dm="1"></span></div>
</div>
</div>

<!-- JS-drawn connecting arrows overlay -->
<svg id="mace-arrows" style="position:absolute;top:0;left:0;pointer-events:none;z-index:5;overflow:visible;"></svg>
</div>
<script>
(function(){
var NS='http://www.w3.org/2000/svg';

function drawMaceArrows(){
  var widget=document.getElementById('mace-fc');
  var asvg=document.getElementById('mace-arrows');
  if(!widget||!asvg)return;
  var W=widget.offsetWidth,H=widget.offsetHeight;
  asvg.setAttribute('width',W);
  asvg.setAttribute('height',H);
  asvg.setAttribute('viewBox','0 0 '+W+' '+H);
  var wRect=widget.getBoundingClientRect();
  asvg.innerHTML='<defs>'+
    '<marker id="mah-g" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L8,4 L0,8 Z" fill="#3a7a28"/></marker>'+
    '<marker id="mah-b" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L8,4 L0,8 Z" fill="#1f6668"/></marker>'+
    '</defs>';
  function rel(id){
    var el=document.getElementById(id);if(!el)return null;
    var r=el.getBoundingClientRect();
    return{left:r.left-wRect.left,top:r.top-wRect.top,right:r.right-wRect.left,
           bottom:r.bottom-wRect.top,cx:r.left+r.width/2-wRect.left,cy:r.top+r.height/2-wRect.top};
  }
  function bezier(x1,y1,x2,y2,col,mid){
    var dy=y2-y1;
    var el=document.createElementNS(NS,'path');
    el.setAttribute('d','M'+x1+','+y1+' C'+x1+','+(y1+dy*0.45)+' '+x2+','+(y1+dy*0.55)+' '+x2+','+y2);
    el.setAttribute('fill','none');el.setAttribute('stroke',col);
    el.setAttribute('stroke-width','1.8');el.setAttribute('marker-end','url(#'+mid+')');
    asvg.appendChild(el);
  }
  // 1. Node (embedding) -> Linear mix (interaction)  [green]
  var en=rel('mace-emb-node'),il=rel('mace-int-linear');
  if(en&&il)bezier(en.cx,en.bottom,il.cx,il.top,'#3a7a28','mah-g');
  // 2. Radial (embedding) -> Radial MLP (interaction)  [blue]
  var er=rel('mace-emb-radial'),ir=rel('mace-int-radial');
  if(er&&ir)bezier(er.cx,er.bottom,ir.cx,ir.top,'#1f6668','mah-b');
  // 3. Angular (embedding) -> Tensor product  [blue bypass, right side]
  var ea=rel('mace-emb-angular'),tp=rel('mace-int-tp');
  if(ea&&tp){
    var rx=W-6,ry=8;
    var sx=ea.right+2,sy=ea.cy;   // exit: right side of Angular box
    var ex=tp.right+2,ey=tp.cy;   // entry: right side of Tensor product box
    var d='M'+sx+','+sy+
      ' L'+(rx-ry)+','+sy+
      ' Q'+rx+','+sy+' '+rx+','+(sy+ry)+
      ' L'+rx+','+(ey-ry)+
      ' Q'+rx+','+ey+' '+(rx-ry)+','+ey+
      ' L'+ex+','+ey;
    var el=document.createElementNS(NS,'path');
    el.setAttribute('d',d);el.setAttribute('fill','none');
    el.setAttribute('stroke','#1f6668');el.setAttribute('stroke-width','1.8');
    el.setAttribute('marker-end','url(#mah-b)');
    asvg.appendChild(el);
    // small rotated label along the right rail
    var lbl=document.createElementNS(NS,'text');
    var my=(sy+ey)/2,mx=rx+3;
    lbl.setAttribute('x',mx);lbl.setAttribute('y',my);
    lbl.setAttribute('font-size','7.5');lbl.setAttribute('fill','#1f6668');
    lbl.setAttribute('font-style','italic');
    lbl.setAttribute('transform','rotate(90,'+mx+','+my+')');
    lbl.textContent='Y edge attrs.';
    asvg.appendChild(lbl);
  }
}

function maceInit(){
  var el=document.getElementById('mace-fc');
  if(!el||el.dataset.miInit)return;
  el.dataset.miInit='1';
  el.querySelectorAll('[data-eq]').forEach(function(e){
    try{
      var eq=e.getAttribute('data-eq'),dm=e.getAttribute('data-dm')==='1';
      e.innerHTML=katex.renderToString(eq,{throwOnError:false,displayMode:dm});
    }catch(err){}
  });
  setTimeout(drawMaceArrows,200);
}

window.addEventListener('resize',function(){drawMaceArrows();});
if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',maceInit);}
else{maceInit();}
if(typeof document$!=='undefined'){document$.subscribe(function(){setTimeout(function(){maceInit();setTimeout(drawMaceArrows,250);},80);});}
})();
</script>

---

