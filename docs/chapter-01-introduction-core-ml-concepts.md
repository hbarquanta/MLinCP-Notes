# Chapter 1 — Introduction & Core ML Concepts

Machine learning (ML) is the discipline of constructing algorithms that learn patterns from data, rather than relying on explicitly programmed rules. Science has evolved through four broad paradigms: empirical observation (experiments, 1600s), theoretical science (physical laws and equations), computational science (numerical simulation, 1950s), and the current **fourth paradigm** of data-driven, AI-accelerated discovery (2000s onwards). In each transition, the new paradigm complemented rather than replaced its predecessors. In 2024, the Nobel Prize in Physics was awarded to John Hopfield and Geoffrey Hinton for foundational discoveries enabling machine learning with artificial neural networks — recognition of how central ML has become to science. In computational physics and chemistry, ML provides a route to building fast, accurate surrogate models for expensive quantum-mechanical calculations.

## 1.1 Learning Paradigms

Consider an unseen process (the data-generating process) that connects an input space $\mathcal{X}$ to a target space $\mathcal{Y}$, described by some function $f: \mathcal{X} \to \mathcal{Y}$.

Statistical samples of $\mathcal{X}$ are called **features** (also: attributes, input variables): $x_i \in \mathcal{X}$ are vectors of dimension $D$ containing reals, integers, or other values. Samples of $\mathcal{Y}$ are called **labels** (also: classes, targets): $y_i \in \mathcal{Y}$ can be scalars, vectors, or tensors. The index $i$ runs over individual examples.

A **dataset** $T$ can be **labeled**, $T = \{(x_i, y_i)\}$, or **unlabeled**, $T = \{x_i\}$.

**Supervised learning** requires a labeled dataset $T = \{(x_i, y_i)\}_{i=1}^n$ of $n$ examples. The goal is to find a model $\hat{f}: \mathcal{X} \to \mathcal{Y}$ that approximates $f$, i.e. $\hat{f}(x_i) = \hat{y}_i \approx y_i$, and generalizes well to new, unseen inputs.

**Unsupervised learning** works on unlabeled data $T = \{x_i\}_{i=1}^n$. The goal is to describe or understand the structure of the data — typical tasks include dimensionality reduction, clustering, outlier detection, and generative modelling (covered in Chapter 3 and Chapter 9).

**Semi-supervised learning** combines a small labeled set $T_L = \{(x_i, y_i)\}$ with a much larger unlabeled pool $T_U = \{x_j\}$. It is used when labeling is expensive (requiring expert annotation of medical images, MRI scans, or costly quantum-mechanical simulation) but unlabeled data is abundant. The unlabeled examples constrain the model by revealing structure in the input distribution, reducing the labeling burden. Applications include generative models trained on unlabeled data (see Chapter 9) and self-supervised pre-training followed by supervised fine-tuning.

**Reinforcement learning (RL)** operates without a static dataset. An *agent* performs actions in an *environment* at each time step in a sequential decision-making framework. Actions change the *state* and can provide positive or negative *feedback*; the goal is to learn a *policy* that maximizes cumulative *reward*. Unlike supervised learning, RL uses rewards instead of labels to learn. Two defining challenges are the **temporal credit assignment problem** (associating a delayed reward with the action that produced it) and the **exploitation vs. exploration** trade-off (acting on current best knowledge versus trying untested actions to gain information).

The standard mathematical framework is the **Markov decision process (MDP)**. A **Markov process** is a sequence of states $s_0, s_1, s_2, \ldots$ governed by a **transition probability** $P_r(s_{t+1}|s_t)$. Adding scalar rewards $r_t$ gives a **Markov reward process**, in which the *return* is the total discounted sum of future rewards:

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k},$$

where $\gamma \in (0,1]$ discounts future rewards. A **Markov decision process** extends the reward process with actions $a_t$: the agent selects an action that influences both the transition probability and the reward. The **policy** $\pi[a|s]$ determines which action to take in each state and can be stochastic or deterministic, stationary or time-dependent.

Given policy $\pi$, two value functions capture expected return. The **state value function** $v[s_t|\pi]$ is the expected return starting from state $s_t$:

$$v[s_t|\pi] = \mathbb{E}_\pi[G_t \mid s_t]$$

The **action value function** $q[s_t, a_t|\pi]$ is the expected return for taking action $a_t$ in state $s_t$:

$$q[s_t, a_t|\pi] = \mathbb{E}_\pi[G_t \mid s_t, a_t]$$

**Bellman equations** define optimal value functions recursively and are the basis for deriving optimal policies. Three practical algorithm families are commonly used: **tabular RL** (exact value-function computation on discrete, enumerable spaces), **fitted Q-learning** (approximates $q[s,a|\pi]$ with a neural network — used in Deep Q-Networks), and **policy gradient methods** (optimize $\pi$ directly by gradient ascent on expected return — used in PPO and the RLHF fine-tuning step of AlphaFold 2). Applications in computational science include adaptive sampling, molecular generation, and active learning of potential energy surfaces.

When $f$ is unknown, the ML process of finding the optimal $\hat{f}$ from a set $F$ is called **inductive learning**. The term reflects a philosophical point: inductions from finite data are never logically certain — they can at best be *useful*. When $f$ is already known but computationally expensive to evaluate, one instead seeks a model $\hat{f}$ that reproduces $f$ much more cheaply; this is called **speed-up learning**, and $\hat{f}$ is called a **surrogate model**. Surrogate models for expensive quantum-mechanical calculations (DFT, CCSD) are the central application throughout this course. An ML model is a **universal approximator** if its accuracy improves with increasing amounts of training data under some performance measure — the model class is rich enough to approach any target $f$ given sufficient data.

The fundamental requirement in every paradigm is **generalization**: the model must perform well on data it has not seen during training, not merely memorize the training set. A model that achieves this is said to generalize.

## 1.2 Risk Minimization and Loss Functions

The quality of a model $\hat{f}$ is measured by a **loss function** $\ell(\hat{f}(x), y)$ that quantifies the discrepancy between the prediction $\hat{f}(x)$ and the true label $y$.

The **expected risk** is the expectation of the loss over the true (unknown) data distribution $\mathcal{P}(\mathcal{X}, \mathcal{Y})$:

$$R(\hat{f}) = \langle \ell(\hat{f}(\mathcal{X}), \mathcal{Y}) \rangle = \int \ell\!\left(\hat{f}(\mathcal{X}), \mathcal{Y}\right) \mathrm{d}\mathcal{P}(\mathcal{X}, \mathcal{Y})$$

Since $\mathcal{P}$ is unknown, the expected risk is approximated by the **empirical risk** — the mean loss over the finite training set $T$:

$$R_\mathrm{emp}(\hat{f}) = \frac{1}{n} \sum_{i=1}^{n} \ell\!\left(\hat{f}(x_i),\, y_i\right)$$

where $n = |T|$ is the number of training examples. The ML training process is the process of minimizing $R_\mathrm{emp}$.

**Common loss functions for regression** (continuous $y_i$):

L1 loss (least absolute deviations, more robust to outliers):

$$\ell(\hat{f}(x_i), y_i) = \sum_i \left|\hat{f}(x_i) - y_i\right|$$

L2 loss (least squares, standard for regression):

$$\ell(\hat{f}(x_i), y_i) = \sum_i \left(\hat{f}(x_i) - y_i\right)^2$$

The L2 loss is easier to optimize (smooth gradient) but more sensitive to outliers than L1.

## 1.3 Gradient-Based Optimization

Model parameters are optimized iteratively via gradient descent. For a linear model $z = w \cdot x + b$ with weight vector $w$ and scalar bias $b$, the parameter update at step $t$ is:

$$w_t \leftarrow w_{t-1} - \alpha \cdot \nabla_{w_{t-1}} \ell(w, y, z)$$

where $\alpha > 0$ is the **learning rate**. The gradient $\nabla_w \ell$ is propagated backwards through the computational graph using automatic differentiation.

**Stochastic Gradient Descent (SGD)** approximates the full gradient using a random **mini-batch** $B \ll M$ of training examples, where $M$ is the total training set size. The full dataset is divided into $N = M / n_\text{batch}$ mini-batches; one pass over all mini-batches is called an **epoch**.

**Adam** (Adaptive Moment Estimation, Kingma & Ba, 2015) maintains per-parameter adaptive learning rates by tracking the first moment (mean) and second moment (uncentered variance) of the gradients. At step $t$, with gradient $g_t = \nabla_{w_{t-1}} \ell$:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad \text{(first moment)}$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \qquad \text{(second moment)}$$

Bias-corrected estimates:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Parameter update:

$$w_{n+1} \leftarrow w_n - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \varepsilon)$$

Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$.

## 1.4 Overfitting, Regularization, and Model Validation

**Underfitting (high bias)** occurs when the model is too simple to capture the structure of the data — it achieves high loss on both training and test data.

**Overfitting (high variance)** occurs when the model is too complex and fits noise in the training data, achieving low training loss but high test loss. Overfitting yields an increased error on unseen data by approximating a simple functional relationship with an overly complex function.

**Regularization** discourages overfitting by adding a complexity penalty to the loss:

$$\ell_\text{reg}(\hat{f}) = \ell(\hat{f}) + \lambda\, R(\hat{f})$$

where $\lambda > 0$ controls the regularization strength. Common choices: $R = \|w\|_2^2$ (L2/Ridge, penalizes large weights) and $R = \|w\|_1$ (L1/Lasso, promotes sparsity).

The widget below fits a degree-$d$ polynomial to noisy samples of $f(x) = \sin(2\pi x)$ and predicts $\hat{f}(x)$ at all $x \in [0,1]$ (the red curve). The right panel tracks train and test MSE across all degrees up to 20. Use the sliders to see how dataset size and the train/test split shift the overfitting profile, and resample to draw a fresh noise realization.

<div id="overfit-widget" style="margin:1.5rem 0 3.5rem 0;">
  <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(195px,1fr));gap:0.45rem 1.1rem;align-items:center;margin-bottom:0.85rem;font-size:0.91rem;">
    <div style="display:flex;align-items:center;gap:0.5rem;">
      <span style="font-weight:600;white-space:nowrap;">Degree</span>
      <input type="range" id="ov-deg" min="1" max="20" value="3" step="1"
             oninput="ovUpdate()" style="width:110px;accent-color:#BA5A5A;">
      <span id="ov-deg-val" style="font-weight:700;min-width:1.6em;color:#BA5A5A;">3</span>
    </div>
    <div style="display:flex;align-items:center;gap:0.5rem;">
      <span style="font-weight:600;white-space:nowrap;">Data points</span>
      <input type="range" id="ov-n" min="10" max="80" value="20" step="5"
             oninput="ovUpdate()" style="width:110px;accent-color:#86BCBD;">
      <span id="ov-n-val" style="font-weight:700;min-width:1.6em;color:#86BCBD;">20</span>
    </div>
    <div style="display:flex;align-items:center;gap:0.5rem;">
      <span style="font-weight:600;white-space:nowrap;">Train %</span>
      <input type="range" id="ov-frac" min="50" max="90" value="75" step="5"
             oninput="ovUpdate()" style="width:110px;accent-color:#A4CE8B;">
      <span id="ov-frac-val" style="font-weight:700;min-width:2.8em;color:#A4CE8B;">75 %</span>
    </div>
    <button onclick="ovResample()" style="padding:0.3rem 0.9rem;border:1.5px solid #F7E49B;border-radius:6px;background:rgba(247,228,155,0.15);color:inherit;cursor:pointer;font-size:0.88rem;font-weight:600;">&#x27F3; Resample</button>
  </div>
  <div id="ov-plot" style="width:100%;height:460px;"></div>
</div>

<script>
(function(){
  var _seed=42, _deg=3, _N=20, _trainPct=75;

  function _lcgNext(s){return(Math.imul(1664525,s)+1013904223)>>>0;}
  function _lcgFloat(s){return _lcgNext(s)/4294967296;}

  function _genData(){
    var step=Math.max(2,Math.round(100/(100-_trainPct)));
    var xTr=[],yTr=[],xTe=[],yTe=[],s=_seed;
    for(var i=0;i<_N;i++){
      s=_lcgNext(s);
      var x=i/Math.max(_N-1,1);
      var y=Math.sin(2*Math.PI*x)+(_lcgFloat(s)-0.5)*0.6;
      if(i%step===0){xTe.push(x);yTe.push(y);}else{xTr.push(x);yTr.push(y);}
    }
    return{xTr:xTr,yTr:yTr,xTe:xTe,yTe:yTe};
  }

  function _vand(xs,deg){
    return xs.map(function(x){
      var row=[],xn=2*x-1,xp=1;
      for(var j=0;j<=deg;j++){row.push(xp);xp*=xn;}
      return row;
    });
  }
  function _solve(M,b){
    var n=b.length,a=M.map(function(r,i){return r.slice().concat([b[i]]);});
    for(var col=0;col<n;col++){
      var mx=col;
      for(var row=col+1;row<n;row++)if(Math.abs(a[row][col])>Math.abs(a[mx][col]))mx=row;
      var tmp=a[col];a[col]=a[mx];a[mx]=tmp;
      if(Math.abs(a[col][col])<1e-14)continue;
      for(var row=col+1;row<n;row++){
        var f=a[row][col]/a[col][col];
        for(var j=col;j<=n;j++)a[row][j]-=f*a[col][j];
      }
    }
    var x=new Array(n);
    for(var i=n-1;i>=0;i--){
      x[i]=a[i][n];
      for(var j=i+1;j<n;j++)x[i]-=a[i][j]*x[j];
      x[i]/=a[i][i];
    }
    return x;
  }
  function _fit(xs,ys,deg){
    var V=_vand(xs,deg);
    var Vt=V[0].map(function(_,c){return V.map(function(r){return r[c];});});
    var AtA=Vt.map(function(r){return Vt.map(function(c){return r.reduce(function(s,v,i){return s+v*c[i];},0);});});
    var Atb=Vt.map(function(r){return r.reduce(function(s,v,i){return s+v*ys[i];},0);});
    return _solve(AtA,Atb);
  }
  function _eval(c,x){
    var xn=2*x-1,val=0,xp=1;
    c.forEach(function(ci){val+=ci*xp;xp*=xn;});
    return val;
  }
  function _mse(c,xs,ys){
    return xs.reduce(function(s,x,i){var d=_eval(c,x)-ys[i];return s+d*d;},0)/xs.length;
  }
  function _dark(){return document.body&&document.body.getAttribute('data-md-color-scheme')==='slate';}

  function _draw(){
    var el=document.getElementById('ov-plot');
    if(!el||!window.Plotly)return;
    var data=_genData();
    var xTr=data.xTr,yTr=data.yTr,xTe=data.xTe,yTe=data.yTe;
    if(!xTr.length)return;
    var maxDeg=Math.min(_deg,xTr.length-1);

    var coef=null;
    try{coef=_fit(xTr,yTr,maxDeg);}catch(e){return;}

    var xL=Array.from({length:200},function(_,i){return i/199;});
    var yL=xL.map(function(x){return _eval(coef,x);});
    var yTrue=xL.map(function(x){return Math.sin(2*Math.PI*x);});

    var maxPlotDeg=Math.min(20,xTr.length-1);
    var degs=[],errTr=[],errTe=[];
    for(var d=1;d<=maxPlotDeg;d++){
      try{
        var c=_fit(xTr,yTr,d);
        var mTr=_mse(c,xTr,yTr),mTe=_mse(c,xTe,yTe);
        if(isFinite(mTr)&&isFinite(mTe)){degs.push(d);errTr.push(mTr);errTe.push(mTe);}
      }catch(e){}
    }
    var eMax=(errTe.length?Math.max.apply(null,errTe):1)*1.3;
    eMax=Math.max(eMax,0.05);

    var dk=_dark(),bg=dk?'#1e2228':'#ffffff',fg=dk?'#e0e0e0':'#333333';
    var gc=dk?'rgba(255,255,255,0.07)':'rgba(0,0,0,0.07)';
    var label=_deg<=2?'underfitting':_deg>=7?'overfitting':'good fit';
    var fs=14;

    Plotly.react('ov-plot',[
      {x:xL,y:yTrue,mode:'lines',name:'true f(x)',
       line:{color:'#86BCBD',width:1.5,dash:'dash'},xaxis:'x',yaxis:'y'},
      {x:xL,y:yL,mode:'lines',name:'degree '+maxDeg+' fit',
       line:{color:'#BA5A5A',width:2.5},xaxis:'x',yaxis:'y'},
      {x:xTr,y:yTr,mode:'markers',name:'train ('+xTr.length+')',
       marker:{color:'#A4CE8B',size:7,line:{color:'rgba(0,0,0,0.25)',width:1}},xaxis:'x',yaxis:'y'},
      {x:xTe,y:yTe,mode:'markers',name:'test ('+xTe.length+')',
       marker:{symbol:'diamond',color:'#F7E49B',size:8,line:{color:'rgba(0,0,0,0.4)',width:1}},xaxis:'x',yaxis:'y'},
      {x:degs,y:errTr,mode:'lines+markers',name:'train MSE',
       line:{color:'#A4CE8B',width:2},marker:{color:'#A4CE8B',size:5},xaxis:'x2',yaxis:'y2'},
      {x:degs,y:errTe,mode:'lines+markers',name:'test MSE',
       line:{color:'#BA5A5A',width:2},marker:{color:'#BA5A5A',size:5},xaxis:'x2',yaxis:'y2'},
      {x:[maxDeg,maxDeg],y:[0,eMax],mode:'lines',showlegend:false,
       line:{color:'#86BCBD',width:1.5,dash:'dot'},xaxis:'x2',yaxis:'y2'},
    ],{
      paper_bgcolor:bg,plot_bgcolor:bg,
      font:{color:fg,family:'inherit',size:fs},
      margin:{t:50,b:58,l:62,r:68},
      annotations:[
        {text:'Polynomial fit to sin(2π<i>x</i>) + noise — <b>'+label+'</b>',
         xref:'paper',yref:'paper',x:0.21,y:1.06,showarrow:false,font:{size:13,color:fg}},
        {text:'Train vs. test MSE by degree',
         xref:'paper',yref:'paper',x:0.79,y:1.06,showarrow:false,font:{size:13,color:fg}},
      ],
      xaxis:{title:{text:'<i>x</i>',font:{size:fs+1}},range:[-0.02,1.02],domain:[0,0.43],
             gridcolor:gc,zerolinecolor:gc,tickfont:{size:fs-1}},
      yaxis:{title:{text:'<i>y</i>',font:{size:fs+1}},range:[-2.5,2.5],
             gridcolor:gc,zerolinecolor:gc,tickfont:{size:fs-1}},
      xaxis2:{title:{text:'Polynomial degree',font:{size:fs+1}},
              range:[0.5,maxPlotDeg+0.5],dtick:Math.max(1,Math.floor(maxPlotDeg/10)),
              domain:[0.57,1],gridcolor:gc,zerolinecolor:gc,tickfont:{size:fs-1}},
      yaxis2:{title:{text:'MSE',font:{size:fs+1}},range:[0,eMax],
              anchor:'x2',side:'right',gridcolor:gc,zerolinecolor:gc,tickfont:{size:fs-1}},
      legend:{x:0.01,y:0.02,bgcolor:'rgba(0,0,0,0)',font:{size:fs-2}},
    },{displayModeBar:false,responsive:true});
  }

  window.ovUpdate=function(){
    _deg=+document.getElementById('ov-deg').value||3;
    _N=+document.getElementById('ov-n').value||20;
    _trainPct=+document.getElementById('ov-frac').value||75;
    document.getElementById('ov-deg-val').textContent=_deg;
    document.getElementById('ov-n-val').textContent=_N;
    document.getElementById('ov-frac-val').textContent=_trainPct+' %';
    _draw();
  };
  window.ovResample=function(){
    _seed=_lcgNext(_seed^1013904223);
    ovUpdate();
  };

  function _init(){
    if(!document.getElementById('ov-plot'))return;
    if(!window.Plotly){
      if(!document.getElementById('plotly-cdn')){
        var s=document.createElement('script');
        s.id='plotly-cdn';
        s.src='https://cdn.plot.ly/plotly-2.27.0.min.js';
        s.onload=function(){_draw();};
        document.head.appendChild(s);
      }
    }else{_draw();}
  }

  if(typeof document$!=='undefined'){document$.subscribe(function(){setTimeout(_init,80);});}
  else if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',_init);}
  else{_init();}
})();
</script>

## 1.5 Data Splitting and Cross-Validation

The dataset is partitioned into three strictly non-overlapping subsets:

- **Training set** (typically 60–80%): the model sees this data directly; parameters are updated by gradient descent on training batches.
- **Validation set** (typically 10–20%): used for hyperparameter tuning and early stopping; the model is indirectly influenced — do not over-optimize on it.
- **Test set** (typically 10–20%): touched exactly once at the very end; provides an honest estimate of generalization to unseen data.

**$K$-fold cross-validation** is used when data is scarce. The training data is divided into $K$ equal folds; the model is trained $K$ times, each time holding out a different fold for validation. The $K$ validation losses are averaged for a more reliable estimate of generalization. Leave-one-out CV (LOO-CV) is the special case $K = n$.

**Data augmentation** artificially expands the training set by applying label-preserving transformations. For atomistic data: 3D rotations and reflections, noise on atomic positions, permutation of equivalent atoms.

## 1.6 Historical Context

Two converging trajectories explain why ML and computational physics now overlap: the century-long refinement of *ab initio* electronic-structure methods (left) and the rise of machine learning from statistical theory to large-scale AI (right). Both timelines share the same axis — milestones that belong to both fields appear in the same row.

<div style="position:relative;margin:1.4rem 0 2rem 0;padding:0.5rem 0;">
<div style="position:absolute;left:50%;top:0;bottom:0;width:2px;transform:translateX(-50%);background:linear-gradient(to bottom,#86BCBD 0%,#86BCBD 45%,#BA5A5A 60%,#BA5A5A 100%);opacity:0.4;z-index:0;"></div>
<div style="display:flex;gap:1.4rem;justify-content:center;margin-bottom:0.9rem;font-size:0.78rem;"><span style="display:flex;align-items:center;gap:0.35rem;"><span style="width:12px;height:12px;border-radius:50%;background:#86BCBD;display:inline-block;"></span><em>Ab initio</em> (left)</span><span style="display:flex;align-items:center;gap:0.35rem;"><span style="width:12px;height:12px;border-radius:50%;background:#BA5A5A;display:inline-block;"></span>Machine learning (right)</span></div><div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#BA5A5A;white-space:nowrap;">1812</div><div style="padding-left:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #BA5A5A;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(186,90,90,0.08);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">Bayesian inference (Laplace)</div></div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div style="text-align:right;padding-right:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #86BCBD;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(134,188,189,0.10);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">Schrödinger equation</div></div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#86BCBD;white-space:nowrap;">1926</div><div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div style="text-align:right;padding-right:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #86BCBD;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(134,188,189,0.10);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">Thomas-Fermi model; Hartree SCF</div></div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#86BCBD;white-space:nowrap;">1927</div><div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div style="text-align:right;padding-right:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #86BCBD;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(134,188,189,0.10);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">Hartree-Fock method</div></div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#86BCBD;white-space:nowrap;">1930</div><div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#BA5A5A;white-space:nowrap;">1943</div><div style="padding-left:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #BA5A5A;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(186,90,90,0.08);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">McCulloch-Pitts neuron</div></div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div style="text-align:right;padding-right:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #86BCBD;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(134,188,189,0.10);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">Slater X&#945; exchange</div></div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#86BCBD;white-space:nowrap;">1951</div><div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div style="text-align:right;padding-right:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #86BCBD;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(134,188,189,0.10);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">Hohenberg-Kohn theorems — DFT foundation</div></div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#86BCBD;white-space:nowrap;">1964</div><div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div style="text-align:right;padding-right:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #86BCBD;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(134,188,189,0.10);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">Kohn-Sham equations; LDA</div></div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#86BCBD;white-space:nowrap;">1965</div><div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div style="text-align:right;padding-right:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #86BCBD;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(134,188,189,0.10);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">Generalized gradient approximation (GGA)</div></div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#86BCBD;white-space:nowrap;">1980</div><div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#BA5A5A;white-space:nowrap;">1986</div><div style="padding-left:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #BA5A5A;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(186,90,90,0.08);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">Backpropagation (Rumelhart, Hinton, Williams)</div></div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#BA5A5A;white-space:nowrap;">1988</div><div style="padding-left:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #BA5A5A;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(186,90,90,0.08);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">Universal approximation theorem (Cybenko)</div></div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#BA5A5A;white-space:nowrap;">1989</div><div style="padding-left:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #BA5A5A;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(186,90,90,0.08);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">Q-learning (Watkins); MNIST dataset</div></div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div style="text-align:right;padding-right:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #86BCBD;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(134,188,189,0.10);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">Hybrid functionals (B3LYP)</div></div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#86BCBD;white-space:nowrap;">1993</div><div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#BA5A5A;white-space:nowrap;">1997</div><div style="padding-left:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #BA5A5A;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(186,90,90,0.08);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">Deep Blue defeats world chess champion; LSTM (Hochreiter &amp; Schmidhuber)</div></div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div style="text-align:right;padding-right:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #86BCBD;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(134,188,189,0.10);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">Nobel Prize in Chemistry (Kohn, Pople)</div></div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#86BCBD;white-space:nowrap;">1998</div><div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div style="text-align:right;padding-right:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #86BCBD;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(134,188,189,0.10);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">Meta-GGA, dispersion corrections, range-separated hybrids</div></div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#86BCBD;white-space:nowrap;">2001+</div><div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#BA5A5A;white-space:nowrap;">2006</div><div style="padding-left:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #BA5A5A;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(186,90,90,0.08);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">&#8220;Deep Learning&#8221; coined (Hinton)</div></div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#BA5A5A;white-space:nowrap;">2009</div><div style="padding-left:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #BA5A5A;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(186,90,90,0.08);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">ImageNet released; ILSVRC competition (2010)</div></div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#BA5A5A;white-space:nowrap;">2012</div><div style="padding-left:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #BA5A5A;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(186,90,90,0.08);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">AlexNet — deep learning breakthrough on ImageNet</div></div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#BA5A5A;white-space:nowrap;">2016</div><div style="padding-left:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #BA5A5A;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(186,90,90,0.08);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">AlphaGo defeats Lee Sedol</div></div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#BA5A5A;white-space:nowrap;">2017</div><div style="padding-left:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #BA5A5A;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(186,90,90,0.08);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">Transformers — &#8220;Attention Is All You Need&#8221;</div></div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div style="text-align:right;padding-right:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #86BCBD;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(134,188,189,0.10);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">ML exchange-correlation functionals; neural XC</div></div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#888;white-space:nowrap;">2018</div><div style="padding-left:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #BA5A5A;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(186,90,90,0.08);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">AlphaFold 1; BERT; GPT-1</div></div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#BA5A5A;white-space:nowrap;">2021</div><div style="padding-left:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #BA5A5A;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(186,90,90,0.08);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">AlphaFold 2 — protein structure revolution</div></div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#BA5A5A;white-space:nowrap;">2022</div><div style="padding-left:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #BA5A5A;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(186,90,90,0.08);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">ChatGPT; DALL-E 2; Stable Diffusion</div></div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#BA5A5A;white-space:nowrap;">2024</div><div style="padding-left:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #BA5A5A;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(186,90,90,0.08);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">Nobel Prize in Physics (Hopfield, Hinton); EU AI Act; GPT-4o</div></div></div></div>
<div style="display:grid;grid-template-columns:minmax(0,1fr) 5.2rem minmax(0,1fr);align-items:center;margin-bottom:0.5rem;position:relative;"><div></div><div style="text-align:center;z-index:2;position:relative;background:var(--md-default-bg-color,#fff);border-radius:4px;padding:0.05rem 0.15rem;font-weight:600;font-size:0.76rem;color:#BA5A5A;white-space:nowrap;">2025</div><div style="padding-left:0.8rem;"><div style="display:inline-block;max-width:100%;border:1.5px solid #BA5A5A;border-radius:7px;padding:0.3rem 0.6rem;background:rgba(186,90,90,0.08);"><div style="font-size:0.77rem;line-height:1.35;word-break:break-word;">Agentic AI systems; frontier model race</div></div></div></div>
</div>
## 1.7 Evaluation Metrics

Beyond the training loss, three scalar metrics are standard for reporting regression performance on a held-out test set.

**Mean absolute error (MAE)** is the average absolute residual:

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} \left|y_i - \hat{y}_i\right|$$

**Root mean square error (RMSE)** penalizes large errors more heavily:

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**Coefficient of determination** ($R^2$) measures the fraction of target variance explained by the model:

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}, \qquad \bar{y} = \frac{1}{n}\sum_i y_i$$

A perfect model gives $R^2 = 1$; a model that always predicts $\bar{y}$ gives $R^2 = 0$; a model worse than the mean gives $R^2 < 0$.

## 1.8 Aleatoric and Epistemic Uncertainty

Not all prediction errors have the same origin. Distinguishing them guides how to reduce them.

**Aleatoric uncertainty** (also: data or irreducible uncertainty) arises from inherent noise in the data-generating process — measurement error, thermal fluctuations, quantum uncertainty in atomic positions. It cannot be reduced by collecting more data; it sets a floor on achievable accuracy.

**Epistemic uncertainty** (also: model or reducible uncertainty) arises from limited training data and the resulting uncertainty about model parameters. It can be reduced by adding more training examples or choosing a more expressive model class. It is large in data-sparse regions of input space and vanishes in the infinite-data limit.

Bayesian models (Gaussian processes, Bayesian neural networks) or ensemble methods quantify both types: the predictive variance decomposes into a data-noise term (aleatoric) and a parameter-uncertainty term (epistemic). Reliable uncertainty estimates are critical for active learning and safety-critical physics applications.

## 1.9 Inductive Bias

**Bias** is the *a priori* choice of the set of functions $F$ from which the learning process selects the optimal $\hat{f}$. A learner that makes no such choice — placing no restrictions on $F$ — cannot generalize: with unconstrained hypotheses, any function consistent with the training data is equally valid, and there is no basis for preferring one prediction over another on unseen inputs.

A concrete illustration: there are $2^{2^N}$ possible Boolean functions on $N$ binary inputs. After observing one labeled example, exactly half of all remaining consistent functions assign value 1 to any unseen pattern and half assign value 0 — no generalization is possible. The learner can only memorize. Restricting to a much smaller set $F' \subset F$ (with $\log |F'| \ll \log |F|$) can leave only a single consistent function after training, enabling meaningful predictions.

Two mechanisms achieve this restriction.

**Absolute bias** restricts $F$ to a defined subset determined by the model architecture — a linear model can only represent affine functions; a convolutional network has built-in translation equivariance. Anything outside the restricted $F$ is structurally inaccessible to the learner.

**Preference bias** does not hard-restrict $F$ but selects among hypotheses within $F$ according to an ordering scheme or domain requirements — for example, preferring smoother functions (L2 regularization), sparser weight vectors (L1), or models that satisfy known symmetry requirements of the physical system.

The **no free lunch** theorem formalizes the unavoidability of bias: no single algorithm outperforms random guessing across *all* possible problem distributions simultaneously. Choosing a good bias — a hypothesis class matched to the structure of the target $f$ — is therefore the central design decision in ML.

## 1.10 Scientific Machine Learning

**Scientific machine learning (SciML)** combines the data-efficiency of physical laws with the flexibility of data-driven models. Physical constraints — conservation laws, symmetries, PDEs — can be enforced directly in the model or loss, reducing the effective problem dimensionality and improving generalization far from training data.

**Physics-informed neural networks (PINNs)** minimize a composite loss

$$\mathcal{L} = \mathcal{L}_\text{data} + \lambda\,\mathcal{L}_\text{PDE},$$

where $\mathcal{L}_\text{data}$ fits labeled observations and $\mathcal{L}_\text{PDE}$ penalizes violation of a known governing equation (Navier-Stokes, Schrödinger, etc.) at collocation points. The PDE residual acts as a strong physics-based regularizer.

**Symmetry-adapted architectures** build physical symmetries into the network by construction rather than learning them from data. Equivariant graph neural networks (Chapter 6) produce outputs that transform predictably under 3D rotations and reflections, automatically satisfying conservation of angular momentum — a property that a generic MLP must learn from augmented data.

## 1.11 A Typical ML Workflow in Computational Physics

A structured workflow prevents common pitfalls: data leakage, metric shopping, and irreproducible results.

<div style="margin:1.6rem 0 2rem 0;">
<div style="width:min(520px,100%);margin:0 auto;border:2px solid #86BCBD;border-radius:10px;padding:0.6rem 1rem;background:rgba(186,90,90,0.12);text-align:left;">
  <div style="display:flex;align-items:baseline;gap:0.5rem;">
    <span style="font-weight:700;font-size:0.72rem;color:#86BCBD;letter-spacing:0.06em;text-transform:uppercase;white-space:nowrap;">Step 1</span>
    <span style="font-weight:700;font-size:0.9rem;">Define the task</span>
  </div>
  <div style="font-size:0.79rem;line-height:1.45;margin-top:0.25rem;opacity:0.9;">Identify the target property, acceptable error tolerance, and how reference data will be generated.</div>
</div>
<div style="display:flex;justify-content:center;margin:0.05rem 0;"><div style="width:2px;height:1.4rem;background:#999;opacity:0.5;"></div></div>
<div style="display:flex;justify-content:center;margin:-0.05rem 0 0.05rem 0;"><div style="width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;border-top:8px solid #999;opacity:0.5;"></div></div>
<div style="width:min(520px,100%);margin:0 auto;border:2px solid #86BCBD;border-radius:10px;padding:0.6rem 1rem;background:rgba(134,188,189,0.12);text-align:left;">
  <div style="display:flex;align-items:baseline;gap:0.5rem;">
    <span style="font-weight:700;font-size:0.72rem;color:#86BCBD;letter-spacing:0.06em;text-transform:uppercase;white-space:nowrap;">Step 2</span>
    <span style="font-weight:700;font-size:0.9rem;">Generate and clean the data</span>
  </div>
  <div style="font-size:0.79rem;line-height:1.45;margin-top:0.25rem;opacity:0.9;">Run reference calculations (DFT, CCSD) or use an existing database. Check units, convergence, and data provenance.</div>
</div>
<div style="display:flex;justify-content:center;margin:0.05rem 0;"><div style="width:2px;height:1.4rem;background:#999;opacity:0.5;"></div></div>
<div style="display:flex;justify-content:center;margin:-0.05rem 0 0.05rem 0;"><div style="width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;border-top:8px solid #999;opacity:0.5;"></div></div>
<div style="width:min(520px,100%);margin:0 auto;border:2px solid #86BCBD;border-radius:10px;padding:0.6rem 1rem;background:rgba(134,188,189,0.12);text-align:left;">
  <div style="display:flex;align-items:baseline;gap:0.5rem;">
    <span style="font-weight:700;font-size:0.72rem;color:#86BCBD;letter-spacing:0.06em;text-transform:uppercase;white-space:nowrap;">Step 3</span>
    <span style="font-weight:700;font-size:0.9rem;">Visualize and explore</span>
  </div>
  <div style="font-size:0.79rem;line-height:1.45;margin-top:0.25rem;opacity:0.9;">Inspect feature distributions and correlations to catch systematic errors before modelling.</div>
</div>
<div style="display:flex;justify-content:center;margin:0.05rem 0;"><div style="width:2px;height:1.4rem;background:#999;opacity:0.5;"></div></div>
<div style="display:flex;justify-content:center;margin:-0.05rem 0 0.05rem 0;"><div style="width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;border-top:8px solid #999;opacity:0.5;"></div></div>
<div style="width:min(520px,100%);margin:0 auto;border:2px solid #A4CE8B;border-radius:10px;padding:0.6rem 1rem;background:rgba(164,206,139,0.12);text-align:left;">
  <div style="display:flex;align-items:baseline;gap:0.5rem;">
    <span style="font-weight:700;font-size:0.72rem;color:#A4CE8B;letter-spacing:0.06em;text-transform:uppercase;white-space:nowrap;">Step 4</span>
    <span style="font-weight:700;font-size:0.9rem;">Prepare the data</span>
  </div>
  <div style="font-size:0.79rem;line-height:1.45;margin-top:0.25rem;opacity:0.9;">Split by structure into train / validation / test. Fit preprocessing (scaling, normalization) on the training set only.</div>
</div>
<div style="display:flex;justify-content:center;margin:0.05rem 0;"><div style="width:2px;height:1.4rem;background:#999;opacity:0.5;"></div></div>
<div style="display:flex;justify-content:center;margin:-0.05rem 0 0.05rem 0;"><div style="width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;border-top:8px solid #999;opacity:0.5;"></div></div>
<div style="width:min(520px,100%);margin:0 auto;border:2px solid #A4CE8B;border-radius:10px;padding:0.6rem 1rem;background:rgba(164,206,139,0.12);text-align:left;">
  <div style="display:flex;align-items:baseline;gap:0.5rem;">
    <span style="font-weight:700;font-size:0.72rem;color:#A4CE8B;letter-spacing:0.06em;text-transform:uppercase;white-space:nowrap;">Step 5</span>
    <span style="font-weight:700;font-size:0.9rem;">Featurize the inputs</span>
  </div>
  <div style="font-size:0.79rem;line-height:1.45;margin-top:0.25rem;opacity:0.9;">Encode atomic structures as symmetry-invariant descriptors (Chapter 2) or graphs (Chapter 6).</div>
</div>
<div style="display:flex;justify-content:center;margin:0.05rem 0;"><div style="width:2px;height:1.4rem;background:#999;opacity:0.5;"></div></div>
<div style="display:flex;justify-content:center;margin:-0.05rem 0 0.05rem 0;"><div style="width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;border-top:8px solid #999;opacity:0.5;"></div></div>
<div style="width:min(520px,100%);margin:0 auto;border:2px solid #F7E49B;border-radius:10px;padding:0.6rem 1rem;background:rgba(247,228,155,0.12);text-align:left;">
  <div style="display:flex;align-items:baseline;gap:0.5rem;">
    <span style="font-weight:700;font-size:0.72rem;color:#F7E49B;letter-spacing:0.06em;text-transform:uppercase;white-space:nowrap;">Step 6</span>
    <span style="font-weight:700;font-size:0.9rem;">Select, train, and evaluate</span>
  </div>
  <div style="font-size:0.79rem;line-height:1.45;margin-top:0.25rem;opacity:0.9;">Choose a model class suited to the data regime and output type. Fit on the training set; monitor generalization on the validation set.</div>
</div>
<div style="display:flex;justify-content:center;margin:0.05rem 0;"><div style="width:2px;height:1.4rem;background:#999;opacity:0.5;"></div></div>
<div style="display:flex;justify-content:center;margin:-0.05rem 0 0.05rem 0;"><div style="width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;border-top:8px solid #999;opacity:0.5;"></div></div>
<div style="width:min(520px,100%);margin:0 auto;border:2px solid #F7E49B;border-radius:10px;padding:0.6rem 1rem;background:rgba(247,228,155,0.12);text-align:left;">
  <div style="display:flex;align-items:baseline;gap:0.5rem;">
    <span style="font-weight:700;font-size:0.72rem;color:#F7E49B;letter-spacing:0.06em;text-transform:uppercase;white-space:nowrap;">Step 7</span>
    <span style="font-weight:700;font-size:0.9rem;">Optimise and fine-tune</span>
  </div>
  <div style="font-size:0.79rem;line-height:1.45;margin-top:0.25rem;opacity:0.9;">Tune hyperparameters (architecture, learning rate, regularization) on the validation set.</div>
</div>
<div style="display:flex;justify-content:center;margin:0.05rem 0;"><div style="width:2px;height:1.4rem;background:#999;opacity:0.5;"></div></div>
<div style="display:flex;justify-content:center;margin:-0.05rem 0 0.05rem 0;"><div style="width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;border-top:8px solid #999;opacity:0.5;"></div></div>
<div style="width:min(520px,100%);margin:0 auto;border:2px solid #F7E49B;border-radius:10px;padding:0.6rem 1rem;background:rgba(247,228,155,0.12);text-align:left;">
  <div style="display:flex;align-items:baseline;gap:0.5rem;">
    <span style="font-weight:700;font-size:0.72rem;color:#F7E49B;letter-spacing:0.06em;text-transform:uppercase;white-space:nowrap;">Step 8</span>
    <span style="font-weight:700;font-size:0.9rem;">Assess accuracy and uncertainty</span>
  </div>
  <div style="font-size:0.79rem;line-height:1.45;margin-top:0.25rem;opacity:0.9;">Evaluate on the held-out test set: MAE, RMSE, R². Quantify aleatoric and epistemic uncertainty (Chapter 4).</div>
</div>
<div style="display:flex;justify-content:center;margin:0.05rem 0;"><div style="width:2px;height:1.4rem;background:#999;opacity:0.5;"></div></div>
<div style="display:flex;justify-content:center;margin:-0.05rem 0 0.05rem 0;"><div style="width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;border-top:8px solid #999;opacity:0.5;"></div></div>
<div style="width:min(520px,100%);margin:0 auto;border:2px solid #BA5A5A;border-radius:10px;padding:0.6rem 1rem;background:rgba(186,90,90,0.12);text-align:left;">
  <div style="display:flex;align-items:baseline;gap:0.5rem;">
    <span style="font-weight:700;font-size:0.72rem;color:#BA5A5A;letter-spacing:0.06em;text-transform:uppercase;white-space:nowrap;">Step 9</span>
    <span style="font-weight:700;font-size:0.9rem;">Generate more data if needed</span>
  </div>
  <div style="font-size:0.79rem;line-height:1.45;margin-top:0.25rem;opacity:0.9;">Use active learning to target new calculations where model uncertainty is highest.</div>
</div>
<div style="display:flex;justify-content:center;margin:0.05rem 0;"><div style="width:2px;height:1.4rem;background:#999;opacity:0.5;"></div></div>
<div style="display:flex;justify-content:center;margin:-0.05rem 0 0.05rem 0;"><div style="width:0;height:0;border-left:6px solid transparent;border-right:6px solid transparent;border-top:8px solid #999;opacity:0.5;"></div></div>
<div style="width:min(520px,100%);margin:0 auto;border:2px solid #BA5A5A;border-radius:10px;padding:0.6rem 1rem;background:rgba(186,90,90,0.12);text-align:left;">
  <div style="display:flex;align-items:baseline;gap:0.5rem;">
    <span style="font-weight:700;font-size:0.72rem;color:#BA5A5A;letter-spacing:0.06em;text-transform:uppercase;white-space:nowrap;">Step 10</span>
    <span style="font-weight:700;font-size:0.9rem;">Deploy</span>
  </div>
  <div style="font-size:0.79rem;line-height:1.45;margin-top:0.25rem;opacity:0.9;">Version the model; record training provenance: functional, basis set, code version, dataset composition.</div>
</div>
</div>

## 1.12 Data Quality and Pipeline

Model performance is bounded by data quality. Three issues are especially common in physics ML workflows.

**Data leakage** occurs when information from the test set influences model training — most often through normalization statistics computed over the full dataset before splitting, or through feature selection that uses test labels. The result is optimistic test metrics that do not reflect true generalization. Prevention: always fit preprocessing transforms on the training split only, then apply them to validation and test.

**Data provenance** tracks the origin and processing history of each data point: which code version, which DFT functional, which basis set, which convergence threshold. Without provenance, combining datasets from different sources silently introduces systematic offsets. In practice, store metadata alongside each calculation in a structured format (ASE database, JSON sidecar files).

**Public datasets** widely used in ML for chemistry and materials include QM9 (134k small organic molecules, DFT/B3LYP properties), the Materials Project (hundreds of thousands of inorganic compounds), ANI-1 (organic molecule energies), MD17 (molecular dynamics trajectories), and OC20/OC22 (open catalyst datasets for heterogeneous catalysis).

## 1.13 Software and Hardware

**ML frameworks.** PyTorch (Meta) is the dominant research framework; its dynamic computation graph and `autograd` engine make custom architectures easy to prototype. JAX (Google) offers composable function transforms (`jit`, `grad`, `vmap`) suited for scientific computing with hardware-agnostic acceleration. TensorFlow (Google) remains common in production deployments.

**GPU accelerators.** GPUs accelerate the matrix multiplications at the core of neural network training by 100–1000× over a CPU, via thousands of parallel cores and extremely high memory bandwidth. The table below compares current data-centre accelerators used in ML for physics:

<div style="display:flex;justify-content:center;overflow-x:auto;margin:1rem 0 1.4rem;">
<table style="margin:0 auto;border-collapse:collapse;font-size:0.84rem;">
<thead><tr style="border-bottom:2px solid var(--md-default-fg-color--light,#999);">
<th style="padding:0.45rem 0.85rem;text-align:left;">Chip</th>
<th style="padding:0.45rem 0.85rem;text-align:left;">Vendor</th>
<th style="padding:0.45rem 0.85rem;text-align:center;">Year</th>
<th style="padding:0.45rem 0.85rem;text-align:left;">Memory</th>
<th style="padding:0.45rem 0.85rem;text-align:left;">Bandwidth</th>
<th style="padding:0.45rem 0.85rem;text-align:left;">Generation</th>
</tr></thead>
<tbody>
<tr style="border-bottom:1px solid var(--md-default-fg-color--lightest,#ddd);"><td style="padding:0.4rem 0.85rem;text-align:left;">A100 SXM4</td><td style="padding:0.4rem 0.85rem;text-align:left;">NVIDIA</td><td style="padding:0.4rem 0.85rem;text-align:center;">2020</td><td style="padding:0.4rem 0.85rem;text-align:left;">80 GB HBM2e</td><td style="padding:0.4rem 0.85rem;text-align:left;">2.0 TB/s</td><td style="padding:0.4rem 0.85rem;text-align:left;">Ampere</td></tr>
<tr style="border-bottom:1px solid var(--md-default-fg-color--lightest,#ddd);"><td style="padding:0.4rem 0.85rem;text-align:left;">H100 SXM</td><td style="padding:0.4rem 0.85rem;text-align:left;">NVIDIA</td><td style="padding:0.4rem 0.85rem;text-align:center;">2022</td><td style="padding:0.4rem 0.85rem;text-align:left;">80 GB HBM3</td><td style="padding:0.4rem 0.85rem;text-align:left;">3.35 TB/s</td><td style="padding:0.4rem 0.85rem;text-align:left;">Hopper</td></tr>
<tr style="border-bottom:1px solid var(--md-default-fg-color--lightest,#ddd);"><td style="padding:0.4rem 0.85rem;text-align:left;">L40S</td><td style="padding:0.4rem 0.85rem;text-align:left;">NVIDIA</td><td style="padding:0.4rem 0.85rem;text-align:center;">2023</td><td style="padding:0.4rem 0.85rem;text-align:left;">48 GB GDDR6</td><td style="padding:0.4rem 0.85rem;text-align:left;">864 GB/s</td><td style="padding:0.4rem 0.85rem;text-align:left;">Ada Lovelace</td></tr>
<tr style="border-bottom:1px solid var(--md-default-fg-color--lightest,#ddd);"><td style="padding:0.4rem 0.85rem;text-align:left;">H200 SXM</td><td style="padding:0.4rem 0.85rem;text-align:left;">NVIDIA</td><td style="padding:0.4rem 0.85rem;text-align:center;">2024</td><td style="padding:0.4rem 0.85rem;text-align:left;">141 GB HBM3e</td><td style="padding:0.4rem 0.85rem;text-align:left;">4.8 TB/s</td><td style="padding:0.4rem 0.85rem;text-align:left;">Hopper</td></tr>
<tr style="border-bottom:1px solid var(--md-default-fg-color--lightest,#ddd);"><td style="padding:0.4rem 0.85rem;text-align:left;">B200</td><td style="padding:0.4rem 0.85rem;text-align:left;">NVIDIA</td><td style="padding:0.4rem 0.85rem;text-align:center;">2025</td><td style="padding:0.4rem 0.85rem;text-align:left;">180 GB HBM3e</td><td style="padding:0.4rem 0.85rem;text-align:left;">8.0 TB/s</td><td style="padding:0.4rem 0.85rem;text-align:left;">Blackwell</td></tr>
<tr style="border-bottom:1px solid var(--md-default-fg-color--lightest,#ddd);"><td style="padding:0.4rem 0.85rem;text-align:left;">GB200</td><td style="padding:0.4rem 0.85rem;text-align:left;">NVIDIA</td><td style="padding:0.4rem 0.85rem;text-align:center;">2025</td><td style="padding:0.4rem 0.85rem;text-align:left;">2×B200 + Grace ARM CPU</td><td style="padding:0.4rem 0.85rem;text-align:left;">900 GB/s NVLink-C2C</td><td style="padding:0.4rem 0.85rem;text-align:left;">Blackwell</td></tr>
<tr style="border-bottom:1px solid var(--md-default-fg-color--lightest,#ddd);"><td style="padding:0.4rem 0.85rem;text-align:left;">MI300X</td><td style="padding:0.4rem 0.85rem;text-align:left;">AMD</td><td style="padding:0.4rem 0.85rem;text-align:center;">2023</td><td style="padding:0.4rem 0.85rem;text-align:left;">192 GB HBM3</td><td style="padding:0.4rem 0.85rem;text-align:left;">5.3 TB/s</td><td style="padding:0.4rem 0.85rem;text-align:left;">CDNA 3</td></tr>
<tr style=""><td style="padding:0.4rem 0.85rem;text-align:left;">MI350X</td><td style="padding:0.4rem 0.85rem;text-align:left;">AMD</td><td style="padding:0.4rem 0.85rem;text-align:center;">2025</td><td style="padding:0.4rem 0.85rem;text-align:left;">288 GB HBM3e</td><td style="padding:0.4rem 0.85rem;text-align:left;">8.0 TB/s</td><td style="padding:0.4rem 0.85rem;text-align:left;">CDNA 4</td></tr>
</tbody></table></div>

The H200 carries the same compute die as the H100 but nearly doubles the memory capacity, relieving the bandwidth bottleneck for large models. The B200 (Blackwell) is a dual-die chip with fifth-generation Tensor Cores and NVLink 5.0; a GB200 NVL72 rack integrates 72 B200 chips over NVLink. AMD's MI300X powers El Capitan (#2 globally) thanks to its record memory capacity; the MI350X (CDNA 4, released mid-2025) raises this to 288 GB.

**HPC hierarchy.** Supercomputing resources span several tiers. Named examples from the [June 2026 TOP500](https://top500.org/lists/top500/2026/06/) and major facility lists are shown below:

<div style="margin:1.8rem 0 2.5rem;max-width:660px;margin-left:auto;margin-right:auto;font-size:0.75rem;line-height:1.5;">

<!-- Tier 0: Exascale — narrowest trapezoid (top 22%–78%, bottom 14%–86%) -->
<div style="clip-path:polygon(35% 0%,65% 0%,72% 100%,28% 100%);background:rgba(186,90,90,0.30);padding:0.75rem 36%;text-align:center;min-height:5.8rem;display:flex;flex-direction:column;justify-content:center;gap:0.18rem;">
<span style="font-weight:700;font-size:0.6rem;color:#8B2020;text-transform:uppercase;letter-spacing:0.09em;">Tier 0 &nbsp;·&nbsp; Exascale (&gt;1 EF/s)</span>
<span>LineShine (China) · El Capitan (USA)</span>
<span>Frontier (USA) · Aurora (USA) · JUPITER (Germany)</span>
</div>

<!-- Tier 1: Pre-exascale (top 14%–86%, bottom 7%–93%) -->
<div style="clip-path:polygon(28% 0%,72% 0%,79% 100%,21% 100%);background:rgba(186,90,90,0.18);padding:0.75rem 29%;text-align:center;min-height:5.8rem;display:flex;flex-direction:column;justify-content:center;gap:0.18rem;margin-top:-1px;">
<span style="font-weight:700;font-size:0.6rem;color:#8B2020;text-transform:uppercase;letter-spacing:0.09em;">Tier 1 &nbsp;·&nbsp; Pre-exascale Flagship</span>
<span>MareNostrum 5 (Spain) · LUMI (Finland) · Leonardo (Italy)</span>
<span>JUWELS Booster (Germany) · Perlmutter (USA)</span>
</div>

<!-- Tier 2: National (top 7%–93%, bottom 0%–100%) -->
<div style="clip-path:polygon(21% 0%,79% 0%,86% 100%,14% 100%);background:rgba(247,228,155,0.38);padding:0.75rem 22%;text-align:center;min-height:5.8rem;display:flex;flex-direction:column;justify-content:center;gap:0.18rem;margin-top:-1px;">
<span style="font-weight:700;font-size:0.6rem;color:#6A5000;text-transform:uppercase;letter-spacing:0.09em;">Tier 2 &nbsp;·&nbsp; National HPC Facilities</span>
<span>MUSICA (Austria) · VSC-5 (Austria) · ARCHER2 (UK)<br>Hawk (Germany) · SuperMUC-NG (Germany) · Gadi (Australia)<br>Cedar (Canada) · Shaheen III (Saudi Arabia) · Mahti (Finland)</span>
</div>

<!-- Tier 3: Institutional — full-width rectangle (base of pyramid) -->
<div style="clip-path:polygon(14% 0%,86% 0%,93% 100%,7% 100%);background:rgba(164,206,139,0.35);padding:0.75rem 15%;text-align:center;min-height:4.8rem;display:flex;flex-direction:column;justify-content:center;gap:0.18rem;margin-top:-1px;">
<span style="font-weight:700;font-size:0.6rem;color:#2A5A1A;text-transform:uppercase;letter-spacing:0.09em;">Tier 3 &nbsp;·&nbsp; University / Institutional Clusters</span>
<span>bwUniCluster 2.0 (Germany) · CSD3 (UK, Cambridge) · Narval (Canada) · ALICE (Netherlands, Leiden)</span>
</div>

<!-- Tier 4: Local — base -->
<div style="clip-path:polygon(7% 0%,93% 0%,100% 100%,0% 100%);background:rgba(134,188,189,0.35);padding:0.75rem 8%;text-align:center;min-height:4rem;display:flex;flex-direction:column;justify-content:center;gap:0.18rem;margin-top:-1px;">
<span style="font-weight:700;font-size:0.6rem;color:#1A5A5C;text-transform:uppercase;letter-spacing:0.09em;">Tier 4 &nbsp;·&nbsp; Local Workstation / Lab Server</span>
<span>NVIDIA DGX H100 · single/multi-GPU servers · laptop or desktop CPU</span>
</div>

</div>

---

