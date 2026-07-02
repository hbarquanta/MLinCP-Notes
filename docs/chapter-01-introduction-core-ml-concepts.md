# Chapter 1 — Introduction & Core ML Concepts

Machine learning (ML) is the discipline of constructing algorithms that learn patterns from data, rather than relying on explicitly programmed rules. In computational physics and chemistry, ML provides a route to building fast, accurate surrogate models for expensive quantum-mechanical calculations.

## 1.1 Learning Paradigms

Consider an unseen process (the data-generating process) that connects an input space $\mathcal{X}$ to a target space $\mathcal{Y}$, described by some function $f: \mathcal{X} \to \mathcal{Y}$.

Statistical samples of $\mathcal{X}$ are called **features** (also: attributes, input variables): $x_i \in \mathcal{X}$ are vectors of dimension $D$ containing reals, integers, or other values. Samples of $\mathcal{Y}$ are called **labels** (also: classes, targets): $y_i \in \mathcal{Y}$ can be scalars, vectors, or tensors. The index $i$ runs over individual examples.

A **dataset** $T$ can be labeled, $T = \{(x_i, y_i)\}$, or unlabeled, $T = \{x_i\}$.

**Supervised learning** requires a labeled dataset $T = \{(x_i, y_i)\}_{i=1}^n$ of $n$ examples. The goal is to find a model $\hat{f}: \mathcal{X} \to \mathcal{Y}$ that approximates $f$, i.e. $\hat{f}(x_i) = \hat{y}_i \approx y_i$, and generalizes well to new, unseen inputs.

**Unsupervised learning** works on unlabeled data $T = \{x_i\}_{i=1}^n$. The goal is to describe or understand the structure of the data — typical tasks include dimensionality reduction, clustering, outlier detection, and generative modelling (covered in Chapter 3 and Chapter 9).

The fundamental requirement in both cases is **generalization**: the model must perform well on data it has not seen during training, not merely memorize the training set. A model that achieves this is said to generalize.

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

The figure below shows polynomial regression on noisy samples from $f(x) = \sin(2\pi x)$. At low degree the model lacks capacity (underfitting); at high degree it interpolates the training noise and the test error explodes (overfitting). The degree at which test error is minimal is the sweet spot.

<div id="overfit-widget" style="margin:1.5rem 0;">
  <div style="display:flex;gap:1.2rem;align-items:center;flex-wrap:wrap;margin-bottom:0.6rem;">
    <span style="font-size:0.93rem;font-weight:600;">Polynomial degree</span>
    <input type="range" id="ov-deg" min="1" max="10" value="3" step="1"
           oninput="ovUpdate(+this.value)" style="width:180px;accent-color:#BA5A5A;">
    <span id="ov-deg-val" style="font-weight:700;min-width:1.4em;">3</span>
  </div>
  <div id="ov-plot" style="width:100%;height:420px;"></div>
</div>

<script>
(function(){
  var N=20;
  var _noise=[0.12,-0.18,0.07,-0.22,0.15,0.09,-0.14,0.21,-0.08,0.17,
              -0.19,0.11,0.23,-0.06,0.18,-0.13,0.08,-0.21,0.16,-0.10];
  var _xs=Array.from({length:N},function(_,i){return i/(N-1);});
  var _ys=_xs.map(function(x,i){return Math.sin(2*Math.PI*x)+_noise[i];});
  var _xTr=[],_yTr=[],_xTe=[],_yTe=[];
  _xs.forEach(function(x,i){
    if(i%3!==0){_xTr.push(x);_yTr.push(_ys[i]);}
    else{_xTe.push(x);_yTe.push(_ys[i]);}
  });

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

  var _degs=Array.from({length:10},function(_,i){return i+1;});
  var _errTr=[],_errTe=[];
  _degs.forEach(function(d){
    var c=_fit(_xTr,_yTr,d);
    _errTr.push(_mse(c,_xTr,_yTr));
    _errTe.push(_mse(c,_xTe,_yTe));
  });

  function _dark(){return document.body&&document.body.getAttribute('data-md-color-scheme')==='slate';}

  function _draw(deg){
    var el=document.getElementById('ov-plot');
    if(!el||!window.Plotly)return;
    var c=_fit(_xTr,_yTr,deg);
    var xL=Array.from({length:200},function(_,i){return i/199;});
    var yL=xL.map(function(x){return _eval(c,x);});
    var yTrue=xL.map(function(x){return Math.sin(2*Math.PI*x);});
    var dk=_dark(),bg=dk?'#1e2228':'#ffffff',fg=dk?'#e0e0e0':'#333333';
    var gc=dk?'rgba(255,255,255,0.07)':'rgba(0,0,0,0.07)';
    var eMax=Math.max.apply(null,_errTe)*1.2;
    var label=deg<=2?'underfitting':deg>=7?'overfitting':'good fit';

    Plotly.react('ov-plot',[
      {x:xL,y:yTrue,mode:'lines',name:'true f(x)',
       line:{color:'#86BCBD',width:1.5,dash:'dash'},xaxis:'x',yaxis:'y'},
      {x:xL,y:yL,mode:'lines',name:'degree '+deg,
       line:{color:'#BA5A5A',width:2.5},xaxis:'x',yaxis:'y'},
      {x:_xTr,y:_yTr,mode:'markers',name:'train',
       marker:{color:'#A4CE8B',size:7,line:{color:'rgba(0,0,0,0.25)',width:1}},xaxis:'x',yaxis:'y'},
      {x:_xTe,y:_yTe,mode:'markers',name:'test',
       marker:{symbol:'diamond',color:'#F7E49B',size:8,line:{color:'rgba(0,0,0,0.4)',width:1}},xaxis:'x',yaxis:'y'},
      {x:_degs,y:_errTr,mode:'lines+markers',name:'train MSE',
       line:{color:'#A4CE8B',width:2},marker:{color:'#A4CE8B',size:5},xaxis:'x2',yaxis:'y2'},
      {x:_degs,y:_errTe,mode:'lines+markers',name:'test MSE',
       line:{color:'#BA5A5A',width:2},marker:{color:'#BA5A5A',size:5},xaxis:'x2',yaxis:'y2'},
      {x:[deg,deg],y:[0,eMax],mode:'lines',showlegend:false,
       line:{color:'#86BCBD',width:1.5,dash:'dot'},xaxis:'x2',yaxis:'y2'},
    ],{
      paper_bgcolor:bg,plot_bgcolor:bg,
      font:{color:fg,family:'inherit',size:12},
      margin:{t:45,b:48,l:55,r:15},
      annotations:[
        {text:'Fit to noisy data ( '+label+' )',xref:'paper',yref:'paper',
         x:0.22,y:1.06,showarrow:false,font:{size:12,color:fg}},
        {text:'Training vs. test error',xref:'paper',yref:'paper',
         x:0.78,y:1.06,showarrow:false,font:{size:12,color:fg}},
      ],
      xaxis:{title:'x',range:[-0.02,1.02],domain:[0,0.44],gridcolor:gc,zerolinecolor:gc},
      yaxis:{title:'y',range:[-2.2,2.2],gridcolor:gc,zerolinecolor:gc},
      xaxis2:{title:'degree',range:[0.5,10.5],dtick:1,domain:[0.56,1],gridcolor:gc,zerolinecolor:gc},
      yaxis2:{title:'MSE',range:[0,eMax],gridcolor:gc,zerolinecolor:gc},
      legend:{x:0.01,y:0.02,bgcolor:'rgba(0,0,0,0)',font:{size:10}},
    },{displayModeBar:false,responsive:true});
  }

  window.ovUpdate=function(deg){
    document.getElementById('ov-deg-val').textContent=deg;
    _draw(deg);
  };

  function _init(){
    if(!document.getElementById('ov-plot'))return;
    if(!window.Plotly){
      if(!document.getElementById('plotly-cdn')){
        var s=document.createElement('script');
        s.id='plotly-cdn';
        s.src='https://cdn.plot.ly/plotly-2.27.0.min.js';
        s.onload=function(){_draw(3);};
        document.head.appendChild(s);
      }
    } else {
      _draw(parseInt(document.getElementById('ov-deg').value)||3);
    }
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

---

