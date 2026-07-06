# Chapter 5: Neural Networks, Trees & Regularization

This chapter covers two families of universal approximators: tree-based models and feed-forward neural networks, together with the optimization and regularization strategies used to train them.

Three levels of nonlinearity can be built into a regression model. A *linear model* is a weighted sum of raw input features: $f(\mathbf{x},\omega) = w_0 + \sum_j w_j x_j$. A model with *fixed nonlinear features* applies a predetermined set of basis functions $\phi_b$ to the input: $f(\mathbf{x},\omega) = w_0 + \sum_b w_b \phi_b(\mathbf{x})$; the $\phi_b$ are fixed before training and only the weights $w_b$ are learned. A model with *learnable nonlinear features* lets the sub-functions themselves carry tunable parameters: $f(\mathbf{x},\omega) = w_0 + \sum_b w_b f_b(\mathbf{x},\omega_b)$. Decision trees and feed-forward neural networks both belong to this third category; the $f_b$ are learned end-to-end alongside the outer weights.

## 5.1 Decision Stumps and Decision Trees

The simplest nonlinear model is a **decision stump**: a single step function along one input coordinate. For a one-dimensional input $x \in \mathbb{R}$, a stump has three parameters (a bias $w_0$, a step height $w_1$, and a split point $s_1$) and produces

$$f(x;\,w_0, w_1, s_1) = w_0 + w_1 \,\mathbf{1}\!\left\{x > s_1\right\},$$

where $\mathbf{1}\{x > s_1\}$ is 1 if $x > s_1$ and 0 otherwise. For an $N$-dimensional input $\mathbf{x} \in \mathbb{R}^N$ the stump still cuts along a single coordinate: $\mathbf{1}\{x_i > s_1\}$ for some chosen feature index $i$.

A **decision tree** of depth $D$ is a hierarchy of $2^D - 1$ stumps arranged so that the output of each stump routes a data point to the left or right child stump, terminating in $2^D$ leaf nodes (each holding a constant prediction value). The total number of free parameters grows exponentially with depth: each internal node contributes a split index, a split threshold $s$, and the two leaf values $v_L$ and $v_R$.

**Training a tree of depth $D$** proceeds recursively. For each candidate split (choosing feature index $i$ and threshold $s$), partition the $P$ training points into two groups $\mathcal{P}_L = \{p : x_i^{(p)} \leq s\}$ and $\mathcal{P}_R = \{p : x_i^{(p)} > s\}$. The optimal constant predictions are the group means,

$$v_L = \frac{1}{|\mathcal{P}_L|}\sum_{p \in \mathcal{P}_L} y^{(p)}, \qquad v_R = \frac{1}{|\mathcal{P}_R|}\sum_{p \in \mathcal{P}_R} y^{(p)},$$

and the split is chosen to minimize the least-squares error across both groups. The procedure is repeated recursively within each child node until depth $D$ is reached. In practice, recursion is also halted early when the label variance within a node falls below a threshold or when a node contains fewer than a minimum number of data points. For classification tasks, leaf values are found by minimizing a classifier loss rather than squared error; a common choice is log-loss, $\sum_{p \in \mathcal{P}} \log(1 + e^{-v\,y^{(p)}})$, whose minimizer gives the leaf log-odds. Trees are high-variance, low-bias models: a sufficiently deep tree memorizes the training data.

The widget below shows how a depth-$D$ tree partitions the feature axis into piecewise-constant segments. Increasing $D$ halves each segment's width and doubles the leaf count.

<div id="tw-widget" style="margin:1.5rem 0;">
  <div style="display:flex;gap:0.75rem;align-items:center;margin-bottom:0.6rem;flex-wrap:wrap;">
    <span style="font-size:0.88rem;font-weight:600;">Depth <em>D</em> =</span>
    <input type="range" id="tw-depth" min="0" max="20" value="2" step="1"
           oninput="twSetD(+this.value)" style="width:140px;accent-color:#86BCBD;">
    <span id="tw-dval" style="font-weight:700;min-width:1em;">2</span>
    <span style="font-size:0.79rem;opacity:0.65;margin-left:0.4rem;">
      <span id="tw-leaves">4</span> leaves &middot; <span id="tw-params">7</span> parameters
    </span>
  </div>
  <div id="tw-plot" style="height:300px;"></div>
  <div style="font-size:0.8rem;opacity:0.65;margin-top:0.3rem;">True function: sin(2&pi;x) &mdash; dashed green. Each new depth splits the highest-variance node.</div>
</div>

<script>
(function(){
  function _rng(s){var st=s>>>0;return function(){st=(st*1664525+1013904223)>>>0;return st/0x100000000;};}
  var r=_rng(314),_N=80,_pts=[];
  for(var i=0;i<_N;i++){var xi=r();_pts.push([xi,Math.sin(2*Math.PI*xi)+(r()-0.5)*0.4]);}
  _pts.sort(function(a,b){return a[0]-b[0];});
  var _xF=[],_yF=[];for(var i=0;i<150;i++){var xf=i/149;_xF.push(xf);_yF.push(Math.sin(2*Math.PI*xf));}
  function _mean(ps){return ps.reduce(function(s,p){return s+p[1];},0)/(ps.length||1);}
  function _var(ps){var m=_mean(ps);return ps.reduce(function(s,p){return s+(p[1]-m)*(p[1]-m);},0)/(ps.length||1);}
  function _fit(pts,D,x0,x1){
    if(D===0||pts.length<2)return [{x0:x0,x1:x1,y:_mean(pts)}];
    var best=Infinity,bk=-1;
    for(var k=0;k<pts.length-1;k++){var L=pts.slice(0,k+1),R=pts.slice(k+1);var cost=L.length*_var(L)+R.length*_var(R);if(cost<best){best=cost;bk=k;}}
    if(bk<0)return [{x0:x0,x1:x1,y:_mean(pts)}];
    var sp=(pts[bk][0]+pts[bk+1][0])/2;
    return _fit(pts.slice(0,bk+1),D-1,x0,sp).concat(_fit(pts.slice(bk+1),D-1,sp,x1));
  }
  var _D=2;
  function _dark(){return document.body&&document.body.getAttribute('data-md-color-scheme')==='slate';}
  function _draw(){
    var el=document.getElementById('tw-plot');if(!el||!window.Plotly)return;
    var dk=_dark(),bg=dk?'#1e2228':'#fff',fg=dk?'#e0e0e0':'#333',gc=dk?'rgba(255,255,255,0.07)':'rgba(0,0,0,0.07)';
    var segs=_fit(_pts,_D,0,1);
    document.getElementById('tw-dval').textContent=_D;
    document.getElementById('tw-leaves').textContent=segs.length;
    document.getElementById('tw-params').textContent=2*segs.length-1;
    var xs=segs.map(function(s){return s.x0;}).concat([1]);
    var ys=segs.map(function(s){return s.y;}).concat([segs[segs.length-1].y]);
    Plotly.react(el,[
      {x:_xF,y:_yF,mode:'lines',name:'true f(x)',line:{color:'#A4CE8B',width:1.8,dash:'dot'},showlegend:true},
      {x:xs,y:ys,mode:'lines',name:'tree fit (D='+_D+')',line:{color:'#86BCBD',width:2.5,shape:'hv'},showlegend:true},
      {x:_pts.map(function(p){return p[0];}),y:_pts.map(function(p){return p[1];}),
       mode:'markers',name:'data',marker:{color:'#BA5A5A',size:7,line:{color:'#fff',width:1}},showlegend:true},
    ],{paper_bgcolor:bg,plot_bgcolor:bg,font:{color:fg,size:10.5},margin:{t:10,b:38,l:38,r:10},
       xaxis:{range:[0,1],title:'x',gridcolor:gc,zerolinecolor:gc},
       yaxis:{range:[-2,2],title:'y',gridcolor:gc,zerolinecolor:gc},
       legend:{x:0.01,y:0.99,bgcolor:'transparent',font:{size:9.5}},
    },{displayModeBar:false,responsive:true});
  }
  window.twSetD=function(d){_D=+d;_draw();};
  function _ep(cb){if(window.Plotly){cb();return;}var s=document.getElementById('plotly-cdn');if(!s){s=document.createElement('script');s.id='plotly-cdn';s.src='https://cdn.plot.ly/plotly-2.27.0.min.js';document.head.appendChild(s);}s.addEventListener('load',cb);}
  function _init(){if(!document.getElementById('tw-plot'))return;_ep(_draw);}
  if(typeof document$!=='undefined'){document$.subscribe(function(){setTimeout(_init,80);});}
  else if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',_init);}
  else{setTimeout(_init,80);}
})();
</script>

## 5.2 Boosted Trees

**Boosting** converts a sequence of weak learners (shallow trees) into a strong model by fitting each new tree to the residuals left by the current ensemble. Let $\hat{h}_b(\mathbf{x}) = \sum_{b'=1}^b f_{b'}(\mathbf{x}, \omega_{b'})$ denote the ensemble after $b$ rounds, where $\omega_b = (s, v_L, v_R)_b$ collects the parameters of tree $b$. At round $b+1$, define the residuals

$$r_{b}^{(p)} = y^{(p)} - \hat{h}_b(\mathbf{x}^{(p)}), \qquad p = 1, \ldots, P,$$

and train tree $f_{b+1}$ to minimize the sum of squared residuals $\sum_p (r_b^{(p)} - f_{b+1}(\mathbf{x}^{(p)}))^2$. The ensemble is then updated: $\hat{h}_{b+1} = \hat{h}_b + f_{b+1}$. This continues for $B$ rounds, yielding

$$\hat{h}_B(\mathbf{x}) = \sum_{b=1}^B f_b(\mathbf{x}, \omega_b).$$

Each new tree corrects the mistakes of all previous trees. Boosting reduces bias by iterative refinement; combined with early stopping, it is the dominant approach for structured tabular data (implemented in XGBoost, LightGBM, etc.).

## 5.3 Bagging and Random Forests

**Bagging** (bootstrap aggregating) is an ensemble method that trains many models in parallel on different bootstrap samples of the training set (random draws with replacement) and averages their predictions. Unlike boosting, bagging reduces variance rather than bias, and the base models are independent so training is embarrassingly parallel.

A **random forest** is bagged decision trees with an additional randomization: at each split, only a random subset of $\sqrt{N}$ out of $N$ input features is considered as split candidates (where $N$ is the total feature dimensionality). This decorrelates the trees: if one feature dominates, individual trees no longer all exploit it in the same way, which further reduces variance. Random forests are robust, require minimal hyperparameter tuning, and provide a natural out-of-bag error estimate (using the samples not drawn for each tree).

## 5.4 Feed-Forward Neural Networks

A **feed-forward neural network (FFNN)** is a composition of parameterized nonlinear transformations arranged in layers. Let the input layer be $\mathbf{f}^{(0)}(\mathbf{x}) = \mathbf{x} \in \mathbb{R}^N$. The network has $L$ hidden layers followed by an output layer. The $i$-th unit in layer $k \in \{1, \ldots, L\}$ computes

$$f_i^{(k)}(\mathbf{x}, \omega^{(k)}) = \sigma\!\left(b_i^{(k)} + \sum_j w_{i,j}^{(k)} f_j^{(k-1)}(\mathbf{x}, \omega^{(k-1)})\right),$$

where $b_i^{(k)}$ is a bias for unit $i$ in layer $k$, $w_{i,j}^{(k)}$ is the weight from unit $j$ in layer $k-1$ to unit $i$ in layer $k$, and $\sigma(\cdot)$ is a pointwise nonlinear **activation function**. All parameters for layer $k$ are collected in $\omega^{(k)}$. The network depth equals the number of recursive applications of this formula; each application constitutes one hidden layer. The output layer applies no activation function for regression (raw linear output) or a softmax for classification.

The entire parameter set is $\omega = \{\omega^{(1)}, \ldots, \omega^{(L)}, \omega^{(\text{out})}\}$. The **universal approximation theorem** (Cybenko 1989; Hornik 1991) states: for any continuous function $f: [0,1]^N \to \mathbb{R}$, any $\varepsilon > 0$, and any continuous non-polynomial activation $\sigma$, there exists a single-hidden-layer network $\hat{f}$ such that $\sup_{\mathbf{x}} |\hat{f}(\mathbf{x}) - f(\mathbf{x})| < \varepsilon$. The theorem guarantees existence but says nothing about how many units are needed, how to find the weights, or whether gradient descent will converge to such a solution. Deeper networks are more *parameter-efficient*: functions that require exponentially many units in a shallow network can often be represented with polynomially many units arranged in $O(\log(1/\varepsilon))$ layers, because each layer can learn reusable intermediate features rather than reconstructing everything from scratch.

<div id="nn-widget" style="margin:1.5rem 0;display:flex;flex-direction:column;align-items:center;">
  <div style="display:flex;gap:0.8rem;align-items:center;margin-bottom:0.55rem;flex-wrap:wrap;justify-content:center;font-size:0.84rem;">
    <label>N (inputs):&nbsp;<input type="range" id="nn-inp" min="1" max="5" value="2" step="1"
      oninput="nnDraw()" style="width:60px;accent-color:#A4CE8B;vertical-align:middle;"> <b id="nn-inp-v">2</b></label>
    <label>L (hidden layers):&nbsp;<input type="range" id="nn-hl" min="1" max="4" value="1" step="1"
      oninput="nnDraw()" style="width:60px;accent-color:#86BCBD;vertical-align:middle;"> <b id="nn-hl-v">1</b></label>
    <label>U (width):&nbsp;<input type="range" id="nn-w" min="1" max="7" value="2" step="1"
      oninput="nnDraw()" style="width:60px;accent-color:#86BCBD;vertical-align:middle;"> <b id="nn-w-v">2</b></label>
    <label>&sigma;:&nbsp;<select id="nn-act" onchange="nnDraw()"
      style="font-size:0.82rem;padding:2px 6px;border-radius:4px;border:1px solid rgba(128,128,128,0.4);">
      <option value="sigma">&sigma; (generic)</option>
      <option value="tanh">tanh</option>
      <option value="relu">ReLU</option>
    </select></label>
  </div>
  <svg id="nn-svg" width="100%" style="max-width:640px;height:255px;" viewBox="0 0 640 255"></svg>
  <div style="display:flex;gap:1.1rem;margin-top:0.3rem;font-size:0.79rem;flex-wrap:wrap;justify-content:center;">
    <span><span style="display:inline-block;width:11px;height:11px;background:#A4CE8B;border-radius:50%;vertical-align:middle;margin-right:3px;"></span>Input</span>
    <span><span style="display:inline-block;width:11px;height:11px;background:#86BCBD;border-radius:50%;vertical-align:middle;margin-right:3px;"></span>Hidden</span>
    <span><span style="display:inline-block;width:11px;height:11px;background:#BA5A5A;border-radius:50%;vertical-align:middle;margin-right:3px;"></span>Output</span>
    <span><span style="display:inline-block;width:10px;height:10px;background:#555;border-radius:2px;vertical-align:middle;margin-right:3px;"></span>Bias unit (value = 1)</span>
    <span style="opacity:0.55;font-style:italic;">dashed = bias connections; edge weight = b<sub>i</sub><sup>(k)</sup></span>
  </div>
  <div id="nn-params" style="font-size:0.79rem;opacity:0.62;margin-top:0.28rem;text-align:center;"></div>
  <div id="nn-formula" style="margin-top:0.8rem;width:100%;max-width:700px;padding:0.65rem 1rem 0.75rem;border-radius:6px;overflow-x:auto;line-height:2.6;"></div>
</div>

<script>
(function(){
  function _dk(){return document.body&&document.body.getAttribute('data-md-color-scheme')==='slate';}
  function _el(tag,at,tx){
    var e=document.createElementNS('http://www.w3.org/2000/svg',tag);
    for(var k in at)e.setAttribute(k,at[k]);
    if(tx!==undefined)e.textContent=tx;
    return e;
  }

  function drawSVG(N,L,U){
    var svg=document.getElementById('nn-svg');if(!svg)return;
    svg.innerHTML='';
    var W=640,H=255,dk=_dk();
    var eCol=dk?'rgba(180,180,180,0.15)':'rgba(70,70,70,0.12)';
    var bCol=dk?'rgba(247,228,155,0.45)':'rgba(150,100,15,0.4)';
    var nodeBd=dk?'#1a1e24':'#fff';
    var lblC=dk?'#999':'#555';
    var biasF=dk?'#888':'#444';

    var struct=[N];for(var k=0;k<L;k++)struct.push(U);struct.push(1);
    var nC=struct.length;
    var BSY=13,BSZ=9; // bias square: center y, half-size
    var TPAD=BSY+BSZ+10,BPAD=22,AH=H-TPAD-BPAD;
    var maxN=Math.max.apply(null,struct);
    var R=Math.max(5,Math.min(13,Math.floor(AH/(2*(maxN+1)))));
    var xs=struct.map(function(_,i){return W*(i+1)/(nC+1);});
    function nY(n,i){return TPAD+AH*(i+1)/(n+1);}

    // regular edges
    for(var ci=0;ci<nC-1;ci++){
      var nA=struct[ci],nB=struct[ci+1],xA=xs[ci],xB=xs[ci+1];
      for(var a=0;a<nA;a++){var yA=nY(nA,a);
        for(var b=0;b<nB;b++)svg.appendChild(_el('line',{x1:xA,y1:yA,x2:xB,y2:nY(nB,b),stroke:eCol,'stroke-width':'1'}));
      }
    }
    // bias edges (dashed, bias square → next-layer nodes)
    for(var ci=0;ci<nC-1;ci++){
      var nB=struct[ci+1],xB=xs[ci+1];
      for(var b=0;b<nB;b++)
        svg.appendChild(_el('line',{x1:xs[ci],y1:BSY+BSZ,x2:xB,y2:nY(nB,b)-R,
          stroke:bCol,'stroke-width':'0.9','stroke-dasharray':'3,3'}));
    }
    // regular nodes
    var fills=['#A4CE8B','#86BCBD','#BA5A5A'];
    for(var ci=0;ci<nC;ci++){
      var n=struct[ci],x=xs[ci];
      var f=ci===0?fills[0]:(ci===nC-1?fills[2]:fills[1]);
      for(var ni=0;ni<n;ni++)
        svg.appendChild(_el('circle',{cx:x,cy:nY(n,ni),r:R,fill:f,stroke:nodeBd,'stroke-width':'2'}));
    }
    // bias squares
    for(var ci=0;ci<nC-1;ci++){
      var x=xs[ci];
      svg.appendChild(_el('rect',{x:x-BSZ,y:BSY-BSZ,width:BSZ*2,height:BSZ*2,fill:biasF,rx:'2'}));
      svg.appendChild(_el('text',{x:x,y:BSY+4,'text-anchor':'middle','font-size':'9','font-weight':'bold',fill:'#fff'},'1'));
    }
    // labels
    var lbls=['Input'];for(var k=0;k<L;k++)lbls.push(L===1?'Hidden':'H'+(k+1));lbls.push('Output');
    for(var ci=0;ci<nC;ci++)
      svg.appendChild(_el('text',{x:xs[ci],y:H-4,'text-anchor':'middle','font-size':'10',fill:lblC},lbls[ci]));
  }

  function iRange(U){
    if(U===1)return 'i=1';
    if(U===2)return 'i=1,\\,2';
    return 'i=1,\\ldots,'+U;
  }

  function buildFormula(N,L,U,aSym){
    var lines=[],EXIN=N<=4,EXHID=U<=4,EXOUT=U<=5;
    // Layer 1
    var in1;
    if(EXIN){
      in1='b_i^{(1)}';
      for(var j=1;j<=N;j++)in1+=' + w_{i,'+j+'}^{(1)}\\,x_{'+j+'}';
    } else {
      in1='b_i^{(1)} + {\\textstyle\\sum_{j=1}^{'+N+'}}\\, w_{i,j}^{(1)}\\,x_j';
    }
    lines.push('f_i^{(1)} = '+aSym+'\\!\\left('+in1+'\\right),\\quad '+iRange(U));
    // Layers 2..L
    if(L>1){
      var hid;
      if(EXHID){
        hid='b_i^{(k)}';
        for(var j=1;j<=U;j++)hid+=' + w_{i,'+j+'}^{(k)}\\,f_{'+j+'}^{(k-1)}';
      } else {
        hid='b_i^{(k)} + {\\textstyle\\sum_{j=1}^{'+U+'}}\\, w_{i,j}^{(k)}\\,f_j^{(k-1)}';
      }
      var kr=L===2?'k=2':'k=2,\\ldots,'+L;
      lines.push('f_i^{(k)} = '+aSym+'\\!\\left('+hid+'\\right),\\quad '+kr+',\\;'+iRange(U));
    }
    // Output
    var out;
    if(EXOUT){
      out='b';
      for(var ui=1;ui<=U;ui++)out+=' + w_{'+ui+'}\\,f_{'+ui+'}^{('+L+')}';
    } else {
      out='b + {\\textstyle\\sum_{u=1}^{'+U+'}}\\, w_u\\,f_u^{('+L+')}';
    }
    lines.push('\\hat{f}(\\mathbf{x},\\boldsymbol{\\Theta}) = '+out);
    return lines;
  }

  window.nnDraw=function(){
    var N=+document.getElementById('nn-inp').value;
    var L=+document.getElementById('nn-hl').value;
    var U=+document.getElementById('nn-w').value;
    var asel=(document.getElementById('nn-act')||{}).value||'sigma';
    var aSym={sigma:'\\sigma',tanh:'\\tanh',relu:'\\operatorname{ReLU}'}[asel]||'\\sigma';
    document.getElementById('nn-inp-v').textContent=N;
    document.getElementById('nn-hl-v').textContent=L;
    document.getElementById('nn-w-v').textContent=U;
    var params=(N+1)*U+(L-1)*(U+1)*U+(U+1);
    document.getElementById('nn-params').textContent=params+' trainable parameters';
    drawSVG(N,L,U);
    var fEl=document.getElementById('nn-formula');if(!fEl)return;
    var dk=_dk();
    fEl.style.background=dk?'rgba(255,255,255,0.05)':'rgba(0,0,0,0.04)';
    fEl.style.border='1px solid '+(dk?'rgba(255,255,255,0.1)':'rgba(0,0,0,0.09)');
    var lines=buildFormula(N,L,U,aSym);
    if(window.katex){
      fEl.innerHTML=lines.map(function(line){
        return '<div>'+katex.renderToString('\\displaystyle '+line,{throwOnError:false})+'</div>';
      }).join('');
    } else {
      fEl.innerHTML='<pre style="font-size:0.77rem;white-space:pre-wrap;">'+lines.join('\n')+'</pre>';
    }
  };

  function _init(){if(document.getElementById('nn-svg'))window.nnDraw();}
  if(typeof document$!=='undefined'){document$.subscribe(function(){setTimeout(_init,80);});}
  else if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',_init);}
  else{setTimeout(_init,80);}
})();
</script>

[TensorFlow Playground](https://playground.tensorflow.org) provides an interactive browser-based visualization of how depth, width, activation choice, and regularization affect the decision boundary of a network trained on 2D data, useful for building intuition before working through the math.

## 5.5 Activation Functions

The activation function $\sigma$ is the source of nonlinearity. Four standard choices are:

**Sigmoid**: $\sigma(x) = 1/(1 + e^{-x})$. Output lies in $(0, 1)$, making it a natural probability model. Saturates for large $|x|$, causing vanishing gradients in deep networks.

**Hyperbolic tangent**: $\tanh(x) = (e^x - e^{-x})/(e^x + e^{-x})$. Output in $(-1, 1)$, zero-centered (unlike sigmoid), but still saturates.

**Rectified Linear Unit (ReLU)**: $\text{ReLU}(x) = \max(0, x)$. Gradient is exactly 1 for $x > 0$, solving the vanishing-gradient problem. However, whenever a unit's pre-activation $z < 0$ its output is exactly zero and its gradient $\text{ReLU}'(z) = 0$ as well, so the weights feeding that unit receive no update and the unit stays permanently silent. This is the **dying neuron** (or *dead ReLU*) problem. It is triggered by a large learning rate that drives a unit negative in one step, or by an initialisation that starts many units below zero. Common fixes are activations with a small negative-side gradient (ELU below, or *Leaky ReLU* $\max(\alpha x,\, x)$ with $\alpha \approx 0.01$) paired with proper weight initialisation.

**Exponential Linear Unit (ELU)**: $\text{ELU}(x) = x$ if $x \geq 0$, and $\alpha(e^x - 1)$ if $x < 0$, where $\alpha > 0$ is a hyperparameter. ELU is smooth everywhere and has a nonzero gradient for $x < 0$, mitigating the dying-neuron issue while retaining the non-saturating behavior of ReLU for positive inputs.

**Weight initialisation** is a critical but easily overlooked design choice. Setting all weights to zero makes every unit compute identical gradients, producing symmetric updates that never break symmetry: the network fails to learn. Random initialisation breaks symmetry, but the scale matters. If initial weights are too large, pre-activations grow exponentially with depth, saturating activations and destroying gradients. If too small, activations collapse toward zero at the same rate. *Xavier (Glorot) initialisation* draws each weight from $\mathcal{N}\!\left(0,\,\tfrac{2}{n_\text{in}+n_\text{out}}\right)$, where $n_\text{in}$ and $n_\text{out}$ are the fan-in and fan-out of the layer. This keeps the variance of activations and gradients roughly constant through tanh or sigmoid layers. *He (Kaiming) initialisation* uses $\mathcal{N}\!\left(0,\,\tfrac{2}{n_\text{in}}\right)$, accounting for the fact that ReLU zeroes out roughly half its inputs on average, and is the default for ReLU networks. Biases are typically initialised to zero.

<div id="act-widget" style="margin:1.5rem 0;">
  <div style="display:flex;gap:0.45rem;flex-wrap:wrap;margin-bottom:0.45rem;justify-content:center;">
    <button id="act-btn-0" onclick="actToggle(0)"
      style="padding:4px 15px;border-radius:5px;border:2px solid #BA5A5A;background:#BA5A5A;color:#fff;font-size:0.82rem;cursor:pointer;transition:opacity 0.15s;font-weight:600;">Sigmoid</button>
    <button id="act-btn-1" onclick="actToggle(1)"
      style="padding:4px 15px;border-radius:5px;border:2px solid #4d9e6b;background:#A4CE8B;color:#1a3a26;font-size:0.82rem;cursor:pointer;transition:opacity 0.15s;font-weight:600;">Tanh</button>
    <button id="act-btn-2" onclick="actToggle(2)"
      style="padding:4px 15px;border-radius:5px;border:2px solid #4a8a8c;background:#86BCBD;color:#fff;font-size:0.82rem;cursor:pointer;transition:opacity 0.15s;font-weight:600;">ReLU</button>
    <button id="act-btn-3" onclick="actToggle(3)"
      style="padding:4px 15px;border-radius:5px;border:2px solid #b8a800;background:#F7E49B;color:#3a3200;font-size:0.82rem;cursor:pointer;transition:opacity 0.15s;font-weight:600;">ELU (&alpha;=1)</button>
  </div>
  <div style="font-size:0.79rem;opacity:0.62;text-align:center;margin-bottom:0.35rem;">Click a button to toggle that function on/off.</div>
  <div id="act-plot" style="height:280px;"></div>
  <div style="font-size:0.8rem;opacity:0.65;margin-top:0.3rem;text-align:center;">ELU uses &alpha;&nbsp;=&nbsp;1. Note saturation of sigmoid and tanh vs the unbounded positive regime of ReLU/ELU.</div>
</div>

<script>
(function(){
  var _vis=[true,true,true,true];
  function _dark(){return document.body&&document.body.getAttribute('data-md-color-scheme')==='slate';}
  function _draw(){
    var el=document.getElementById('act-plot');if(!el||!window.Plotly)return;
    var dk=_dark(),bg=dk?'#1e2228':'#fff',fg=dk?'#e0e0e0':'#333',gc=dk?'rgba(255,255,255,0.07)':'rgba(0,0,0,0.07)';
    var x=Array.from({length:200},function(_,i){return -4+8*i/199;});
    Plotly.react(el,[
      {x:x,y:x.map(function(xi){return 1/(1+Math.exp(-xi));}),mode:'lines',name:'Sigmoid',line:{color:'#BA5A5A',width:2.5},visible:_vis[0]?true:'legendonly'},
      {x:x,y:x.map(function(xi){return Math.tanh(xi);}),mode:'lines',name:'Tanh',line:{color:'#A4CE8B',width:2.5},visible:_vis[1]?true:'legendonly'},
      {x:x,y:x.map(function(xi){return Math.max(0,xi);}),mode:'lines',name:'ReLU',line:{color:'#86BCBD',width:2.5},visible:_vis[2]?true:'legendonly'},
      {x:x,y:x.map(function(xi){return xi>=0?xi:Math.exp(xi)-1;}),mode:'lines',name:'ELU (a=1)',line:{color:'#F7E49B',width:2.5},visible:_vis[3]?true:'legendonly'},
    ],{paper_bgcolor:bg,plot_bgcolor:bg,font:{color:fg,size:10.5},margin:{t:10,b:38,l:38,r:10},
       xaxis:{title:'x',range:[-4,4],gridcolor:gc,zerolinecolor:gc,zeroline:true,zerolinewidth:1.5},
       yaxis:{title:'activation',range:[-1.5,4],gridcolor:gc,zerolinecolor:gc,zeroline:true,zerolinewidth:1.5},
       showlegend:false,
    },{displayModeBar:false,responsive:true});
  }
  window.actToggle=function(i){
    var el=document.getElementById('act-plot');if(!el||!window.Plotly)return;
    _vis[i]=!_vis[i];
    var btn=document.getElementById('act-btn-'+i);
    if(btn)btn.style.opacity=_vis[i]?'1':'0.28';
    Plotly.restyle(el,{visible:[_vis[i]?true:'legendonly']},[i]);
  };
  function _ep(cb){if(window.Plotly){cb();return;}var s=document.getElementById('plotly-cdn');if(!s){s=document.createElement('script');s.id='plotly-cdn';s.src='https://cdn.plot.ly/plotly-2.27.0.min.js';document.head.appendChild(s);}s.addEventListener('load',cb);}
  function _init(){if(!document.getElementById('act-plot'))return;_ep(_draw);}
  if(typeof document$!=='undefined'){document$.subscribe(function(){setTimeout(_init,80);});}
  else if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',_init);}
  else{setTimeout(_init,80);}
})();
</script>

## 5.6 Skip Connections and Residual Networks

Training very deep networks (tens or hundreds of layers) is difficult because gradients must flow back through every layer without vanishing. **Skip (residual) connections**, introduced by He et al. (*CVPR*, 2016), add the input of a block directly to its output:

$$f_i^{(k)}(\mathbf{x}) = f_i^{(k-1)}(\mathbf{x}) + \sigma\!\left(b_i^{(k)} + \sum_j w_{i,j}^{(k)} f_j^{(k-1)}(\mathbf{x})\right).$$

The Jacobian of $f_i^{(k)}$ with respect to $f_j^{(k-1)}$ is

$$\frac{\partial f_i^{(k)}}{\partial f_j^{(k-1)}} = \delta_{ij} + \sigma'\!\left(z_i^{(k)}\right) w_{i,j}^{(k)},$$

where $z_i^{(k)} = b_i^{(k)} + \sum_j w_{i,j}^{(k)} f_j^{(k-1)}$ is the pre-activation and $\delta_{ij}$ is the Kronecker delta. The identity term $\delta_{ij}$ guarantees that the gradient is at least 1 along the diagonal even if $\sigma'$ vanishes, so gradients can propagate without exponential decay. This identity path also smooths the loss landscape and enables architectures hundreds of layers deep (ResNets, Transformers).

## 5.7 Batch Normalization

During training, the distribution of each layer's pre-activations shifts as weights change, a phenomenon called **internal covariate shift**, which slows convergence and makes the network sensitive to the choice of learning rate. **Batch normalization** addresses this by standardizing the activations of each hidden unit $j$ in layer $k$ across the current mini-batch of $P'$ examples.

For unit $j$ at layer $k$, collect activations $\{f_j^{(k)}(\mathbf{x}^{(p)})\}_{p=1}^{P'}$ and compute the mini-batch mean and variance:

$$\mu_{f_j^{(k)}} = \frac{1}{P'}\sum_{p=1}^{P'} f_j^{(k)}(\mathbf{x}^{(p)}), \qquad \sigma^2_{f_j^{(k)}} = \frac{1}{P'}\sum_{p=1}^{P'} \left(f_j^{(k)}(\mathbf{x}^{(p)}) - \mu_{f_j^{(k)}}\right)^2.$$

The normalized activation is $\hat{f}_j^{(k)} = (f_j^{(k)} - \mu_{f_j^{(k)}})/\sigma_{f_j^{(k)}}$. To prevent this normalization from permanently restricting the representational capacity of the network, two **learnable** scalar parameters $\gamma_j^{(k)}$ (scale) and $\beta_j^{(k)}$ (shift) are introduced, and the output of the batch normalization layer is

$$\tilde{f}_j^{(k)} = \gamma_j^{(k)} \hat{f}_j^{(k)} + \beta_j^{(k)}.$$

This allows the network to recover any mean and variance if that is optimal. Because mean and variance are re-computed at every weight update for every layer, batch normalization layers are typically inserted between affine transformations and activation functions. During inference, the running mean and variance accumulated over training are used rather than the current mini-batch statistics.

## 5.8 Optimization

The training loss for a network with parameters $\omega$ over $P$ data points is

$$\mathcal{L}(\omega) = \frac{1}{P} \sum_{p=1}^{P} \ell\!\left(\hat{f}(\mathbf{x}^{(p)}, \omega),\, y^{(p)}\right),$$

where $\ell$ is a pointwise loss (e.g. squared error). Deep network losses are almost always **non-convex**, so no closed-form solution exists and iterative gradient-based methods are required.

### Gradient Descent

The simplest approach computes the exact gradient over all $P$ training examples:

$$\omega_{k+1} = \omega_k - \varepsilon\, \frac{1}{P} \sum_{p=1}^{P} \nabla_\omega \ell\!\left(\hat{f}(\mathbf{x}^{(p)}, \omega_k),\, y^{(p)}\right),$$

where $\varepsilon > 0$ is the **learning rate**. Each update requires a full forward and backward pass over the dataset, which costs $\mathcal{O}(P)$ and becomes infeasible for large datasets.

### Stochastic Gradient Descent (SGD)

**Stochastic gradient descent** replaces the full gradient with a cheap estimate computed over a random mini-batch $\mathcal{B}$ of $P' \ll P$ examples:

$$g = \frac{1}{P'} \sum_{p \in \mathcal{B}} \nabla_\omega \ell\!\left(\hat{f}(\mathbf{x}^{(p)}, \omega), y^{(p)}\right), \qquad \omega_{k+1} = \omega_k - \varepsilon\, g.$$

One pass through the full dataset is called an **epoch**; a single gradient step is a **step**. Gradients of the layer composition are computed efficiently by **backpropagation** (reverse-mode automatic differentiation), which propagates error signals layer by layer from output to input.

A limitation of plain SGD is that in elongated loss landscapes the negative gradient zig-zags across the valley rather than pointing down it. The simplest fix is **momentum**, which replaces the raw gradient with an exponentially weighted moving average of past gradients:

$$d_k = \beta\, d_{k-1} + (1 - \beta)\, g_k, \qquad \omega_{k+1} = \omega_k - \varepsilon\, d_k,$$

where $\beta \approx 0.9$ is the momentum coefficient. Expanding the recursion shows that $d_k = \sum_{j=0}^k \beta^{k-j}(1-\beta)\,g_j$: each past gradient contributes with exponentially decaying weight, so the update accumulates speed in consistent directions and is dampened by noisy cancellations.

### Backpropagation

Backpropagation is the algorithm that computes $\nabla_\omega \mathcal{L}$ by applying the chain rule from output to input. Define the *pre-activation* of unit $i$ in layer $k$ as

$$z_i^{(k)} = b_i^{(k)} + \sum_j w_{i,j}^{(k)}\, f_j^{(k-1)},$$

so $f_i^{(k)} = \sigma(z_i^{(k)})$. The *error signal* for that unit is $\delta_i^{(k)} = \partial\mathcal{L}/\partial z_i^{(k)}$.

**Forward pass.** Propagate inputs through the network layer by layer, storing every $z_i^{(k)}$ and $f_i^{(k)}$.

**Output error.** For a squared loss, $\delta^{(\text{out})} = \hat{f} - y$. For cross-entropy with softmax, $\delta_i^{(\text{out})} = \hat{p}_i - y_i$. In general, $\delta^{(\text{out})} = \partial\ell/\partial z^{(\text{out})}$.

**Backward pass.** Propagate error signals from layer $L$ down to layer $1$. For each unit in layer $k$ (in reverse order):

$$\delta_i^{(k)} = \sigma'\!\left(z_i^{(k)}\right) \sum_m w_{m,i}^{(k+1)}\, \delta_m^{(k+1)}.$$

Each $\delta_i^{(k)}$ is the local activation derivative multiplied by the weighted sum of error signals from the layer above, the chain rule applied one step at a time.

**Gradient readout.** With all $\delta_i^{(k)}$ in hand, the gradients are

$$\frac{\partial\mathcal{L}}{\partial w_{i,j}^{(k)}} = \delta_i^{(k)}\, f_j^{(k-1)}, \qquad \frac{\partial\mathcal{L}}{\partial b_i^{(k)}} = \delta_i^{(k)}.$$

Both the forward and backward passes cost $\mathcal{O}(W)$ operations, where $W$ is the total number of weights. Backpropagation therefore computes the full gradient in only twice the cost of a single forward pass, which is what makes training large networks feasible.

The **vanishing gradient** problem appears when $\sigma'(z_i^{(k)})$ is consistently small: each multiplication shrinks $\delta$ further, and by the time the signal reaches early layers it has essentially disappeared. Sigmoid and tanh saturate for large $|z|$, making $\sigma' \approx 0$ there. ReLU eliminates this for positive pre-activations ($\sigma' = 1$ exactly), which is the primary reason it became the default activation for deep networks.

### Adam (Adaptive Moment Estimation)

**Adam** (short for *Adaptive Moment Estimation*, Kingma & Ba 2014) addresses a remaining weakness of momentum SGD: near saddle points the gradient magnitude approaches zero, making every step vanishingly small regardless of direction. Adam combines momentum with *per-parameter adaptive step sizes* by maintaining two exponentially weighted running averages for each parameter $i$: the first moment $d_k^i$ (mean of recent gradients) and the second moment $h_k^i$ (mean of recent squared gradients):

$$d_k^i = \beta_1 d_{k-1}^i + (1 - \beta_1)\, g_i(\omega_k), \qquad h_k^i = \beta_2 h_{k-1}^i + (1 - \beta_2)\, g_i(\omega_k)^2,$$

with defaults $\beta_1 = 0.9$ and $\beta_2 = 0.999$. The update is

$$\omega_{k+1}^i = \omega_k^i - \varepsilon\, \frac{\hat{d}_k^i}{\sqrt{\hat{h}_k^i} + \epsilon},$$

where $\hat{d}_k^i = d_k^i\,/\,(1 - \beta_1^k)$ and $\hat{h}_k^i = h_k^i\,/\,(1 - \beta_2^k)$ are bias-corrected estimates that compensate for the zero initialization of both moments. Dividing by $\sqrt{\hat{h}_k^i}$ normalizes each coordinate's step by its recent gradient magnitude: directions with consistently large gradients are slowed, and directions with small or inconsistent gradients are amplified. This is what "adaptive" means in the name: the effective learning rate is different for every parameter, automatically tuned by the geometry of the loss surface around that parameter. In practice, bias-correction factors are applied to $\hat{d}_k^i$ and $\hat{h}_k^i$ to account for their initialization at zero.

The widget below animates three optimization trajectories on the loss surface $\mathcal{L}(w_1, w_2) = w_1^2 + 25\,w_2^2$ (minimum at the origin). The high curvature in $w_2$ forces GD to use a small learning rate that barely moves $w_1$; the zig-zag in $w_2$ is the classical signature of anisotropic GD. Adam normalises each coordinate's effective step size and converges in roughly half as many iterations.

<div id="opt-widget" style="margin:1.5rem 0;">
  <div style="display:flex;gap:0.75rem;align-items:center;margin-bottom:0.55rem;justify-content:center;flex-wrap:wrap;">
    <button onclick="optPlay()" id="opt-btn"
      style="padding:5px 18px;border-radius:6px;border:1px solid #86BCBD;background:#86BCBD;color:#1a4a50;font-weight:600;cursor:pointer;font-size:0.85rem;">&#9654; Play</button>
    <button onclick="optRestart()"
      style="padding:5px 14px;border-radius:6px;border:1px solid rgba(128,128,128,0.4);background:transparent;cursor:pointer;font-size:0.85rem;">&#8635; Restart</button>
    <span id="opt-step" style="font-size:0.81rem;opacity:0.65;">Step 0 / 80</span>
  </div>
  <div id="opt-plot" style="height:340px;"></div>
  <div style="display:flex;gap:1.4rem;margin-top:0.45rem;font-size:0.81rem;flex-wrap:wrap;justify-content:center;">
    <span><span style="display:inline-block;width:22px;height:3px;background:#BA5A5A;vertical-align:middle;margin-right:4px;border-radius:2px;"></span>GD (lr=0.038)</span>
    <span><span style="display:inline-block;width:22px;height:3px;background:#F7E49B;vertical-align:middle;margin-right:4px;border-radius:2px;"></span>SGD (lr=0.038 + mini-batch noise)</span>
    <span><span style="display:inline-block;width:22px;height:3px;background:#A4CE8B;vertical-align:middle;margin-right:4px;border-radius:2px;"></span>Adam (lr=0.12, &beta;&#8321;=0.9, &beta;&#8322;=0.999)</span>
  </div>
</div>

<script>
(function(){
  function _rng(s){var st=s>>>0;return function(){st=(st*1664525+1013904223)>>>0;return st/0x100000000;};}
  // Surface: L = w1^2 + 25*w2^2, grad = (2w1, 50w2)
  function _grad(w){return [2*w[0],50*w[1]];}
  var STEPS=80,LR=0.038,LRA=0.12,B1=0.9,B2=0.999,EP=1e-8;
  var W0=[2.5,1.5];

  // GD: lr=0.038, w2 factor = 1-50*0.038 = -0.9 (strong zig-zag), w1 factor = 0.924 (slow)
  var gd=[W0.slice()];
  for(var k=0;k<STEPS;k++){var g=_grad(gd[k]);gd.push([gd[k][0]-LR*g[0],gd[k][1]-LR*g[1]]);}

  // SGD: same lr + moderate mini-batch noise
  var sgd=[W0.slice()];var r=_rng(777);
  for(var k=0;k<STEPS;k++){
    var g=_grad(sgd[k]);
    var n1=(r()-0.5)*3,n2=(r()-0.5)*12;
    sgd.push([sgd[k][0]-LR*(g[0]+n1),sgd[k][1]-LR*(g[1]+n2)]);
  }

  // Adam with bias correction
  var adam=[W0.slice()];var d1=0,d2=0,h1=0,h2=0;
  for(var k=0;k<STEPS;k++){
    var g=_grad(adam[k]),t=k+1;
    d1=B1*d1+(1-B1)*g[0];d2=B1*d2+(1-B1)*g[1];
    h1=B2*h1+(1-B2)*g[0]*g[0];h2=B2*h2+(1-B2)*g[1]*g[1];
    var dh1=d1/(1-Math.pow(B1,t)),dh2=d2/(1-Math.pow(B1,t));
    var hh1=h1/(1-Math.pow(B2,t)),hh2=h2/(1-Math.pow(B2,t));
    adam.push([adam[k][0]-LRA*dh1/(Math.sqrt(hh1)+EP),adam[k][1]-LRA*dh2/(Math.sqrt(hh2)+EP)]);
  }

  // Contour grid for L = w1^2 + 25*w2^2
  var NX=60,NY=50;
  var xR=Array.from({length:NX},function(_,i){return -3+6*i/(NX-1);});
  var yR=Array.from({length:NY},function(_,i){return -2+4*i/(NY-1);});
  var zC=yR.map(function(y){return xR.map(function(x){return Math.min(60,x*x+25*y*y);});});

  var _fr=1,_tmr=null,_ready=false;
  function _xy(path,n){return{x:path.slice(0,n).map(function(p){return p[0];}),y:path.slice(0,n).map(function(p){return p[1];})};}
  function _dark(){return document.body&&document.body.getAttribute('data-md-color-scheme')==='slate';}

  function _initPlot(){
    var el=document.getElementById('opt-plot');if(!el||!window.Plotly)return;
    var dk=_dark(),bg=dk?'#1e2228':'#fff',fg=dk?'#e0e0e0':'#333',gc=dk?'rgba(255,255,255,0.07)':'rgba(0,0,0,0.07)';
    Plotly.react(el,[
      {type:'contour',x:xR,y:yR,z:zC,colorscale:'Blues',reversescale:true,
       contours:{coloring:'heatmap',showlabels:false,start:0,end:60,size:5},
       showscale:false,opacity:0.55,hoverinfo:'skip'},
      {x:[W0[0]],y:[W0[1]],mode:'lines+markers',line:{color:'#BA5A5A',width:2.5},marker:{size:5,color:'#BA5A5A'},showlegend:false},
      {x:[W0[0]],y:[W0[1]],mode:'lines+markers',line:{color:'#F7E49B',width:2.5},marker:{size:5,color:'#F7E49B'},showlegend:false},
      {x:[W0[0]],y:[W0[1]],mode:'lines+markers',line:{color:'#A4CE8B',width:2.5},marker:{size:5,color:'#A4CE8B'},showlegend:false},
      {x:[0],y:[0],mode:'markers',marker:{symbol:'star',size:16,color:'white',line:{color:'#666',width:1.5}},showlegend:false},
    ],{paper_bgcolor:bg,plot_bgcolor:bg,font:{color:fg,size:10.5},margin:{t:10,b:42,l:44,r:10},
       xaxis:{title:'w₁',range:[-3,3],gridcolor:gc,zerolinecolor:gc},
       yaxis:{title:'w₂',range:[-2,2],gridcolor:gc,zerolinecolor:gc},
    },{displayModeBar:false,responsive:true});
    _ready=true;
  }

  function _update(){
    if(!_ready)return;
    var el=document.getElementById('opt-plot');if(!el)return;
    var n=Math.min(_fr,STEPS+1);
    var g=_xy(gd,n),s=_xy(sgd,n),a=_xy(adam,n);
    Plotly.restyle(el,{x:[g.x,s.x,a.x],y:[g.y,s.y,a.y]},[1,2,3]);
    document.getElementById('opt-step').textContent='Step '+Math.min(_fr-1,STEPS)+' / '+STEPS;
  }

  window.optPlay=function(){
    var btn=document.getElementById('opt-btn');
    if(_tmr){clearInterval(_tmr);_tmr=null;btn.innerHTML='&#9654; Play';return;}
    if(_fr>STEPS)optRestart();
    btn.innerHTML='&#9646;&#9646; Pause';
    _tmr=setInterval(function(){_fr++;_update();if(_fr>STEPS){clearInterval(_tmr);_tmr=null;btn.innerHTML='&#9654; Play';}},80);
  };
  window.optRestart=function(){
    if(_tmr){clearInterval(_tmr);_tmr=null;}
    document.getElementById('opt-btn').innerHTML='&#9654; Play';
    _fr=1;_update();
  };

  function _ep(cb){if(window.Plotly){cb();return;}var s=document.getElementById('plotly-cdn');if(!s){s=document.createElement('script');s.id='plotly-cdn';s.src='https://cdn.plot.ly/plotly-2.27.0.min.js';document.head.appendChild(s);}s.addEventListener('load',cb);}
  function _init(){if(!document.getElementById('opt-plot'))return;_ep(function(){_initPlot();_update();});}
  if(typeof document$!=='undefined'){document$.subscribe(function(){setTimeout(_init,80);});}
  else if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',_init);}
  else{setTimeout(_init,80);}
})();
</script>

**Learning-rate schedulers** adjust $\varepsilon$ during training according to a predetermined or adaptive rule. A single fixed $\varepsilon$ works poorly across an entire run: a value large enough for fast early progress will overshoot minima later, while a value small enough for precise late-stage convergence makes early training unnecessarily slow. Schedulers resolve this tension. Four standard strategies are used in practice:

- **Exponential decay** multiplies $\varepsilon$ by a constant factor $\gamma < 1$ every $n$ steps, giving $\varepsilon_k = \varepsilon_0 \gamma^{k/n}$. The rate decays steadily regardless of the loss, which is simple but inflexible.
- **ReduceLROnPlateau** monitors the validation loss and reduces $\varepsilon$ by a fixed factor whenever the loss has not improved for a specified number of epochs. It is reactive: the rate only falls when progress genuinely stalls.
- **Cosine annealing with warm restarts** follows a cosine curve from $\varepsilon_{\max}$ down to $\varepsilon_{\min}$ over a cycle of $T$ steps, then resets to $\varepsilon_{\max}$. The periodic restarts allow the optimizer to escape sharp local minima by briefly taking larger steps again before reconverging.
- **Cyclic learning rate** oscillates $\varepsilon$ between a minimum and maximum value. Counterintuitively, periodically increasing the rate can help escape saddle points and sharp minima, often yielding faster convergence than monotone decay.

## 5.9 Regularization

Neural networks are highly overparameterized and prone to overfitting. Several strategies reduce this:

**Data augmentation** exploits known symmetries of the problem to generate additional training examples synthetically. For images, this includes rotations, flips, and color jitter; for molecules, this includes permutation of equivalent atoms. Augmentation is the most principled regularization: it directly encodes prior knowledge. Care must be taken not to apply transformations that change the label (e.g. reflecting the letter "b" to "d" in handwriting recognition). A related strategy is **noise injection**: adding small random perturbations to the training inputs, or to intermediate hidden-unit activations. Input-level noise is mathematically equivalent to data augmentation; hidden-unit noise prevents the network from becoming brittle to specific activation magnitudes.

**Early stopping** trains a high-capacity model and monitors validation loss at regular intervals. Each time the validation loss reaches a new minimum, the current parameters are saved. Training is halted when the validation loss has not improved for $P_{\text{patience}}$ consecutive checkpoints, and the saved best parameters are restored. The algorithm depends on a patience hyperparameter $P_{\text{patience}}$ and a check frequency, but requires no modification to the loss or architecture.

**Dropout** addresses a specific failure mode called co-adaptation: during training, units can become mutually dependent (unit $A$ learns to correct the errors of unit $B$ and vice versa), which causes the network to overfit because that cooperation only works on training data. Dropout breaks this by randomly zeroing each non-output unit independently with probability $p_{\text{drop}}$ on every forward pass during training, so no unit can rely on any specific partner being present. This forces each unit to be individually useful and learn redundant, general-purpose representations.

At inference, all units are active and their outputs are scaled by $(1 - p_{\text{drop}})$ to match the expected activation magnitude seen during training (since on average a fraction $p_{\text{drop}}$ of inputs were zeroed). Dropout has a clean ensemble interpretation: with $N$ droppable units there are $2^N$ possible binary masks, and training with dropout samples a different sub-network at each step, all sharing weights. Inference with weight scaling approximates averaging the predictions of all $2^N$ sub-networks simultaneously. Unlike ordinary bagging, no separate training run is needed for each sub-network, making dropout computationally cheap.

**Weight decay (L2 regularization)** adds a penalty $\lambda \|\omega\|^2$ to the loss, keeping weights small and discouraging the network from relying on any single parameter. L1 regularization ($\lambda \|\omega\|_1$) promotes sparsity.

---

