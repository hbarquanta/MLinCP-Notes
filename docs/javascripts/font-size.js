(function () {
  var STEP = 0.1, MIN = 0.8, MAX = 1.5;
  var root = document.documentElement;

  function getBasePx() {
    if (!root.dataset.baseFontPx) {
      var prev = root.style.fontSize;
      root.style.fontSize = "";
      root.dataset.baseFontPx = parseFloat(getComputedStyle(root).fontSize);
      root.style.fontSize = prev;
    }
    return parseFloat(root.dataset.baseFontPx);
  }

  function apply(scale) {
    root.style.fontSize = (getBasePx() * scale) + "px";
    localStorage.setItem("md-font-scale", scale);
  }

  document$.subscribe(function () {
    var scale = parseFloat(localStorage.getItem("md-font-scale")) || 1;
    apply(scale);

    if (document.querySelector(".md-font-size")) return;
    var header = document.querySelector(".md-header__inner");
    var anchor = header && header.querySelector(".md-search, .md-header__source");
    if (!header || !anchor) return;

    var group = document.createElement("div");
    group.className = "md-font-size";

    var minus = document.createElement("button");
    minus.type = "button";
    minus.textContent = "−";
    minus.setAttribute("aria-label", "Decrease text size");
    minus.addEventListener("click", function () {
      scale = Math.max(MIN, +(scale - STEP).toFixed(2));
      apply(scale);
    });

    var plus = document.createElement("button");
    plus.type = "button";
    plus.textContent = "+";
    plus.setAttribute("aria-label", "Increase text size");
    plus.addEventListener("click", function () {
      scale = Math.min(MAX, +(scale + STEP).toFixed(2));
      apply(scale);
    });

    group.appendChild(minus);
    group.appendChild(plus);
    header.insertBefore(group, anchor);
  });
})();
