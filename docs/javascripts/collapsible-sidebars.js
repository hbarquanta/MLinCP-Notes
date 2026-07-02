(function () {
  var SIDEBARS = [
    { selector: ".md-sidebar--primary", key: "md-nav-collapsed", side: "right" },
    { selector: ".md-sidebar--secondary", key: "md-toc-collapsed", side: "left" }
  ];

  function setup(cfg) {
    var el = document.querySelector(cfg.selector);
    if (!el) return;

    var collapsed = localStorage.getItem(cfg.key) === "1";
    el.classList.toggle("md-sidebar--collapsed", collapsed);

    if (el.querySelector(".md-sidebar__toggle")) return;

    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "md-sidebar__toggle";
    btn.setAttribute("aria-label", "Toggle panel");
    btn.textContent = cfg.side === "right" ? "‹" : "›";

    function render() {
      var isCollapsed = el.classList.contains("md-sidebar--collapsed");
      if (cfg.side === "right") {
        btn.textContent = isCollapsed ? "›" : "‹";
      } else {
        btn.textContent = isCollapsed ? "‹" : "›";
      }
    }

    btn.addEventListener("click", function () {
      var isCollapsed = !el.classList.contains("md-sidebar--collapsed");
      el.classList.toggle("md-sidebar--collapsed", isCollapsed);
      localStorage.setItem(cfg.key, isCollapsed ? "1" : "0");
      render();
    });

    render();
    el.appendChild(btn);
  }

  document$.subscribe(function () {
    SIDEBARS.forEach(setup);
  });
})();
