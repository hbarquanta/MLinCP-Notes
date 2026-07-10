(function () {
  var SELECTOR = ".md-sidebar--primary";
  var KEY = "md-nav-collapsed";

  function setup() {
    var el = document.querySelector(SELECTOR);
    if (!el) return;

    var collapsed = localStorage.getItem(KEY) === "1";
    el.classList.toggle("md-sidebar--collapsed", collapsed);

    if (el.querySelector(".md-sidebar__toggle")) return;

    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "md-sidebar__toggle";
    btn.setAttribute("aria-label", "Toggle navigation panel");

    function render() {
      var isCollapsed = el.classList.contains("md-sidebar--collapsed");
      btn.textContent = isCollapsed ? "›" : "‹";
    }

    btn.addEventListener("click", function () {
      var isCollapsed = !el.classList.contains("md-sidebar--collapsed");
      el.classList.toggle("md-sidebar--collapsed", isCollapsed);
      localStorage.setItem(KEY, isCollapsed ? "1" : "0");
      render();
    });

    render();
    el.appendChild(btn);
  }

  document$.subscribe(setup);
})();
