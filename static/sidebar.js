(function () {
    const storageKey = "pharma-sidebar-collapsed";
    const toggleButtons = document.querySelectorAll(".sidebar-toggle");
    const body = document.body;

    if (!toggleButtons.length) {
        return;
    }

    function applyState(collapsed) {
        body.classList.toggle("sidebar-collapsed", collapsed);
        toggleButtons.forEach((button) => {
            button.setAttribute("aria-expanded", String(!collapsed));
            button.setAttribute("title", collapsed ? "Expand sidebar" : "Collapse sidebar");
        });
    }

    function readState() {
        try {
            return window.localStorage.getItem(storageKey) === "true";
        } catch (error) {
            return false;
        }
    }

    function saveState(collapsed) {
        try {
            window.localStorage.setItem(storageKey, String(collapsed));
        } catch (error) {
        }
    }

    applyState(readState());

    toggleButtons.forEach((button) => {
        button.addEventListener("click", () => {
            const collapsed = !body.classList.contains("sidebar-collapsed");
            applyState(collapsed);
            saveState(collapsed);
        });
    });
})();
