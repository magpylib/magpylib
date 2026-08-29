/*
Figures written by docs/_ext/magpydocs.py.

Each one comes in two variants, one per theme, and carries both urls rather
than a src: picking one here means a reader fetches a single variant, and it
follows the theme toggle. Pyvista scenes additionally carry the scene itself,
which trame's standalone viewer loads into the Interactive Scene tab the first
time a reader opens it - on demand, so the megabyte of viewer is not fetched by
everyone who scrolls past a figure. The scene carries its own background, so
each theme has its own export to load.
*/
(function () {
    "use strict";

    function isDark() {
        return document.documentElement.dataset.theme === "dark";
    }

    function sceneUrl(scene) {
        return (isDark() && scene.dataset.viewerDark) || scene.dataset.viewer;
    }

    function applyImages() {
        var dark = isDark();
        document.querySelectorAll("img.magpy-themed").forEach(function (image) {
            var wanted = dark ? image.dataset.dark : image.dataset.light;
            if (wanted && image.getAttribute("src") !== wanted) {
                image.setAttribute("src", wanted);
            }
        });
    }

    function applyScenes() {
        document.querySelectorAll(".magpy-scene").forEach(function (scene) {
            var frame = scene.querySelector(".magpy-scene-viewer iframe");
            var wanted = sceneUrl(scene);
            if (frame && frame.getAttribute("src") !== wanted) {
                frame.setAttribute("src", wanted);
            }
        });
    }

    function apply() {
        applyImages();
        applyScenes();
    }

    document.addEventListener("change", function (event) {
        var input = event.target;
        if (!input.id || !input.id.endsWith("-interactive") || !input.checked) {
            return;
        }
        var scene = input.closest(".magpy-scene");
        var panel = scene && scene.querySelector(".magpy-scene-viewer");
        if (!panel || panel.firstChild) {
            return;
        }
        var frame = document.createElement("iframe");
        frame.src = sceneUrl(scene);
        // the panel already carries the screenshot's aspect ratio, so the
        // viewer lands in exactly the box the static tab occupied
        panel.appendChild(frame);
    });

    function start() {
        apply();
        new MutationObserver(apply).observe(document.documentElement, {
            attributes: true,
            attributeFilter: ["data-theme"],
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", start);
    } else {
        start();
    }
})();
