/*
Keep embedded plotly figures in sync with the page they sit on.

Size: a figure is drawn at whatever width its container has at draw time, and
`responsive: true` only re-draws on a *window* resize event. A figure drawn
while the layout is still settling therefore stays clipped on the right until
the reader resizes the browser by hand.

Theme: the theme has no dark-mode story for embedded figures
(pydata/pydata-sphinx-theme#1248, closed as not planned), so a figure keeps the
light template it was exported with and glares out of a dark page. Recolor from
the theme's own CSS variables, and hand the figure back to its embedded template
- by resetting the very same properties to null - on the way back to light.

Both end up in a single relayout per figure, so a theme switch that also moves
the layout costs one redraw rather than two. A figure is held back until that
first relayout has painted, so a dark page never flashes a light figure.
*/
(function () {
    "use strict";

    // 3d scenes and 2d axes are keyed scene/scene2/..., xaxis/xaxis2/...
    var SCENE = /^scene\d*$/;
    var AXIS = /^[xy]axis\d*$/;
    var SWEEPS = 10;
    var SWEEP_MS = 500;
    var SETTLE_MS = 150;

    var watched = new WeakSet();

    // plotly 7 puts Plotly on the window; older notebook renderers loaded it
    // through require.js and stashed it on window._Plotly.
    function getPlotly() {
        return window.Plotly || window._Plotly;
    }

    function cssVar(name, fallback) {
        var style = window.getComputedStyle(document.documentElement);
        return style.getPropertyValue(name).trim() || fallback;
    }

    // null resets a property to the value in the figure's own template
    function themeUpdate(gd, dark) {
        var page = dark ? cssVar("--pst-color-background", "#14181e") : null;
        var pane = dark ? cssVar("--pst-color-on-background", "#222832") : null;
        var text = dark ? cssVar("--pst-color-text-base", "#ced6dd") : null;
        var line = dark ? cssVar("--pst-color-border", "#48566b") : null;

        var update = {
            paper_bgcolor: page,
            plot_bgcolor: pane,
            "font.color": text,
            "legend.bgcolor": dark ? "rgba(0,0,0,0)" : null,
        };
        // the animation slider rail is hard-coded in plotly's defaults rather
        // than taken from the template, so it stays near-white on its own
        (gd._fullLayout.sliders || []).forEach(function (_, i) {
            var at = "sliders[" + i + "].";
            update[at + "bgcolor"] = dark
                ? cssVar("--pst-color-surface", "#29313d")
                : null;
            update[at + "activebgcolor"] = line;
            update[at + "bordercolor"] = line;
            update[at + "tickcolor"] = text;
        });
        Object.keys(gd._fullLayout).forEach(function (key) {
            if (SCENE.test(key)) {
                ["xaxis", "yaxis", "zaxis"].forEach(function (axis) {
                    var at = key + "." + axis + ".";
                    update[at + "backgroundcolor"] = pane;
                    update[at + "gridcolor"] = line;
                    update[at + "zerolinecolor"] = line;
                    update[at + "color"] = text;
                });
            } else if (AXIS.test(key)) {
                update[key + ".gridcolor"] = line;
                update[key + ".zerolinecolor"] = line;
                update[key + ".linecolor"] = line;
                update[key + ".color"] = text;
            }
        });
        return update;
    }

    function sync(gd) {
        var plotly = getPlotly();
        var width = gd.clientWidth;
        if (!plotly || !gd._fullLayout || !width) {
            return; // not drawn yet
        }
        if (gd._magpySyncing) {
            // recorded as applied only once it has been: a theme change
            // arriving mid-relayout is picked up when that relayout lands
            return;
        }
        var wanted =
            document.documentElement.dataset.theme === "dark" ? "dark" : "light";
        // figures are exported with the light template
        var recolor = (gd._pstTheme || "light") !== wanted;
        var refit = Math.abs(gd._fullLayout.width - width) > 1;
        gd._pstTheme = wanted;
        if (!recolor && !refit) {
            return;
        }
        var update = recolor ? themeUpdate(gd, wanted === "dark") : {};
        if (refit) {
            // what Plotly.Plots.resize does internally, folded into the same
            // relayout so the two never redraw the figure one after the other
            delete gd.layout.width;
            delete gd.layout.height;
            update.autosize = true;
        }
        gd._magpySyncing = true;
        var done = function () {
            gd._magpySyncing = false;
            sync(gd); // in case the theme moved while this was painting
        };
        plotly.relayout(gd, update).then(done, done);
    }

    function watch(gd) {
        if (watched.has(gd)) {
            return;
        }
        watched.add(gd);
        if (typeof ResizeObserver === "undefined") {
            return;
        }
        var box = gd.parentElement || gd;
        var lastWidth = 0;
        var timer = null;
        new ResizeObserver(function () {
            // resizing the figure resizes the box too - only react to width
            // changes that come from the outside
            var width = box.clientWidth;
            if (!width || width === lastWidth) {
                return;
            }
            lastWidth = width;
            // one reflow arrives as a burst of callbacks, redraw once
            clearTimeout(timer);
            timer = setTimeout(function () {
                sync(gd);
            }, SETTLE_MS);
        }).observe(box);
    }

    // The stylesheet drops the theme's light panel behind plotly's outputs by
    // class rather than by :has(), which would have to be re-evaluated as
    // plotly mutates the page.
    function tagOutput(el) {
        var output = el.closest && el.closest(".cell_output div.text_html");
        if (output) {
            output.classList.add("magpy-plotly-output");
        }
    }

    function tagScriptOnlyOutputs() {
        document.querySelectorAll(".cell_output div.text_html").forEach(function (out) {
            var children = out.children;
            if (!children.length) {
                return;
            }
            for (var i = 0; i < children.length; i++) {
                if (children[i].tagName !== "SCRIPT") {
                    return;
                }
            }
            out.classList.add("magpy-plotly-output");
        });
    }

    function adopt(gd) {
        tagOutput(gd);
        if (!watched.has(gd)) {
            watch(gd);
        }
        sync(gd);
    }

    function sweep() {
        document.querySelectorAll(".plotly-graph-div").forEach(adopt);
    }

    function start() {
        tagScriptOnlyOutputs();
        sweep();
        // a figure is only findable once plotly has classed it, which can
        // land well after DOMContentLoaded
        var sweeps = 0;
        var timer = setInterval(function () {
            sweep();
            if (++sweeps >= SWEEPS) {
                clearInterval(timer);
            }
        }, SWEEP_MS);

        new MutationObserver(sweep).observe(document.documentElement, {
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
