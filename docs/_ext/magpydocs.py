"""Helpers that run inside the notebook kernel rather than in sphinx.

myst-nb executes every page in a kernel that inherits this build's environment
- which is where the variables used here come from, see ``conf.py`` - and runs
it with the page's own directory as the working directory, the same assumption
the pages already make when they read data files out of ``_static``.

Importing this module does two things to the figures the pages produce:

* pyvista scenes are rendered the way the pyvista documentation renders them, a
  static screenshot in the page plus the scene as a small ``.vtksz`` file that a
  shared viewer loads only when a reader asks for it. Inlining the viewer with
  every scene instead - what pyvista's own ``html`` backend does - costs about
  1 MB per figure, against 9 KB for the scene alone.
* pyvista screenshots and matplotlib figures are written twice, once for each
  theme, and the page carries both urls. ``webcode/figures.js`` picks one,
  so a reader fetches a single variant and it follows the theme toggle.

Pages keep calling ``magpy.show(...)`` unchanged; ``PYVISTA_JUPYTER_BACKEND``
points pyvista here, and matplotlib arrives through a display formatter.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import os
import re
import shutil
import warnings
from pathlib import Path

from IPython.display import HTML

BACKEND = "magpylib-docs"
SCENES = "scenes"
FIGURES = "figures"
VIEWER = "static_viewer.html"

#: A scene heavier than this stays a plain screenshot. Interactive scenes are
#: only fetched on demand, but they are still build output that has to be
#: stored and served, and geometry-heavy scenes run to tens of megabytes.
MAX_SCENE_BYTES = 5_000_000

#: The theme's dark palette, for the dark twin of every figure. A scene takes
#: the colour of the panel it sits in, a matplotlib figure is drawn on
#: transparency and only needs its ink lightened.
DARK_PANEL = "#222832"  # --pst-color-on-background
DARK_INK = "#ced6dd"  # --pst-color-text-base
DARK_GRID = "#48566b"  # --pst-color-border

#: Background for the dark scene, top and bottom of the gradient. Deliberately
#: a slate rather than the page colour: a scene is mostly background, so
#: painting it the colour of the page turns the figure into a black box, and
#: anything drawn in a dark colour disappears into it. This mirrors what the
#: light scene does on a white page - a shaded card the objects sit on.
DARK_SCENE_TOP = "#3b4453"
DARK_SCENE_BOTTOM = "#242b36"

#: distinguishes the tab controls of two identical scenes on one page
_ELEMENTS = itertools.count()


def _renders_html() -> bool:
    """Whether this build can use what we emit.

    text/html only outranks the image mime types for an html build, and the
    urls written here assume the page keeps its source tree depth, which
    dirhtml does not. Anywhere else, hand back to the native output.
    """
    return os.environ.get("MAGPYLIB_DOCS_BUILDER") == "html"


def _static() -> Path:
    """Absolute path of the docs ``_static`` directory."""
    return Path(os.environ["MAGPYLIB_DOCS_STATIC"])


def _static_url() -> str:
    """``_static`` as seen from the page being executed.

    The built page sits at the same depth under the html output directory, so
    the same relative url works there.
    """
    return os.path.relpath(_static(), Path.cwd()).replace(os.sep, "/")


def _folder(name: str) -> Path:
    path = _static() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _themed_img(folder: str, name: str, ext: str, alt: str, style: str) -> str:
    """An image that follows the theme toggle.

    The ``src`` is left to ``webcode/figures.js`` so a reader only ever
    fetches the variant their theme calls for; the aspect ratio holds the space
    while it arrives, and noscript keeps the light one working without
    javascript.
    """
    url = f"{_static_url()}/{folder}/{name}"
    return (
        f'<img class="magpy-themed" alt="{alt}" style="{style}" '
        f'data-light="{url}.{ext}" data-dark="{url}-dark.{ext}">'
        f'<noscript><img src="{url}.{ext}" alt="{alt}"></noscript>'
    )


def _install_viewer(static: Path) -> None:
    """Copy trame's standalone viewer next to the scenes, once per build."""
    from trame_vtk.tools.vtksz2html import HTML_VIEWER_PATH  # noqa: PLC0415

    target = static / VIEWER
    if not target.exists():
        shutil.copy(HTML_VIEWER_PATH, target)


def _tabs(name: str, element: str, body: str, shape: str, interactive: bool) -> str:
    """The screenshot, and the scene behind a second tab if there is one.

    The markup is sphinx-design's, whose stylesheet every page already loads,
    so the tabs match the ones written by hand elsewhere in the docs.
    """
    if not interactive:
        return f'<div class="magpy-scene">{body}</div>'
    # the viewer resolves fileURL against its own location in _static, while
    # the viewer itself is addressed from the page
    viewer = f"{_static_url()}/{VIEWER}?fileURL={SCENES}/{name}"
    return (
        f'<div class="sd-tab-set magpy-scene" data-viewer="{viewer}.vtksz" '
        f'data-viewer-dark="{viewer}-dark.vtksz">'
        f'<input checked="checked" id="{element}-static" name="{element}" type="radio">'
        f'<label class="sd-tab-label" for="{element}-static">Static Scene</label>'
        f'<div class="sd-tab-content">{body}</div>'
        f'<input id="{element}-interactive" name="{element}" type="radio">'
        f'<label class="sd-tab-label" for="{element}-interactive">Interactive Scene</label>'
        f'<div class="sd-tab-content">'
        f'<div class="magpy-scene-viewer" style="aspect-ratio: {shape}"></div>'
        f"</div>"
        f"</div>"
    )


def _show_scene(plotter, screenshot=None, **kwargs):  # noqa: ARG001
    """Render one plotter as a screenshot plus an on-demand interactive scene."""
    # magpylib renders animations by opening a gif/movie on the plotter and
    # embeds the result itself; pyvista routes the plotter here afterwards all
    # the same, and exporting it would write every frame to disk for nothing.
    # off_screen cannot be used to spot these - pyvista forces it on for every
    # plotter in a notebook - so key off the writer, as pyvista's scraper does.
    animation = (
        getattr(plotter, "_gif_filename", None) is not None
        or getattr(plotter, "mwriter", None) is not None
    )
    if animation or not _renders_html():
        from pyvista.jupyter.notebook import show_static_image  # noqa: PLC0415

        return show_static_image(plotter, screenshot)

    # the plotter is handed over mid-show and has not rendered yet;
    # screenshotting it before that segfaults, as pyvista's own handler notes
    plotter.render()

    payload = plotter.export_vtksz(filename=None)
    name = hashlib.sha256(payload).hexdigest()[:16]

    scenes = _folder(SCENES)
    image = plotter.screenshot(scenes / f"{name}.png", return_img=True)
    interactive = len(payload) <= MAX_SCENE_BYTES

    # The dark twin keeps the gradient that gives the light scene its depth. A
    # screenshot reuses the last render, so the new background needs one of its
    # own, and the plotter is handed back as it arrived - a page is free to
    # show the same plotter again.
    # captured off the renderers: background_color reports only the bottom of
    # a gradient, and magpylib's scenes are gradients
    saved = [
        (r.GetBackground(), r.GetBackground2(), bool(r.GetGradientBackground()))
        for r in plotter.renderers
    ]
    try:
        plotter.set_background(DARK_SCENE_BOTTOM, top=DARK_SCENE_TOP)
        plotter.render()
        plotter.screenshot(scenes / f"{name}-dark.png")
        if interactive:
            # the scene carries its own background, so the viewer needs the
            # dark export too, or the interactive tab lights up on a dark page
            (scenes / f"{name}.vtksz").write_bytes(payload)
            (scenes / f"{name}-dark.vtksz").write_bytes(
                plotter.export_vtksz(filename=None)
            )
            _install_viewer(_static())
    finally:
        for renderer, (bottom, top, gradient) in zip(
            plotter.renderers, saved, strict=True
        ):
            renderer.SetBackground(bottom)
            renderer.SetBackground2(top)
            renderer.SetGradientBackground(gradient)
        plotter.render()

    height, width = image.shape[:2]

    shape = f"{width} / {height}"
    body = _themed_img(SCENES, name, "png", "3d scene", f"aspect-ratio: {shape}")
    # the file name is a content hash, so two identical scenes on one page
    # would share their radio ids and drive each other
    element = f"{name}-{next(_ELEMENTS)}"
    return HTML(_tabs(name, element, body, shape, interactive=interactive))


def _save_svg(fig) -> bytes:
    buffer = io.BytesIO()
    # bbox_inches matches what the inline backend does, transparency lets the
    # page - light or dark - show through. Layout warnings raised by this
    # render are ours, not the page's: they would surface as stderr blocks in
    # the docs, which the single inline render this replaces never produced.
    with warnings.catch_warnings():
        # not filtered by module: matplotlib raises these through
        # warn_external, which attributes them to this file
        warnings.simplefilter("ignore", UserWarning)
        # no Date in the metadata: the file name is a content hash, and a
        # timestamp would give the same figure a new name on every save
        fig.savefig(
            buffer,
            format="svg",
            transparent=True,
            bbox_inches="tight",
            metadata={"Date": None},
        )
    return buffer.getvalue()


def _svg_size(payload: bytes) -> tuple[float, float]:
    """Width and height of an svg in css pixels, for the aspect ratio."""
    head = payload[:400].decode("utf8", errors="replace")
    width = re.search(r'width="([\d.]+)pt"', head)
    height = re.search(r'height="([\d.]+)pt"', head)
    if not (width and height):
        # consumed as css pixels: a bare ratio here would lay the figure out
        # four pixels wide
        return (640.0, 480.0)
    return (float(width.group(1)) * 4 / 3, float(height.group(1)) * 4 / 3)


def _darken(fig) -> None:
    """Lighten a figure's ink so it reads on the dark page.

    Applied to a copy of the figure - mutating the original would change what a
    second display of it renders, and the file names are content hashes.
    """
    fig.patch.set_alpha(0)
    if fig._suptitle is not None:
        fig._suptitle.set_color(DARK_INK)
    for ax in fig.axes:
        ax.patch.set_alpha(0)
        ax.tick_params(colors=DARK_INK, which="both")
        for spine in ax.spines.values():
            spine.set_color(DARK_GRID)
        for text in (ax.title, *ax.texts, *ax.get_xticklabels(), *ax.get_yticklabels()):
            text.set_color(DARK_INK)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(DARK_PANEL)
            legend.get_frame().set_edgecolor(DARK_GRID)
            for text in legend.get_texts():
                text.set_color(DARK_INK)
        for name in ("xaxis", "yaxis", "zaxis"):
            axis = getattr(ax, name, None)
            if axis is None:
                continue
            axis.label.set_color(DARK_INK)
            # 3d axes draw their own panes and grid, out of reach of spines
            pane = getattr(axis, "pane", None)
            if pane is not None:
                pane.set_facecolor(DARK_PANEL)
                pane.set_edgecolor(DARK_GRID)
                pane.set_alpha(1)
            info = getattr(axis, "_axinfo", None)
            if info and "grid" in info:
                info["grid"]["color"] = DARK_GRID
        for text in getattr(ax, "get_zticklabels", list)():
            text.set_color(DARK_INK)


def _drop_inline_figure_output() -> None:
    """Stop the inline backend rendering a figure we are about to render.

    text/html outranks the image mime types in myst-nb, so the inline output
    would be ignored anyway - it would just cost another render per figure.
    Done lazily, because magpylib configures the inline backend on import,
    which is after this module is loaded.
    """
    from IPython import get_ipython  # noqa: PLC0415
    from matplotlib.figure import Figure  # noqa: PLC0415

    shell = get_ipython()
    if shell is None:  # pragma: no cover - not running under a kernel
        return
    if not _renders_html():
        return
    for mime in ("image/png", "image/svg+xml", "image/jpeg", "application/pdf"):
        shell.display_formatter.formatters[mime].type_printers.pop(Figure, None)


def _dark_twin(fig):
    """A darkened copy of a figure, or None if it cannot be copied.

    A pickle round-trip is matplotlib's own way of copying a figure. The copy
    is registered with pyplot like any other figure, so the caller has to close
    it: left open, the next cell's flush would display it as an output of its
    own. Darkening the original instead is not an option - it would change what
    a second display of that figure renders.
    """
    import pickle  # noqa: PLC0415

    try:
        twin = pickle.loads(pickle.dumps(fig))
    except Exception:  # noqa: BLE001 - any unpicklable artist
        return None
    # unpickling re-registers the copy with pyplot, so it is displayed in its
    # own right unless it is both marked and closed: marked, because a twin
    # handed back here would be given a twin of its own, and so on
    twin._magpydocs_twin = True
    _darken(twin)
    return twin


def _show_figure(fig):
    """Write a matplotlib figure twice and hand the page both urls."""
    if not _renders_html():
        return None  # the inline backend's own image output stands
    if getattr(fig, "_magpydocs_twin", False):
        return None  # a dark twin of ours, on its way out

    _drop_inline_figure_output()
    figures = _folder(FIGURES)
    light = _save_svg(fig)
    name = hashlib.sha256(light).hexdigest()[:16]
    (figures / f"{name}.svg").write_bytes(light)

    width, height = _svg_size(light)
    style = f"width: {width:.0f}px; max-width: 100%; aspect-ratio: {width:.0f} / {height:.0f}"

    twin = _dark_twin(fig)
    if twin is None:
        # nothing to swap to, so the light figure stands on its own
        return f'<img src="{_static_url()}/{FIGURES}/{name}.svg" alt="figure" style="{style}">'

    import matplotlib.pyplot as plt  # noqa: PLC0415

    try:
        (figures / f"{name}-dark.svg").write_bytes(_save_svg(twin))
    finally:
        plt.close(twin)
    return _themed_img(FIGURES, name, "svg", "figure", style)


def register() -> None:
    """Point pyvista and matplotlib at the renderers above."""
    import pyvista as pv  # noqa: PLC0415

    pv.register_jupyter_backend(BACKEND, _show_scene, override=True)

    # A formatter registered against Figure would not survive: magpylib calls
    # set_matplotlib_formats on import, and IPython implements that by dropping
    # every registered Figure printer, ours included. A _repr_html_ on the
    # class is looked up separately, so it stands.
    from matplotlib.figure import Figure  # noqa: PLC0415

    Figure._repr_html_ = _show_figure


register()
