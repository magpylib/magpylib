"""What a legend lists: `legend_maxitems`, the most entries, the rest summed
up; and `style.legend.show`, which objects are listed at all."""

import numpy as np
import pytest

import magpylib as magpy
from magpylib._src.display.api import DisplayBackend
from magpylib._src.exceptions import MagpylibBadUserInput


def cuboids(n):
    """`n` labelled, separately listed sources."""
    return [
        magpy.magnet.Cuboid(
            dimension=(1, 1, 1),
            polarization=(0, 0, 1),
            position=(2 * i, 0, 0),
            style_label=f"c{i}",
        )
        for i in range(n)
    ]


def legend_entries(backend, fig):
    """The texts the figure's legend lists, in order."""
    if backend == "plotly":
        # plotly lists only traces that have data
        listed = [t for t in fig.data if t.showlegend is not False and len(t.x)]
        return [t.name.split(" ")[0] for t in listed]
    if backend == "matplotlib":
        legends = [ax.get_legend() for ax in fig.axes if ax.get_legend()]
        return [t.get_text().split(" ")[0] for lg in legends for t in lg.get_texts()]
    legend = fig.renderer._legend  # pylint: disable=protected-access
    if legend is None:
        return []
    n = legend.GetNumberOfEntries()
    return [legend.GetEntryString(i).split(" ")[0] for i in range(n)]


@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "pyvista"])
@pytest.mark.parametrize(
    ("n", "maxitems", "expected"),
    [
        (3, 3, ["c0", "c1", "c2"]),  # at the cap: everything, nothing added
        (5, 2, ["c0", "c1", "…"]),  # past it: truncated, not hidden
        (5, 0, []),  # 0: no legend at all
    ],
)
def test_legend_maxitems(backend, n, maxitems, expected):
    """Past the cap the first entries stay and the rest are summed up."""
    fig = magpy.show(
        *cuboids(n), backend=backend, return_fig=True, legend_maxitems=maxitems
    )
    assert legend_entries(backend, fig) == expected


def test_legend_maxitems_default():
    """The default is configurable, `show()` overrides it, None resets it."""
    magpy.defaults.display.legend.maxitems = 1
    fig = magpy.show(*cuboids(3), backend="plotly", return_fig=True)
    assert legend_entries("plotly", fig) == ["c0", "…"]
    fig = magpy.show(*cuboids(3), backend="plotly", return_fig=True, legend_maxitems=5)
    assert legend_entries("plotly", fig) == ["c0", "c1", "c2"]
    magpy.defaults.display.legend.maxitems = None
    assert magpy.defaults.display.legend.maxitems == 20


@pytest.mark.parametrize("value", [-1, 1.5, True, "3"])
def test_legend_maxitems_is_validated(value):
    """Only a non-negative int is accepted."""
    with pytest.raises(MagpylibBadUserInput, match="legend_maxitems"):
        magpy.show(*cuboids(1), backend="matplotlib", legend_maxitems=value)


def test_plotly_cap_holds_in_every_animation_frame():
    """Frames line up with the figure's traces by index, so each one hides the
    same entries and carries the same summary."""
    objs = cuboids(4)
    for obj in objs:
        obj.move([(0, 0, 0), (0, 0, 1)])
    fig = magpy.show(
        *objs, backend="plotly", return_fig=True, animation=True, legend_maxitems=2
    )
    for frame in fig.frames:
        assert len(frame.data) == len(fig.data)
        shown = [t.name for t in frame.data if t.showlegend is not False]
        assert shown[-1] == "… and 2 more"


def test_the_cap_leaves_the_scene_payload_whole():
    """The cap is drawn by the built-in backends, never applied to the
    payload: a backend with its own legend (a collapsible tree, say) still
    receives every trace."""
    scenes = []
    magpy.register_backend("capture", scenes.append)
    try:
        magpy.show(*cuboids(5), backend="capture", legend_maxitems=2)
    finally:
        DisplayBackend.backends.pop("capture", None)
    (scene,) = scenes
    assert sum(1 for tr in scene.frames[0].traces if tr.get("showlegend")) == 5


def chart_entries(backend, fig):
    """The texts the legends of the figure's 2D panels list, in order. For
    "capture", the scene a backend is handed: the curves it should list."""
    if backend == "capture":
        curves = [t for f in fig.frames[:1] for t in f.traces if t["type"] == "scatter"]
        return [t["name"].split(" ")[0] for t in curves if t["showlegend"]]
    if backend == "plotly":
        listed = [
            t for t in fig.data if t.type == "scatter" and t.showlegend is not False
        ]
        return [t.name.split(" ")[0] for t in listed]
    if backend == "matplotlib":
        legends = [ax.get_legend() for ax in fig.axes if ax.name != "3d"]
        return [
            t.get_text().split(" ")[0] for lg in legends if lg for t in lg.get_texts()
        ]
    plots = [p for r in fig.renderers for c in (r._charts or []) for p in c.plots()]  # pylint: disable=protected-access
    return [p.label.split(" ")[0] for p in plots if p.label]


@pytest.mark.parametrize(
    ("backend", "hide", "sumup", "expected"),
    [
        # each backend draws a hidden curve its own way: pyvista used to keep
        # its label, matplotlib warned about an empty legend
        ("matplotlib", "show", True, []),
        ("plotly", "show", True, []),
        ("pyvista", "show", True, []),
        # which object decides is settled before any backend, so the scene
        # handed over is enough, and free
        ("capture", None, True, ["s1", "s2"]),
        ("capture", "s1", True, ["s2"]),
        ("capture", "collection", True, []),
        # summed up, a curve is named after its sensor; otherwise after its
        # source, whose setting then decides
        ("capture", "cube", True, ["s1", "s2"]),
        ("capture", "cube", False, []),
    ],
)
def test_2d_curves_follow_the_legend_setting(backend, hide, sumup, expected):
    """`style_legend_show=False` hid only the 3D entries of a figure with a 2D
    panel (#875, #672). A curve now follows the setting of the object it is
    named after, resolved as for 3D: show() arguments, defaults, and a parent
    collection's setting."""
    cube = magpy.magnet.Cuboid(
        dimension=(2, 2, 1), polarization=(0, 0, 1), style_label="cube"
    )
    s1, s2 = (
        magpy.Sensor(
            position=np.linspace((-2, y, 2), (2, y, 2), 5), style_label=f"s{y}"
        )
        for y in (1, 2)
    )
    sensors = magpy.Collection(s1, s2) if hide == "collection" else [s1, s2]
    target = {"s1": s1, "cube": cube, "collection": sensors}.get(hide)
    if target is not None:
        target.style.legend.show = False
    objects = [sensors, cube] if hide == "collection" else [*sensors, cube]
    if backend == "capture":
        scenes = []
        magpy.register_backend("capture", scenes.append, supports_subplots=True)
    try:
        fig = magpy.show(
            {"objects": objects, "col": 1},
            {"objects": objects, "col": 2, "output": "Hz", "sumup": sumup},
            backend=backend,
            return_fig=True,
            **({"style_legend_show": False} if hide == "show" else {}),
        )
    finally:
        DisplayBackend.backends.pop("capture", None)
    assert (
        chart_entries(backend, scenes[0] if backend == "capture" else fig) == expected
    )
