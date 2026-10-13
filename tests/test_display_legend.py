"""`legend_maxitems`: the most entries a legend lists, the rest summed up."""

import pytest

import magpylib as magpy
from magpylib._src.display.api import DisplayBackend
from magpylib._src.exceptions import MagpylibBadUserInput

BACKENDS = ["matplotlib", "plotly", "pyvista"]


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
        return [t.name for t in fig.data if t.showlegend is not False]
    if backend == "matplotlib":
        return [
            t.get_text()
            for ax in fig.axes
            if ax.get_legend() is not None
            for t in ax.get_legend().get_texts()
        ]
    legend = fig.renderer._legend  # pylint: disable=protected-access
    if legend is None:
        return []
    return [legend.GetEntryString(i) for i in range(legend.GetNumberOfEntries())]


@pytest.mark.parametrize("backend", BACKENDS)
def test_legend_up_to_the_cap_lists_everything(backend):
    """At or under the cap, every entry is listed and nothing is added."""
    fig = magpy.show(*cuboids(3), backend=backend, return_fig=True, legend_maxitems=3)
    assert [e.split(" ")[0] for e in legend_entries(backend, fig)] == [
        "c0",
        "c1",
        "c2",
    ]


@pytest.mark.parametrize("backend", BACKENDS)
def test_legend_past_the_cap_is_truncated_not_hidden(backend):
    """Past the cap the first entries stay and the rest are summed up -- the
    legend never vanishes because of one object too many."""
    fig = magpy.show(*cuboids(5), backend=backend, return_fig=True, legend_maxitems=2)
    entries = legend_entries(backend, fig)
    assert [e.split(" ")[0] for e in entries[:2]] == ["c0", "c1"]
    assert entries[2:] == ["… and 3 more"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_legend_maxitems_zero_draws_no_legend(backend):
    """0 hides the legend, with no summary entry either."""
    fig = magpy.show(*cuboids(5), backend=backend, return_fig=True, legend_maxitems=0)
    assert legend_entries(backend, fig) == []


def test_legend_maxitems_default_and_override():
    """The default is configurable, and `show()` overrides it per call."""
    magpy.defaults.display.legend.maxitems = 1
    fig = magpy.show(*cuboids(3), backend="plotly", return_fig=True)
    assert len(legend_entries("plotly", fig)) == 2  # one entry and the summary
    fig = magpy.show(*cuboids(3), backend="plotly", return_fig=True, legend_maxitems=5)
    assert len(legend_entries("plotly", fig)) == 3


def test_legend_maxitems_default_magic_underscore():
    """`legend_maxitems` reaches `legend.maxitems`, like `units_length`."""
    magpy.defaults.display.update(legend_maxitems=7)
    assert magpy.defaults.display.legend.maxitems == 7


@pytest.mark.parametrize("value", [-1, 1.5, True, "3", None])
def test_legend_maxitems_default_is_validated(value):
    """The default takes a non-negative int; None resets it."""
    if value is None:
        magpy.defaults.display.legend.maxitems = 3
        magpy.defaults.display.legend.maxitems = None
        assert magpy.defaults.display.legend.maxitems == 20
        return
    with pytest.raises(ValueError, match="maxitems"):
        magpy.defaults.display.legend.maxitems = value


@pytest.mark.parametrize("value", [-1, 1.5, True, "3"])
def test_legend_maxitems_argument_is_validated(value):
    """`show()` rejects what the default rejects."""
    with pytest.raises(MagpylibBadUserInput, match="legend_maxitems"):
        magpy.show(
            *cuboids(2), backend="matplotlib", return_fig=True, legend_maxitems=value
        )


@pytest.mark.parametrize(
    ("kwargs", "width"),
    [
        ({}, 12),  # a legend is drawn: room is made for it
        ({"legend_maxitems": 0}, 8),  # no legend: no room
        ({"style_legend_show": False}, 8),  # nothing to list: no room
    ],
)
def test_matplotlib_widens_the_figure_only_for_a_drawn_legend(kwargs, width):
    """The figure width follows the same rule the legend is drawn by."""
    fig = magpy.show(*cuboids(2), backend="matplotlib", return_fig=True, **kwargs)
    assert fig.get_size_inches()[0] == width


def test_plotly_cap_holds_in_every_animation_frame():
    """Animation frames line up with the figure's traces by index, so each one
    hides the same entries and carries the same summary."""
    objs = cuboids(4)
    for obj in objs:
        obj.move([(0, 0, 0), (0, 0, 1)])
    fig = magpy.show(
        *objs, backend="plotly", return_fig=True, animation=True, legend_maxitems=2
    )
    assert fig.data[-1].name == "… and 2 more"
    for frame in fig.frames:
        assert len(frame.data) == len(fig.data)
        assert frame.data[-1].name == "… and 2 more"
        shown = [t.name for t in frame.data if t.showlegend is not False]
        assert len(shown) == 3


def test_the_cap_leaves_the_scene_payload_whole():
    """A backend receives every trace with its own `showlegend`: the cap is
    drawn by the built-in backends, never applied to the payload. Backends
    with their own legend (a collapsible tree, say) depend on that."""
    scenes = []
    magpy.register_backend("capture", scenes.append)
    try:
        magpy.show(*cuboids(5), backend="capture", legend_maxitems=2)
    finally:
        DisplayBackend.backends.pop("capture", None)
    (scene,) = scenes
    assert scene.legend_maxitems == 2
    traces = scene.frames[0].traces
    assert sum(1 for tr in traces if tr.get("showlegend")) == 5
    assert all("…" not in tr.get("name", "") for tr in traces)
