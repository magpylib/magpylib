"""2D field panels: what their axes say, and which object each curve names."""

import numpy as np
import pytest

import magpylib as magpy
from magpylib._src.display.api import DisplayBackend


def cube_and_sensor():
    cube = magpy.magnet.Cuboid(dimension=(0.02, 0.02, 0.01), polarization=(0, 0, 1))
    sensor = magpy.Sensor(position=np.linspace((-0.02, 0, 0.02), (0.02, 0, 0.02), 5))
    return cube, sensor


@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "pyvista"])
def test_2d_panels_title_their_axes(backend):
    """None of the three backends titled a 2D axis: the field was in the
    legend group, and its unit only in plotly's hover text."""
    cube, sensor = cube_and_sensor()
    fig = magpy.show(
        {"objects": [cube, sensor], "col": 1},
        {"objects": [cube, sensor], "col": 2, "output": ("Bx", "Bz")},
        backend=backend,
        return_fig=True,
    )
    if backend == "plotly":
        titles = (fig.layout.xaxis.title.text, fig.layout.yaxis.title.text)
    elif backend == "matplotlib":
        (ax,) = (ax for ax in fig.axes if ax.name != "3d")
        titles = (ax.get_xlabel(), ax.get_ylabel())
    else:
        (chart,) = (c for r in fig.renderers for c in (r._charts or []))  # pylint: disable=protected-access
        titles = (chart.x_label, chart.y_label)
    assert titles == ("path index", "Bx, Bz (T)")


@pytest.mark.parametrize(
    ("output", "sumup", "y_title", "named"),
    [
        ("Bz", True, "Bz (T)", "sensor"),
        ("Hx", False, "Hx (A/m)", "cube"),
        (("Bxy", "Jz"), True, "Bxy, Jz (T)", "sensor"),  # B and J share the tesla
    ],
)
def test_a_backend_receives_what_a_2d_panel_shows(output, sumup, y_title, named):
    """The titles in `Panel.labels`, and on each curve the object it is named
    after: its sensor, or its source when sources are not summed up. 2D curves
    used to carry no `object_id`, so a click could not select a sensor."""
    cube, sensor = cube_and_sensor()
    scenes = []
    magpy.register_backend("capture2d", scenes.append)
    try:
        magpy.show(cube, sensor, output=output, sumup=sumup, backend="capture2d")
    finally:
        DisplayBackend.backends.pop("capture2d", None)
    (scene,) = scenes
    curves = [t for t in scene.frames[0].traces if t["type"] == "scatter"]
    named_id = id({"sensor": sensor, "cube": cube}[named])
    assert {t["object_id"] for t in curves} == {named_id}
    assert scene.panels[0].labels == {"x": "path index", "y": y_title}


@pytest.mark.parametrize(("sumup", "groups"), [(False, 4), (True, 2)])
def test_a_2d_legend_group_is_one_output_in_one_panel(sumup, groups):
    """Per sensor too when sources are not summed up, its title in
    `legendgrouptitle_text`. Groups were named from labels, so two sensors
    labelled alike, here both "Sensor", shared one."""
    cube, sensor = cube_and_sensor()
    twin = sensor.copy()
    scenes = []
    magpy.register_backend("capture2d", scenes.append, supports_subplots=True)
    magpy.show(
        {"objects": [cube, sensor, twin], "col": 1},
        {"objects": [cube, sensor, twin], "col": 2},
        output="Bz",
        sumup=sumup,
        backend="capture2d",
    )
    (scene,) = scenes
    curves = [t for t in scene.frames[0].traces if t["type"] == "scatter"]
    assert len(curves) == 4
    assert len({t["legendgroup"] for t in curves}) == groups
    assert all(t["legendgrouptitle_text"].startswith("Bz") for t in curves)


@pytest.mark.parametrize("output", [("Bz", "Hx"), "Bq", "Xz"])
def test_a_2d_output_is_checked(output):
    """Fields in T and in A/m cannot share an axis: outside a magnet H is B/mu0
    numerically, so one curve would be flat. 'Bq' and 'Xz' slipped past the
    old check and failed inside the field computation."""
    cube, sensor = cube_and_sensor()
    with pytest.raises(ValueError, match=r"Input output"):
        magpy.show(cube, sensor, output=output, backend="matplotlib", return_fig=True)
