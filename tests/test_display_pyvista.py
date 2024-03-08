import os
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import pyvista as pv

import magpylib as magpy

HAS_IMAGEIO = True
try:
    import imageio
except ModuleNotFoundError:
    HAS_IMAGEIO = False

# pylint: disable=no-member

# pylint: disable=broad-exception-caught
FFMPEG_FAILED = False
try:
    try:
        import imageio_ffmpeg

        imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as err:
        if HAS_IMAGEIO:
            imageio.plugins.ffmpeg.download()
        else:
            raise err
except Exception:  # noqa: E722
    # skip test if ffmpeg cannot be loaded
    FFMPEG_FAILED = True


def test_Cuboid_display():
    "test simple display with path"
    src = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    src.move([[i, 0, 0] for i in range(2)], start=0)
    fig = src.show(return_fig=True, style_path_numbering=True, backend="pyvista")
    assert isinstance(fig, pv.Plotter)


def test_extra_model3d():
    """test extra model 3d"""
    trace_mesh3d = {
        "constructor": "Mesh3d",
        "kwargs": {
            "x": (1, 0, -1, 0),
            "y": (-0.5, 1.2, -0.5, 0),
            "z": (-0.5, -0.5, -0.5, 1),
            "i": (0, 0, 0, 1),
            "j": (1, 1, 2, 2),
            "k": (2, 3, 3, 3),
            "opacity": 0.5,
            "facecolor": ["blue"] * 2 + ["red"] * 2,
        },
    }
    coll = magpy.Collection(position=(0, -3, 0), style_label="'Mesh3d' trace")
    coll.style.model3d.add_trace(trace_mesh3d)

    magpy.show(coll, return_fig=True, backend="pyvista")


def test_subplots():
    """Test pyvista animation"""
    # define sensor and source
    magpy.defaults.reset()
    sensor = magpy.Sensor(
        pixel=np.linspace((0, 0, -0.2), (0, 0, 0.2), 2), style_size=1.5
    )
    sensor.style.label = "Sensor1"
    cyl1 = magpy.magnet.Cylinder(
        polarization=(0.1, 0, 0), dimension=(1, 2), style_label="Cylinder1"
    )

    # define paths
    N = 2
    sensor.position = np.linspace((0, 0, -3), (0, 0, 3), N)
    cyl1.position = (4, 0, 0)
    cyl1.rotate_from_angax(angle=np.linspace(0, 300, N), start=0, axis="z", anchor=0)
    cyl2 = cyl1.copy().move((0, 0, 5))
    objs = cyl1, cyl2, sensor

    magpy.show(
        {"objects": objs, "col": 1, "output": ("Bx", "By", "Bz")},
        {"objects": objs, "col": 2},
        backend="pyvista",
        sumup=True,
        return_fig=True,
    )


def test_animation_warning():
    "animation not supported, should warn and display static"
    pl = pv.Plotter()
    src = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    with pytest.warns(UserWarning):
        src.show(canvas=pl, animation=True, backend="pyvista")


@pytest.mark.parametrize("is_notebook_result", (True, False))
@pytest.mark.parametrize("extension", ("mp4", "gif"))
@pytest.mark.parametrize("filename", (True, False))
def test_pyvista_animation(is_notebook_result, extension, filename):
    """Test pyvista animation"""
    # define sensor and source
    pv.OFF_SCREEN = True
    if sys.platform == "linux":
        pv.start_xvfb()  #  needed for unix systems or it will test will crash with fatal error
    if not HAS_IMAGEIO and extension == "gif":
        pytest.skip("Extension gif skipped because imageio failed to load")
    if FFMPEG_FAILED and extension == "mp4":
        pytest.skip("Extension mp4 skipped because ffmpeg failed to load")
    sens = magpy.Sensor()
    src = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    src.move([[0, 0, 0], [0, 0, 1]], start=0)
    objs = [src, sens]

    with patch("magpylib._src.utility.is_notebook", return_value=is_notebook_result):
        with patch("webbrowser.open"):
            try:
                temp = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
                temp += f".{extension}"
                animation_output = temp if filename else extension
                magpy.show(
                    {"objects": objs, "col": 1, "output": ("Bx", "By", "Bz")},
                    {"objects": objs, "col": 2},
                    backend="pyvista",
                    animation=True,
                    animation_output=animation_output,
                    mp4_quality=1,
                    return_fig=True,
                )
            finally:
                try:
                    os.unlink(temp)
                except FileNotFoundError:
                    # avoid exception if file is not found
                    pass
