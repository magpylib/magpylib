import numpy as np
import plotly.graph_objects as go
import pytest

import magpylib as magpy
from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib.magnet import Cuboid
from magpylib.magnet import Cylinder
from magpylib.magnet import CylinderSegment
from magpylib.magnet import Sphere

magpy.defaults.display.backend = "matplotlib"
cuboid = Cuboid((1, 2, 3), (1, 2, 3))
cuboid.move(np.linspace((0.4, 0.4, 0.4), (12.4, 12.4, 12.4), 33), start=-1)
cuboid.style.model3d.showdefault = False
cuboid.style.model3d.data = [
    {
        "backend": "generic",
        "constructor": "Scatter3d",
        "kwargs": {
            "x": [-1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1],
            "y": [-1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
            "z": [-1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1],
            "mode": "lines",
        },
        "show": True,
    },
    {
        "backend": "generic",
        "constructor": "Mesh3d",
        "kwargs": {
            "i": [7, 0, 0, 0, 4, 4, 2, 6, 4, 0, 3, 7],
            "j": [0, 7, 1, 2, 6, 7, 1, 2, 5, 5, 2, 2],
            "k": [3, 4, 2, 3, 5, 6, 5, 5, 0, 1, 7, 6],
            "x": [-1, -1, 1, 1, -1, -1, 1, 1],
            "y": [-1, 1, 1, -1, -1, 1, 1, -1],
            "z": [-1, -1, -1, -1, 1, 1, 1, 1],
            "facecolor": ["red"] * 12,
        },
        "show": True,
    },
]
coll = magpy.Collection(cuboid)
coll.rotate_from_angax(45, "z")
magpy.show(
    coll,
    animation=False,
    style=dict(model3d_showdefault=False),
)
