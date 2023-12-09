# pylint: disable=eval-used
# pylint: disable=unused-import
import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import magpylib as magpy
from magpylib._src.display.traces_base import make_Prism

# pylint: disable=no-member

magpy.defaults.display.backend = "plotly"

# def create_compound_test_data(path=None):
#     """creates tests data for compound setters testing"""
#     setters = [
#         ("orientation=None", dict(orientation="None")),
#         ("shorter position path", dict(position="np.array([[50, 0, 100]] * 2)")),
#         (
#             "shorter orientation path",
#             dict(orientation="R.from_rotvec([[90,0,0],[0,90,0]], degrees=True)"),
#         ),
#         (
#             "longer position path",
#             dict(position="np.array(np.linspace((280.,0.,0), (280.,0.,300), 8))"),
#         ),
#         (
#             "longer orientation path",
#             dict(
#                 orientation="R.from_rotvec([[0,90*i,0] for i in range(6)], degrees=True)"
#             ),
#         ),
#     ]
#     data = {"test_names": [], "setters_inputs": [], "pos_orient_as_matrix_expected": []}
#     for setter in setters:
#         tname, kwargs = setter
#         coll = create_compound_set(**kwargs)
#         pos_orient = get_pos_orient_from_collection(coll)
#         data["test_names"].append(tname)
#         data["setters_inputs"].append(kwargs)
#         data["pos_orient_as_matrix_expected"].append(pos_orient)
#     if path is None:
#         return data
#     np.save(path, data)


# def display_compound_test_data(path):
#     """display compound test data from file"""
#     data = np.load(path, allow_pickle=True).item()
#     for kwargs in data["setters_inputs"]:
#         create_compound_set(show=True, **kwargs)


def make_wheel(Ncubes=6, height=10, diameter=36, path_len=5, label=None):
    """creates a basic Collection Compound object with a rotary arrangement of cuboid magnets"""

    def cs_lambda():
        return magpy.magnet.Cuboid(
            (1, 0, 0),
            dimension=[height] * 3,
            position=(diameter / 2, 0, 0),
        )

    s0 = cs_lambda().rotate_from_angax(
        np.linspace(0.0, 360.0, Ncubes, endpoint=False), "z", anchor=(0, 0, 0), start=0
    )
    c = magpy.Collection()
    for ind in range(Ncubes):
        s = cs_lambda()
        s.position = s0.position[ind]
        s.orientation = s0.orientation[ind]
        c.add(s)
    c.rotate_from_angax(90, "x")
    c.rotate_from_angax(
        np.linspace(90, 360, path_len), axis="z", start=0, anchor=(80, 0, 0)
    )
    c.move(np.linspace((0, 0, 0), (0, 0, 200), path_len), start=0)
    c.style.label = label

    trace = make_Prism(
        "plotly",
        base=Ncubes,
        diameter=diameter + height * 2,
        height=height * 0.5,
        opacity=0.5,
        color="blue",
    )

    c.style.model3d.data = [trace]  # pylint: disable=no-member
    return c


def create_compound_set(**kwargs):
    """creates a styled Collection Compound object with a rotary arrangement of cuboid magnets.
    A copy is created to show the difference when applying position and/or orientation setters over
    kwargs."""
    c1 = make_wheel(label="Magnetic Wheel after")
    c1.set_children_styles(
        path_show=False,
        magnetization_color_north="magenta",
        magnetization_color_south="cyan",
    )
    c2 = make_wheel(label="Magnetic Wheel before")
    c2.style.model3d.data[0].kwargs["color"] = "red"
    c2.style.model3d.data[0].kwargs["opacity"] = 0.1
    c2.set_children_styles(path_show=False, opacity=0.1)
    for k, v in kwargs.items():
        setattr(c1, k, eval(v))
    # if show:
    #     fig = go.Figure()
    #     magpy.show(c2, c1, style_path_frames=1, canvas=fig)
    #     fig.layout.title = ", ".join(f"c1.{k} = {v}" for k, v in kwargs.items())
    #     fig.show()
    return c1


def get_pos_orient_from_collection(coll):
    """returns a list of (position, orientation.as_matrix()) tuple of a collection and of its
    children"""
    pos_orient = []
    for obj in [coll] + coll.children:
        pos_orient.append((obj.position, obj.orientation.as_matrix()))
    return pos_orient


folder = "tests/testdata"
file = os.path.join(folder, "testdata_compound_setter_cases.npy")
# create_compound_test_data(file)

COMPOUND_DATA = np.load(file, allow_pickle=True).item()


@pytest.mark.parametrize(
    "setters_inputs, pos_orient_as_matrix_expected",
    list(
        zip(
            COMPOUND_DATA["setters_inputs"],
            COMPOUND_DATA["pos_orient_as_matrix_expected"],
        )
    ),
    ids=COMPOUND_DATA["test_names"],
)
def test_compound_setters(setters_inputs, pos_orient_as_matrix_expected):
    """testing of compound object setters and the effects on its children."""
    c1 = create_compound_set(**setters_inputs)
    pos_orient = get_pos_orient_from_collection(c1)
    for ind, (po, po_exp) in enumerate(zip(pos_orient, pos_orient_as_matrix_expected)):
        obj_str = "child"
        if ind == 0:  # first ind is (position, orientation.as_matrix()) of collection
            obj_str = "Collection"
        pos, orient = po
        pos_exp, orient_exp = po_exp
        err_msg = f"{obj_str} position matching failed"
        np.testing.assert_almost_equal(pos, pos_exp, err_msg=err_msg)
        err_msg = f"{obj_str}{ind if ind!=0 else ''} orientation matching failed"
        np.testing.assert_almost_equal(orient, orient_exp, err_msg=err_msg)
