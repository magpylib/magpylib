import os
import pickle
import re

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import magpylib as magpy
from magpylib._src.exceptions import MagpylibBadUserInput

# pylint: disable=unnecessary-lambda
# pylint: disable=unnecessary-lambda-assignment
# pylint: disable=no-member

# # # GENERATE TESTDATA
# # N = 5
# # mags = (np.random.rand(N,6,3)-0.5)*1000
# # dims3 = np.random.rand(N,3,3)*5     # 5x cuboid
# # dims2 = np.random.rand(N,3,2)*5     # 5x cylinder
# # posos = (np.random.rand(N,23,3)-0.5)*10 #readout at 333 positions

# # angs =  (np.random.rand(N,18)-0.5)*2*10 # each step rote by max 10 deg
# # axs =   (np.random.rand(N,18,3)-0.5)
# # anchs = (np.random.rand(N,18,3)-0.5)*5.5
# # movs =  (np.random.rand(N,18,3)-0.5)*0.5
# # rvs = (np.random.rand(N,3)-.5)*0.1

# # B = []
# # for mag,dim2,dim3,ang,ax,anch,mov,poso,rv in zip(
# #        mags,dims2,dims3,angs,axs,anchs,movs,posos,rvs):
# #     rot = R.from_rotvec(rv)
# #     pm1b = magpy.magnet.Cuboid(mag[0],dim3[0])
# #     pm2b = magpy.magnet.Cuboid(mag[1],dim3[1])
# #     pm3b = magpy.magnet.Cuboid(mag[2],dim3[2])
# #     pm4b = magpy.magnet.Cylinder(mag[3],dim2[0])
# #     pm5b = magpy.magnet.Cylinder(mag[4],dim2[1])
# #     pm6b = magpy.magnet.Cylinder(mag[5],dim2[2])

# #     # 18 subsequent operations
# #     for a,aa,aaa,mv in zip(ang,ax,anch,mov):
# #         for pm in [pm1b,pm2b,pm3b,pm4b,pm5b,pm6b]:
# #             pm.move(mv).rotate_from_angax(a,aa,aaa).rotate(rot,aaa)
# #     B += [magpy.getB([pm1b,pm2b,pm3b,pm4b,pm5b,pm6b], poso, sumup=True)]
# # B = np.array(B)
# # inp = [mags,dims2,dims3,posos,angs,axs,anchs,movs,rvs,B]
# # pickle.dump(inp,open('testdata_Collection.p', 'wb'))


def test_Collection_basics():
    """test Collection fundamentals, test against magpylib2 fields"""
    # pylint: disable=pointless-statement
    # data generated below
    with open(os.path.abspath("./tests/testdata/testdata_Collection.p"), "rb") as f:
        data = pickle.load(f)
    mags, dims2, dims3, posos, angs, axs, anchs, movs, rvs, _ = data

    B1, B2 = [], []
    for mag, dim2, dim3, ang, ax, anch, mov, poso, rv in zip(
        mags, dims2, dims3, angs, axs, anchs, movs, posos, rvs
    ):
        rot = R.from_rotvec(rv)

        pm1b = magpy.magnet.Cuboid(mag[0], dim3[0])
        pm2b = magpy.magnet.Cuboid(mag[1], dim3[1])
        pm3b = magpy.magnet.Cuboid(mag[2], dim3[2])
        pm4b = magpy.magnet.Cylinder(mag[3], dim2[0])
        pm5b = magpy.magnet.Cylinder(mag[4], dim2[1])
        pm6b = magpy.magnet.Cylinder(mag[5], dim2[2])

        pm1 = magpy.magnet.Cuboid(mag[0], dim3[0])
        pm2 = magpy.magnet.Cuboid(mag[1], dim3[1])
        pm3 = magpy.magnet.Cuboid(mag[2], dim3[2])
        pm4 = magpy.magnet.Cylinder(mag[3], dim2[0])
        pm5 = magpy.magnet.Cylinder(mag[4], dim2[1])
        pm6 = magpy.magnet.Cylinder(mag[5], dim2[2])

        col1 = magpy.Collection(pm1, pm2, pm3)
        col1.add(pm4, pm5, pm6)

        # 18 subsequent operations
        for a, aa, aaa, mv in zip(ang, ax, anch, mov):
            for pm in [pm1b, pm2b, pm3b, pm4b, pm5b, pm6b]:
                pm.move(mv).rotate_from_angax(a, aa, aaa).rotate(rot, aaa)

            col1.move(mv).rotate_from_angax(a, aa, aaa, start=-1).rotate(
                rot, aaa, start=-1
            )

        B1 += [magpy.getB([pm1b, pm2b, pm3b, pm4b, pm5b, pm6b], poso, sumup=True)]
        B2 += [col1.getB(poso)]

    B1 = np.array(B1)
    B2 = np.array(B2)

    np.testing.assert_allclose(B1, B2)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ("sens_col.getB(src_col).shape", (4, 3)),
        ("src_col.getB(sens_col).shape", (4, 3)),
        ("mixed_col.getB().shape", (4, 3)),
        ("sens_col.getB(src1, src2).shape", (2, 4, 3)),
        ("src_col.getB(sens1,sens2,sens3,sens4).shape", (4, 3)),
        ("src1.getB(sens_col).shape", (4, 3)),
        ("sens1.getB(src_col).shape", (3,)),
        ("sens1.getB(mixed_col).shape", (3,)),
        ("src1.getB(mixed_col).shape", (4, 3)),
        ("src_col.getB(mixed_col).shape", (4, 3)),
        ("sens_col.getB(mixed_col).shape", (4, 3)),
        ("magpy.getB([src1, src2], [sens1,sens2,sens3,sens4]).shape", (2, 4, 3)),
        ("magpy.getB(mixed_col,mixed_col).shape", (4, 3)),
        ("magpy.getB([src1, src2], [[1,2,3],(2,3,4)]).shape", (2, 2, 3)),
        ("src_col.getB([[1,2,3],(2,3,4)]).shape", (2, 3)),
        ("src_col.getB([1,2,3]).shape", (3,)),
        ("src1.getB(np.array([1,2,3])).shape", (3,)),
    ],
)
def test_col_getB(test_input, expected):
    """testing some Collection stuff with getB"""
    src1 = magpy.magnet.Cuboid(
        magnetization=(1, 0, 1), dimension=(8, 4, 6), position=(0, 0, 0)
    )
    src2 = magpy.magnet.Cylinder(
        magnetization=(0, 1, 0), dimension=(8, 5), position=(-15, 0, 0)
    )
    sens1 = magpy.Sensor(position=(0, 0, 6))
    sens2 = magpy.Sensor(position=(0, 0, 6))
    sens3 = magpy.Sensor(position=(0, 0, 6))
    sens4 = magpy.Sensor(position=(0, 0, 6))

    sens_col = sens1 + sens2 + sens3 + sens4
    src_col = src1 + src2
    mixed_col = sens_col + src_col
    variables = {
        "np": np,
        "magpy": magpy,
        "src1": src1,
        "src2": src2,
        "sens1": sens1,
        "sens2": sens2,
        "sens3": sens3,
        "sens4": sens4,
        "sens_col": sens_col,
        "src_col": src_col,
        "mixed_col": mixed_col,
    }

    # pylint: disable=eval-used
    assert eval(test_input, variables) == expected


@pytest.mark.parametrize(
    "test_input",
    [
        "src1.getB()",
        "src1.getB(src1)",
        "magpy.getB(src1,src1)",
        "src1.getB(src_col)",
        "magpy.getB(src1,src_col)",
        "sens1.getB()",
        "magpy.getB(sens1,src1)",
        "sens1.getB(sens1)",
        "magpy.getB(sens1,sens1)",
        "magpy.getB(sens1,mixed_col)",
        "magpy.getB(sens1,src_col)",
        "sens1.getB(sens_col)",
        "magpy.getB(sens1,sens_col)",
        "mixed_col.getB(src1)",
        "magpy.getB(mixed_col,src1)",
        "mixed_col.getB(sens1)",
        "mixed_col.getB(mixed_col)",
        "mixed_col.getB(src_col)",
        "magpy.getB(mixed_col,src_col)",
        "mixed_col.getB(sens_col)",
        "src_col.getB()",
        "src_col.getB(src1)",
        "magpy.getB(src_col,src1)",
        "src_col.getB(src_col)",
        "magpy.getB(src_col,src_col)",
        "sens_col.getB()",
        "magpy.getB(sens_col,src1)",
        "sens_col.getB(sens1)",
        "magpy.getB(sens_col,sens1)",
        "magpy.getB(sens_col,mixed_col)",
        "magpy.getB(sens_col,src_col)",
        "sens_col.getB(sens_col)",
        "magpy.getB(sens_col,sens_col)",
    ],
)
def test_bad_col_getB_inputs(test_input):
    """more undocumented Collection checking"""
    # pylint: disable=eval-used

    src1 = magpy.magnet.Cuboid(
        magnetization=(1, 0, 1), dimension=(8, 4, 6), position=(0, 0, 0)
    )

    src2 = magpy.magnet.Cylinder(
        magnetization=(0, 1, 0), dimension=(8, 5), position=(-15, 0, 0)
    )

    sens1 = magpy.Sensor(position=(0, 0, 6))
    sens2 = magpy.Sensor(position=(0, 0, 6))
    sens3 = magpy.Sensor(position=(0, 0, 6))
    sens4 = magpy.Sensor(position=(0, 0, 6))

    sens_col = sens1 + sens2 + sens3 + sens4
    src_col = src1 + src2
    mixed_col = sens_col + src_col
    variables = {
        "magpy": magpy,
        "src1": src1,
        "src2": src2,
        "sens1": sens1,
        "sens2": sens2,
        "sens3": sens3,
        "sens4": sens4,
        "sens_col": sens_col,
        "src_col": src_col,
        "mixed_col": mixed_col,
    }
    with pytest.raises(MagpylibBadUserInput):
        assert eval(test_input, variables) is not None


def test_col_get_item():
    """test get_item with collections"""
    pm1 = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    pm2 = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    pm3 = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))

    col = magpy.Collection(pm1, pm2, pm3)
    assert col[1] == pm2, "get_item failed"
    assert len(col) == 3, "__len__ failed"


def test_col_getH():
    """test collection getH"""
    pm1 = magpy.magnet.Sphere((1, 2, 3), 3)
    pm2 = magpy.magnet.Sphere((1, 2, 3), 3)
    col = magpy.Collection(pm1, pm2)
    H = col.getH((0, 0, 0))
    H1 = pm1.getH((0, 0, 0))
    assert np.all(H == 2 * H1), "col getH fail"


def test_col_reset_path():
    """testing display"""
    # pylint: disable=no-member
    pm1 = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    pm2 = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    col = magpy.Collection(pm1, pm2)
    col.move([(1, 2, 3)] * 10)
    col.reset_path()
    assert col[0].position.ndim == 1, "col reset path fail"
    assert col[1].position.ndim == 1, "col reset path fail"
    assert col.position.ndim == 1, "col reset path fail"


def test_Collection_squeeze():
    """testing squeeze output"""
    pm1 = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    pm2 = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    col = magpy.Collection(pm1, pm2)
    sensor = magpy.Sensor(pixel=[(1, 2, 3), (1, 2, 3)])
    B = col.getB(sensor)
    assert B.shape == (2, 3)
    H = col.getH(sensor)
    assert H.shape == (2, 3)

    B = col.getB(sensor, squeeze=False)
    assert B.shape == (1, 1, 1, 2, 3)
    H = col.getH(sensor, squeeze=False)
    assert H.shape == (1, 1, 1, 2, 3)


def test_Collection_with_Dipole():
    """Simple test of Dipole in Collection"""
    src = magpy.misc.Dipole(moment=(1, 2, 3), position=(1, 2, 3))
    col = magpy.Collection(src)
    sens = magpy.Sensor()

    B = magpy.getB(col, sens)
    Btest = np.array([0.00303828, 0.00607656, 0.00911485])
    assert np.allclose(B, Btest)


def test_adding_sources():
    """test if all sources can be added"""
    s1 = magpy.magnet.Cuboid()
    s2 = magpy.magnet.Cylinder()
    s3 = magpy.magnet.CylinderSegment()
    s4 = magpy.magnet.Sphere()
    s5 = magpy.current.Circle()
    s6 = magpy.current.Polyline()
    s7 = magpy.misc.Dipole()
    x1 = magpy.Sensor()
    c1 = magpy.Collection()
    c2 = magpy.Collection()

    for obj in [s1, s2, s3, s4, s5, s6, s7, x1, c1]:
        c2.add(obj)

    strs = ""
    for src in c2:
        strs += str(src)[:3]

    assert strs == "CubCylCylSphCirPolDipSenCol"


def test_set_children_styles():
    """test if styles get applied"""
    src1 = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    src2 = magpy.magnet.Cylinder((1, 2, 3), (1, 2))
    col = src1 + src2
    col.set_children_styles(magnetization_show=False)
    assert (
        src1.style.magnetization.show is False
        and src1.style.magnetization.show is False
    ), """failed updating styles to children"""
    with pytest.raises(ValueError):
        col.set_children_styles(bad_input="somevalue")


def test_reprs():
    """test repr strings"""
    # pylint: disable=protected-access

    c = magpy.Collection()
    assert repr(c)[:10] == "Collection"

    s1 = magpy.magnet.Sphere((1, 2, 3), 5)
    c = magpy.Collection(s1)
    assert repr(c)[:10] == "Collection"

    x1 = magpy.Sensor()
    c = magpy.Collection(x1)
    assert repr(c)[:10] == "Collection"

    x1 = magpy.Sensor()
    s1 = magpy.magnet.Sphere((1, 2, 3), 5)
    c = magpy.Collection(s1, x1)
    assert repr(c)[:10] == "Collection"

    x1 = magpy.magnet.Cuboid(style_label="x1")
    x2 = magpy.magnet.Cuboid(style_label="x2")
    cc = x1 + x2
    rep = cc._repr_html_()
    rep = re.sub("id=[0-9]*[0-9]", "id=REGEX", rep)
    test = "<pre>Collection nolabel (id=REGEX)<br>├── Cuboid x1"
    test += " (id=REGEX)<br>└── Cuboid x2 (id=REGEX)</pre>"
    assert rep == test


def test_collection_describe():
    """test describe method"""

    x = magpy.magnet.Cuboid(style_label="x")
    y = magpy.magnet.Cuboid(style_label="y")
    z = magpy.magnet.Cuboid(style_label="z")
    u = magpy.magnet.Cuboid(style_label="u")
    c = x + y + z + u

    desc = c.describe(format="label, type", return_string=True).split("\n")
    test = [
        "Collection nolabel",
        "├── Collection nolabel",
        "│   ├── Collection nolabel",
        "│   │   ├── Cuboid x",
        "│   │   └── Cuboid y",
        "│   └── Cuboid z",
        "└── Cuboid u",
    ]
    assert test == desc

    desc = c.describe(format="label", return_string=True).split("\n")
    test = [
        "Collection",
        "├── Collection",
        "│   ├── Collection",
        "│   │   ├── x",
        "│   │   └── y",
        "│   └── z",
        "└── u",
    ]
    assert test == desc

    desc = c.describe(format="type", return_string=True).split("\n")
    test = [
        "Collection",
        "├── Collection",
        "│   ├── Collection",
        "│   │   ├── Cuboid",
        "│   │   └── Cuboid",
        "│   └── Cuboid",
        "└── Cuboid",
    ]
    assert test == desc

    desc = c.describe(format="label,type,id", return_string=True).split("\n")
    test = [
        "Collection nolabel (id=REGEX)",
        "├── Collection nolabel (id=REGEX)",
        "│   ├── Collection nolabel (id=REGEX)",
        "│   │   ├── Cuboid x (id=REGEX)",
        "│   │   └── Cuboid y (id=REGEX)",
        "│   └── Cuboid z (id=REGEX)",
        "└── Cuboid u (id=REGEX)",
    ]
    assert "".join(test) == re.sub("id=*[0-9]*[0-9]", "id=REGEX", "".join(desc))

    c = magpy.Collection(*[magpy.magnet.Cuboid() for _ in range(100)])
    c.add(*[magpy.current.Circle() for _ in range(50)])
    c.add(*[magpy.misc.CustomSource() for _ in range(25)])

    desc = c.describe(format="type+label", return_string=True).split("\n")
    test = [
        "Collection nolabel",
        "├── 100x Cuboids",
        "├── 50x Circles",
        "└── 25x CustomSources",
    ]
    assert test == desc

    x = magpy.magnet.Cuboid(style_label="x")
    y = magpy.magnet.Cuboid(style_label="y")
    cc = x + y
    desc = cc.describe(format="label, properties", return_string=True).split("\n")
    test = [
        "Collection",
        "│   • position: [0. 0. 0.] mm",
        "│   • orientation: [0. 0. 0.] degrees",
        "├── x",
        "│       • position: [0. 0. 0.] mm",
        "│       • orientation: [0. 0. 0.] degrees",
        "│       • dimension: None mm",
        "│       • magnetization: None mT",
        "└── y",
        "        • position: [0. 0. 0.] mm",
        "        • orientation: [0. 0. 0.] degrees",
        "        • dimension: None mm",
        "        • magnetization: None mT",
    ]
    assert "".join(test) == re.sub("id=*[0-9]*[0-9]", "id=REGEX", "".join(desc))

    desc = cc.describe()
    assert desc is None
