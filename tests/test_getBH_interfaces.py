import numpy as np
import pytest

import magpylib as magpy
from magpylib._src.exceptions import MagpylibMissingInput

# pylint: disable=unnecessary-lambda-assignment


def test_getB_interfaces1():
    """self-consistent test of different possibilities for computing the field"""
    src = magpy.magnet.Cuboid(polarization=(1, 2, 3), dimension=(1, 2, 3))
    src.move(np.linspace((0.1, 0.2, 0.3), (1, 2, 3), 10), start=-1)
    poso = [[(-1, -1, -1)] * 2] * 2
    sens = magpy.Sensor(pixel=poso)
    B = magpy.getB(
        "Cuboid",
        (-1, -1, -1),
        position=src.position,
        polarization=(1, 2, 3),
        dimension=(1, 2, 3),
    )
    B1 = np.tile(B, (2, 2, 1, 1))
    B1 = np.swapaxes(B1, 0, 2)

    B_test = magpy.getB(src, sens)
    np.testing.assert_allclose(B1, B_test)

    B_test = src.getB(poso)
    np.testing.assert_allclose(B1, B_test)

    B_test = src.getB(sens)
    np.testing.assert_allclose(B1, B_test)

    B_test = sens.getB(src)
    np.testing.assert_allclose(B1, B_test)


def test_getB_interfaces2():
    """self-consistent test of different possibilities for computing the field"""
    src = magpy.magnet.Cuboid(polarization=(1, 2, 3), dimension=(1, 2, 3))
    src.move(np.linspace((0.1, 0.2, 0.3), (1, 2, 3), 10), start=-1)
    poso = [[(-1, -1, -1)] * 2] * 2
    sens = magpy.Sensor(pixel=poso)
    B = magpy.getB(
        "Cuboid",
        (-1, -1, -1),
        position=src.position,
        polarization=(1, 2, 3),
        dimension=(1, 2, 3),
    )

    B2 = np.tile(B, (2, 2, 2, 1, 1))
    B2 = np.swapaxes(B2, 1, 3)

    B_test = magpy.getB([src, src], sens)
    np.testing.assert_allclose(B2, B_test)

    B_test = sens.getB([src, src])
    np.testing.assert_allclose(B2, B_test)


def test_getB_interfaces3():
    """self-consistent test of different possibilities for computing the field"""
    src = magpy.magnet.Cuboid(polarization=(1, 2, 3), dimension=(1, 2, 3))
    src.move(np.linspace((0.1, 0.2, 0.3), (1, 2, 3), 10), start=-1)
    poso = [[(-1, -1, -1)] * 2] * 2
    sens = magpy.Sensor(pixel=poso)
    B = magpy.getB(
        "Cuboid",
        (-1, -1, -1),
        position=src.position,
        polarization=(1, 2, 3),
        dimension=(1, 2, 3),
    )

    B3 = np.tile(B, (2, 2, 2, 1, 1))
    B3 = np.swapaxes(B3, 0, 3)

    B_test = magpy.getB(src, [sens, sens])
    np.testing.assert_allclose(B3, B_test)

    B_test = src.getB([poso, poso])
    np.testing.assert_allclose(B3, B_test)

    B_test = src.getB([sens, sens])
    np.testing.assert_allclose(B3, B_test)


def test_getH_interfaces1():
    """self-consistent test of different possibilities for computing the field"""
    mag = (22, -33, 44)
    dim = (3, 2, 3)
    src = magpy.magnet.Cuboid(polarization=mag, dimension=dim)
    src.move(np.linspace((0.1, 0.2, 0.3), (1, 2, 3), 10), start=-1)

    poso = [[(-1, -2, -3)] * 2] * 2
    sens = magpy.Sensor(pixel=poso)

    H = magpy.getH(
        "Cuboid",
        (-1, -2, -3),
        position=src.position,
        polarization=mag,
        dimension=dim,
    )
    H1 = np.tile(H, (2, 2, 1, 1))
    H1 = np.swapaxes(H1, 0, 2)

    H_test = magpy.getH(src, sens)
    np.testing.assert_allclose(H1, H_test)

    H_test = src.getH(poso)
    np.testing.assert_allclose(H1, H_test)

    H_test = src.getH(sens)
    np.testing.assert_allclose(H1, H_test)

    H_test = sens.getH(src)
    np.testing.assert_allclose(H1, H_test)


def test_getH_interfaces2():
    """self-consistent test of different possibilities for computing the field"""
    mag = (22, -33, 44)
    dim = (3, 2, 3)
    src = magpy.magnet.Cuboid(polarization=mag, dimension=dim)
    src.move(np.linspace((0.1, 0.2, 0.3), (1, 2, 3), 10), start=-1)

    poso = [[(-1, -2, -3)] * 2] * 2
    sens = magpy.Sensor(pixel=poso)

    H = magpy.getH(
        "Cuboid",
        (-1, -2, -3),
        position=src.position,
        polarization=mag,
        dimension=dim,
    )

    H2 = np.tile(H, (2, 2, 2, 1, 1))
    H2 = np.swapaxes(H2, 1, 3)

    H_test = magpy.getH([src, src], sens)
    np.testing.assert_allclose(H2, H_test)

    H_test = sens.getH([src, src])
    np.testing.assert_allclose(H2, H_test)


def test_getH_interfaces3():
    """self-consistent test of different possibilities for computing the field"""
    mag = (22, -33, 44)
    dim = (3, 2, 3)
    src = magpy.magnet.Cuboid(polarization=mag, dimension=dim)
    src.move(np.linspace((0.1, 0.2, 0.3), (1, 2, 3), 10), start=-1)

    poso = [[(-1, -2, -3)] * 2] * 2
    sens = magpy.Sensor(pixel=poso)

    H = magpy.getH(
        "Cuboid",
        (-1, -2, -3),
        position=src.position,
        polarization=mag,
        dimension=dim,
    )

    H3 = np.tile(H, (2, 2, 2, 1, 1))
    H3 = np.swapaxes(H3, 0, 3)

    H_test = magpy.getH(src, [sens, sens])
    np.testing.assert_allclose(H3, H_test)

    H_test = src.getH([poso, poso])
    np.testing.assert_allclose(H3, H_test)

    H_test = src.getH([sens, sens])
    np.testing.assert_allclose(H3, H_test)


def test_dataframe_ouptut():
    """test pandas dataframe output"""
    max_path_len = 20
    num_of_pix = 2

    sources = [
        magpy.magnet.Cuboid(polarization=(0, 0, 1000), dimension=(1, 1, 1)).move(
            np.linspace((-4, 0, 0), (4, 0, 0), max_path_len), start=0
        ),
        magpy.magnet.Cylinder(
            polarization=(0, 1000, 0), dimension=(1, 1), style_label="Cylinder1"
        ).move(np.linspace((0, -4, 0), (0, 4, 0), max_path_len), start=0),
    ]
    pixel = np.linspace((0, 0, 0), (0, 3, 0), num_of_pix)
    sens1 = magpy.Sensor(position=(0, 0, 1), pixel=pixel, style_label="sens1")
    sens2 = sens1.copy(position=(0, 0, 3), style_label="sens2")
    sens_col = magpy.Collection(sens1, sens2)

    for field in "BH":
        cols = [f"{field}{k}" for k in "xyz"]
        df_field = getattr(magpy, f"get{field}")(
            sources, sens_col, sumup=False, output="dataframe"
        )
        BH = getattr(magpy, f"get{field}")(
            sources, sens_col, sumup=False, squeeze=False
        )
        for i in range(2):
            np.testing.assert_array_equal(
                BH[i].reshape(-1, 3),
                df_field[df_field["source"] == df_field["source"].unique()[i]][cols],
            )
            np.testing.assert_array_equal(
                BH[:, i].reshape(-1, 3),
                df_field[df_field["path"] == df_field["path"].unique()[i]][cols],
            )
            np.testing.assert_array_equal(
                BH[:, :, i].reshape(-1, 3),
                df_field[df_field["sensor"] == df_field["sensor"].unique()[i]][cols],
            )
            np.testing.assert_array_equal(
                BH[:, :, :, i].reshape(-1, 3),
                df_field[df_field["pixel"] == df_field["pixel"].unique()[i]][cols],
            )


def test_dataframe_ouptut_sumup():
    """test pandas dataframe output when sumup is True"""
    sources = [
        magpy.magnet.Cuboid(polarization=(0, 0, 1000), dimension=(1, 1, 1)),
        magpy.magnet.Cylinder(polarization=(0, 1000, 0), dimension=(1, 1)),
    ]
    df_field = magpy.getB(sources, (0, 0, 0), sumup=True, output="dataframe")
    np.testing.assert_allclose(
        df_field[["Bx", "By", "Bz"]].values,
        np.array([[-2.16489014e-14, 6.46446609e02, 6.66666667e02]]),
    )


def test_dataframe_ouptut_pixel_agg():
    """test pandas dataframe output when sumup is True"""
    src1 = magpy.magnet.Cuboid(polarization=(0, 0, 1000), dimension=(1, 1, 1))
    sens1 = magpy.Sensor(position=(0, 0, 1), pixel=np.zeros((4, 5, 3)))
    sens2 = sens1.copy(position=(0, 0, 2))
    sens3 = sens1.copy(position=(0, 0, 3))

    sources = (src1,)
    sensors = sens1, sens2, sens3
    df_field = magpy.getB(sources, sensors, pixel_agg="mean", output="dataframe")
    np.testing.assert_allclose(
        df_field[["Bx", "By", "Bz"]].values,
        np.array(
            [[0.0, 0.0, 134.78238624], [0.0, 0.0, 19.63857207], [0.0, 0.0, 5.87908614]]
        ),
    )


def test_getBH_bad_output_type():
    """test bad output in `getBH`"""
    src = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    with pytest.raises(
        ValueError,
        match=r"The `output` argument must be one of ('ndarray', 'dataframe')*.",
    ):
        src.getB((0, 0, 0), output="bad_output_type")


def test_sensor_handedness():
    """test sensor handedness"""
    k = 0.1
    N = [5, 4, 3]

    def ls(n):
        return np.linspace(-k / 2, k / 2, n)

    pixel = np.array([[x, y, z] for x in ls(N[0]) for y in ls(N[1]) for z in ls(N[2])])
    pixel = pixel.reshape((*N, 3))
    c = magpy.magnet.Cuboid(
        polarization=(1, 0, 0), dimension=(1, 1, 1), position=(0, 1, 0)
    )
    sr = magpy.Sensor(
        pixel=pixel,
        position=(-1, 0, 0),
        style_label="Sensor (right-handed)",
        style_sizemode="absolute",
    )
    sl = sr.copy(
        handedness="left",
        style_label="Sensor (left-handed)",
    )
    sc = magpy.Collection(sr, sl)
    sc.rotate_from_angax(np.linspace(0, 90, 6), "y", start=0)
    # magpy.show(c, *sc)
    B = c.getB(sc)

    assert B.shape == (6, 2, 5, 4, 3, 3)
    # second index is sensor index, ...,1:3 -> y&z from each sensor must be equal
    np.testing.assert_allclose(B[:, 0, ..., 1:3], B[:, 1, ..., 1:3])

    # second index is sensor index, ...,0 -> x from sl must opposite of x from sr
    np.testing.assert_allclose(B[:, 0, ..., 0], -B[:, 1, ..., 0])


@pytest.mark.parametrize(
    ("module", "class_", "missing_arg"),
    [
        ("magnet", "Cuboid", "dimension"),
        ("magnet", "Cylinder", "dimension"),
        ("magnet", "CylinderSegment", "dimension"),
        ("magnet", "Sphere", "diameter"),
        ("magnet", "Tetrahedron", "vertices"),
        ("magnet", "TriangularMesh", "vertices"),
        ("current", "Circle", "diameter"),
        ("current", "Polyline", "vertices"),
        ("misc", "Triangle", "vertices"),
    ],
)
def test_getB_on_missing_dimensions(module, class_, missing_arg):
    """test_getB_on_missing_dimensions"""
    with pytest.raises(
        MagpylibMissingInput,
        match=rf"Parameter `{missing_arg}` of .* must be set.",
    ):
        getattr(getattr(magpy, module), class_)().getB([0, 0, 0])


@pytest.mark.parametrize(
    ("module", "class_", "missing_arg", "kwargs"),
    [
        ("magnet", "Cuboid", "polarization", {"dimension": (1, 1, 1)}),
        ("magnet", "Cylinder", "polarization", {"dimension": (1, 1)}),
        (
            "magnet",
            "CylinderSegment",
            "polarization",
            {"dimension": (0, 1, 1, 45, 120)},
        ),
        ("magnet", "Sphere", "polarization", {"diameter": 1}),
        (
            "magnet",
            "Tetrahedron",
            "polarization",
            {"vertices": [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]},
        ),
        (
            "magnet",
            "TriangularMesh",
            "polarization",
            {
                "vertices": ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
                "faces": ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
            },
        ),
        ("current", "Circle", "current", {"diameter": 1}),
        ("current", "Polyline", "current", {"vertices": [[0, -1, 0], [0, 1, 0]]}),
        (
            "misc",
            "Triangle",
            "polarization",
            {"vertices": [(0, 0, 0), (1, 0, 0), (0, 1, 0)]},
        ),
        ("misc", "Dipole", "moment", {}),
    ],
)
def test_getB_on_missing_excitations(module, class_, missing_arg, kwargs):
    """test_getB_on_missing_excitations"""
    with pytest.raises(
        MagpylibMissingInput,
        match=rf"Parameter `{missing_arg}` of .* must be set.",
    ):
        getattr(getattr(magpy, module), class_)(**kwargs).getB([0, 0, 0])


@pytest.mark.parametrize("field", ["H", "B", "M", "J"])
def test_getHBMJ_self_consistency(field):
    """test getHBMJ self consistency"""
    sources = [
        magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1)),
        magpy.current.Circle(diameter=1, current=1),
    ]
    sens = magpy.Sensor(position=np.linspace((-1, 0, 0), (1, 0, 0), 10))
    src = sources[0]

    F1 = getattr(magpy, f"get{field}")(src, sens)
    F2 = getattr(sens, f"get{field}")(src)
    F3 = getattr(src, f"get{field}")(sens)
    F4 = getattr(magpy.Collection(src, sens), f"get{field}")()

    np.testing.assert_allclose(F1, F2)
    np.testing.assert_allclose(F1, F3)
    np.testing.assert_allclose(F1, F4)
