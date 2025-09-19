import magpylib as magpy


def test_bare_init():
    """test if magpylib object can be initialized without attributes"""
    magpy.current.Circle()
    magpy.current.Polyline()
    magpy.magnet.Cuboid()
    magpy.magnet.Cylinder()
    magpy.magnet.CylinderSegment()
    magpy.magnet.Sphere()
    magpy.misc.Dipole()
    magpy.misc.CustomSource()
