import numpy as np
import magpylib as mag3


def test_getB_level2_input_simple():
    """ test functionality of getB_level2 to combine various
    inputs - simple position inputs
    """
    mag = (1,2,3)
    dim_cuboid = (1,2,3)
    dim_cyl = (1,2)
    pm1 = mag3.magnet.Cuboid(mag,dim_cuboid)
    pm2 = mag3.magnet.Cuboid(mag,dim_cuboid)
    pm3 = mag3.magnet.Cylinder(mag,dim_cyl)
    pm4 = mag3.magnet.Cylinder(mag,dim_cyl)
    col1 = mag3.Collection([pm1])
    col2 = mag3.Collection([pm1,pm2])
    col3 = mag3.Collection([pm1,pm2,pm3])
    col4 = mag3.Collection([pm1,pm2,pm3,pm4])
    pos_obs = (1,2,3)
    sens1 = mag3.Sensor(position=pos_obs)
    sens2 = mag3.Sensor(pixel=pos_obs)
    sens3 = mag3.Sensor(position=(1,2,0),pixel=(0,0,3))

    fb1 = mag3.getB(pm1,pos_obs)
    fc1 = mag3.getB(pm3,pos_obs)
    fb2 = np.array([fb1,fb1])
    fc2 = np.array([fc1,fc1])

    for poso,fb,fc in zip([pos_obs,sens1,sens2,sens3,[sens1,sens2]],
        [fb1,fb1,fb1,fb1,fb2],
        [fc1,fc1,fc1,fc1,fc2]
        ):

        src_obs_res = [
            [pm1, poso, fb],
            [pm3, poso, fc],
            [[pm1,pm2], poso, [fb,fb]],
            [[pm1,pm2,pm3], poso, [fb,fb,fc]],
            [col1, poso, fb],
            [col2, poso, 2*fb],
            [col3, poso, 2*fb+fc],
            [col4, poso, 2*fb+2*fc],
            [[pm1,col1], poso, [fb, fb]],
            [[pm1,col1,col2,pm2,col4], poso , [fb,fb,2*fb,fb,2*fb+2*fc]],
            ]

        for sor in src_obs_res:
            sources, observers, result = sor
            result = np.array(result)

            B = mag3.getB(sources, observers)
            assert B.shape == result.shape, "FAILOR shape"
            assert np.allclose(B, result), 'FAILOR values'


def test_getB_level2_input_shape22():
    """test functionality of getB_level2 to combine various
    inputs - position input with shape (2,2)
    """
    mag = (1,2,3)
    dim_cuboid = (1,2,3)
    dim_cyl = (1,2)
    pm1 = mag3.magnet.Cuboid(mag,dim_cuboid)
    pm2 = mag3.magnet.Cuboid(mag,dim_cuboid)
    pm3 = mag3.magnet.Cylinder(mag,dim_cyl)
    pm4 = mag3.magnet.Cylinder(mag,dim_cyl)
    col1 = mag3.Collection([pm1])
    col2 = mag3.Collection([pm1,pm2])
    col3 = mag3.Collection([pm1,pm2,pm3])
    col4 = mag3.Collection([pm1,pm2,pm3,pm4])
    pos_obs = [[(1,2,3),(1,2,3)],[(1,2,3),(1,2,3)]]
    sens1 = mag3.Sensor(pixel=pos_obs)

    fb22 = mag3.getB(pm1,pos_obs)
    fc22 = mag3.getB(pm3,pos_obs)


    for poso,fb,fc in zip([pos_obs,sens1,[sens1,sens1,sens1]],
        [fb22,fb22,[fb22,fb22,fb22]],
        [fc22,fc22,[fc22,fc22,fc22]]
        ):
        fb = np.array(fb)
        fc = np.array(fc)
        src_obs_res = [
            [pm1, poso, fb],
            [pm3, poso, fc],
            [[pm1,pm2], poso, [fb,fb]],
            [[pm1,pm2,pm3], poso, [fb,fb,fc]],
            [col1, poso, fb],
            [col2, poso, 2*fb],
            [col3, poso, 2*fb+fc],
            [col4, poso, 2*fb+2*fc],
            [[pm1,col1], poso, [fb, fb]],
            [[pm1,col1,col2,pm2,col4], poso , [fb,fb,2*fb,fb,2*fb+2*fc]],
            ]

        for sor in src_obs_res:
            sources, observers, result = sor
            result = np.array(result)
            B = mag3.getB(sources, observers)
            assert B.shape == result.shape, "FAILOR2 shape"
            assert np.allclose(B, result), 'FAILOR2 values'


def test_getB_level2_input_path():
    """test functionality of getB_level2 to combine various
    inputs - input objects with path
    """
    mag = (1,2,3)
    dim_cuboid = (1,2,3)
    pm1 = mag3.magnet.Cuboid(mag,dim_cuboid)
    pm2 = mag3.magnet.Cuboid(mag,dim_cuboid)
    sens1 = mag3.Sensor()
    sens2 = mag3.Sensor(pixel=[(0,0,0),(0,0,1),(0,0,2)])

    fb = pm1.getB([(x,0,0) for x in np.linspace(0,-1,11)])

    pm1.move([(.1,0,0)]*10, start='append', increment=True)
    B=mag3.getB(pm1,(0,0,0))
    result = fb
    assert B.shape == result.shape, "FAILOR3a shape"
    assert np.allclose(B, result), 'FAILOR3a values'

    B=mag3.getB(pm1,sens1)
    result = fb
    assert B.shape == result.shape, "FAILOR3b shape"
    assert np.allclose(B, result), 'FAILOR3b values'

    B=mag3.getB([pm1,pm1],sens1)
    result = np.array([fb,fb])
    assert B.shape == result.shape, "FAILOR3c shape"
    assert np.allclose(B, result), 'FAILOR3c values'

    fb = pm2.getB([[(x,0,0),(x,0,0)] for x in np.linspace(0,-1,11)])
    B=mag3.getB([pm1,pm1],[sens1,sens1])
    result = np.array([fb,fb])
    assert B.shape == result.shape, "FAILOR3d shape"
    assert np.allclose(B, result), 'FAILOR3d values'

    fb = pm2.getB([[[(x,0,0),(x,0,1),(x,0,2)]]*2 for x in np.linspace(0,-1,11)])
    B=mag3.getB([pm1,pm1],[sens2,sens2])
    result = np.array([fb,fb])
    assert B.shape == result.shape, "FAILOR3e shape"
    assert np.allclose(B, result), 'FAILOR3e values'


def test_path_tile():
    """ Test if auto-tiled paths of objects will properly be reset
    in getB_level2 before returning
    """
    pm1 = mag3.magnet.Cuboid((11,22,33),(1,2,3))
    pm2 = mag3.magnet.Cuboid((11,22,33),(1,2,3))
    pm2.move([[10/33]*3]*33, increment=True)

    path1p = pm1.position
    path1r = pm1.orientation

    path2p = pm2.position
    path2r = pm2.orientation

    _ = mag3.getB([pm1,pm2],[0,0,0])

    assert np.all(path1p == pm1.position), 'FAILED: getB modified object path'
    assert np.all(path1r.as_quat() == pm1.orientation.as_quat()), 'FAILED: getB modified object path'
    assert np.all(path2p == pm2.position), 'FAILED: getB modified object path'
    assert np.all(path2r.as_quat() == pm2.orientation.as_quat()), 'FAILED: getB modified object path'


def test_sensor_rotation1():
    """ Test simple sensor rotation using sin/cos
    """
    src = mag3.magnet.Cuboid((1000,0,0),(1,1,1))
    sens = mag3.Sensor(position=(1,0,0))
    sens.rotate_from_angax([360/55]*55, 'z', start=1, anchor=None, increment=True)
    B = src.getB(sens)

    B0 = B[0,0]
    Brot = np.array([(B0*np.cos(phi),-B0*np.sin(phi),0) for phi in np.linspace(0,2*np.pi,56)])

    assert np.allclose(B,Brot)


def test_sensor_rotation2():
    """ test sensor roations with different combinations of inputs mag/col + sens/pos
    """
    src = mag3.magnet.Cuboid((1000,0,0),(1,1,1),(0,0,2))
    src2 = mag3.magnet.Cuboid((1000,0,0),(1,1,1),(0,0,2))
    col = mag3.Collection(src,src2)

    poss = (0,0,0)
    sens = mag3.Sensor(pixel=poss)
    sens.rotate_from_angax([0,45,90], 'z')

    sens2 = mag3.Sensor(pixel=poss)
    sens2.rotate_from_angax(-45,'z')

    x1 = np.array([-9.82, 0, 0])
    x2 = np.array([-6.94, 6.94, 0])
    x3 = np.array([0, 9.82, 0])
    x1b = np.array([-19.64, 0, 0])
    x2b = np.array([-13.89, 13.89, 0])
    x3b = np.array([0, 19.64, 0])

    B = mag3.getB(src,poss,squeeze=True)
    Btest = x1
    assert np.allclose(np.around(B,decimals=2), Btest), 'FAIL: mag  +  pos'

    B = mag3.getB([src],[sens],squeeze=True)
    Btest = np.array([x1,x2,x3])
    assert np.allclose(np.around(B,decimals=2), Btest), 'FAIL: mag  +  sens_rot_path'

    B = mag3.getB([src],[sens,poss],squeeze=True)
    Btest = np.array([[x1,x1],[x2,x1],[x3,x1]])
    assert np.allclose(np.around(B,decimals=2), Btest), 'FAIL: mag  +  sens_rot_path, pos'

    B = mag3.getB([src,col],[sens,poss],squeeze=True)
    Btest = np.array([[[x1,x1],[x2,x1],[x3,x1]],[[x1b,x1b],[x2b,x1b],[x3b,x1b]]])
    assert np.allclose(np.around(B,decimals=2), Btest), 'FAIL: mag,col  +  sens_rot_path, pos'


def test_sensor_rotation3():
    """ testing rotated static sensor path
    """
    # case static sensor rot
    src = mag3.magnet.Cuboid((1000,0,0),(1,1,1))
    sens = mag3.Sensor()
    sens.rotate_from_angax(45,'z')
    B0 = mag3.getB(src,sens)
    B0t = np.tile(B0,(12,1))

    sens.move([(0,0,0)]*12)
    Bpath = mag3.getB(src,sens)

    assert np.allclose(B0t,Bpath)


def test_object_tiling():
    """ test if object tiling works when input paths are of various lengths
    """
    src1 = mag3.current.Circular(current=1, diameter=1)
    src1.rotate_from_angax([1]*31, 'x', anchor=(0,1,0), increment=True)

    src2 = mag3.magnet.Cuboid(magnetization=(1,1,1), dimension=(1,1,1), position=(1,1,1))
    src2.move([(1,1,1)]*21)

    src3 = mag3.magnet.Cuboid(magnetization=(1,1,1), dimension=(1,1,1), position=(1,1,1))
    src4 = mag3.magnet.Cuboid(magnetization=(1,1,1), dimension=(1,1,1), position=(1,1,1))

    col = mag3.Collection(src3, src4)
    src3.move([(1,1,1)]*12)
    src4.move([(1,1,1)]*31)

    possis = [[1,2,3]]*5
    sens = mag3.Sensor(pixel=possis)

    assert src1.position.shape == (31, 3), 'a1'
    assert src2.position.shape == (21, 3), 'a2'
    assert src3.position.shape == (12, 3), 'a3'
    assert src4.position.shape == (31, 3), 'a4'
    assert sens.position.shape == (3,), 'a5'

    assert src1.orientation.as_quat().shape == (31, 4), 'b1'
    assert src2.orientation.as_quat().shape == (21, 4), 'b2'
    assert src3.orientation.as_quat().shape == (12, 4), 'b3'
    assert src4.orientation.as_quat().shape == (31, 4), 'b4'
    assert sens.orientation.as_quat().shape == (4,), 'b5'

    B = mag3.getB([src1,src2,col], [sens,possis])
    assert B.shape == (3, 31, 2, 5, 3)

    assert src1.position.shape == (31, 3), 'c1'
    assert src2.position.shape == (21, 3), 'c2'
    assert src3.position.shape == (12, 3), 'c3'
    assert src4.position.shape == (31, 3), 'c4'
    assert sens.position.shape == (3,), 'c5'

    assert src1.orientation.as_quat().shape == (31, 4), 'd1'
    assert src2.orientation.as_quat().shape == (21, 4), 'd2'
    assert src3.orientation.as_quat().shape == (12, 4), 'd3'
    assert src4.orientation.as_quat().shape == (31, 4), 'd4'
    assert sens.orientation.as_quat().shape == (4,), 'd5'
