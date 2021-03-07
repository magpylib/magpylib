import numpy as np
import magpylib as mag3

def test_getB_level2_input_simple():
    """ test functionality of getB_level2 to combine various
    inputs - simple position inputs
    """
    mag = (1,2,3)
    dim_box = (1,2,3)
    dim_cyl = (1,2)
    pm1 = mag3.magnet.Box(mag,dim_box)
    pm2 = mag3.magnet.Box(mag,dim_box)
    pm3 = mag3.magnet.Cylinder(mag,dim_cyl)
    pm4 = mag3.magnet.Cylinder(mag,dim_cyl)
    col1 = mag3.Collection([pm1])
    col2 = mag3.Collection([pm1,pm2])
    col3 = mag3.Collection([pm1,pm2,pm3])
    col4 = mag3.Collection([pm1,pm2,pm3,pm4])
    pos_obs = (1,2,3)
    sens1 = mag3.Sensor(pos=pos_obs)
    sens2 = mag3.Sensor(pos_pix=pos_obs)
    sens3 = mag3.Sensor(pos=(1,2,0),pos_pix=(0,0,3))

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
    dim_box = (1,2,3)
    dim_cyl = (1,2)
    pm1 = mag3.magnet.Box(mag,dim_box)
    pm2 = mag3.magnet.Box(mag,dim_box)
    pm3 = mag3.magnet.Cylinder(mag,dim_cyl)
    pm4 = mag3.magnet.Cylinder(mag,dim_cyl)
    col1 = mag3.Collection([pm1])
    col2 = mag3.Collection([pm1,pm2])
    col3 = mag3.Collection([pm1,pm2,pm3])
    col4 = mag3.Collection([pm1,pm2,pm3,pm4])
    pos_obs = [[(1,2,3),(1,2,3)],[(1,2,3),(1,2,3)]]
    sens1 = mag3.Sensor(pos_pix=pos_obs)

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
    dim_box = (1,2,3)
    pm1 = mag3.magnet.Box(mag,dim_box)
    pm2 = mag3.magnet.Box(mag,dim_box)
    sens1 = mag3.Sensor()
    sens2 = mag3.Sensor(pos_pix=[(0,0,0),(0,0,1),(0,0,2)])

    fb = pm1.getB([(x,0,0) for x in np.linspace(0,-1,11)])

    pm1.move_by((1,0,0),steps=10)
    B=mag3.getB(pm1,(0,0,0),)
    result = fb
    assert B.shape == result.shape, "FAILOR3 shape"
    assert np.allclose(B, result), 'FAILOR3 values'

    B=mag3.getB(pm1,sens1)
    result = fb
    assert B.shape == result.shape, "FAILOR3 shape"
    assert np.allclose(B, result), 'FAILOR3 values'

    B=mag3.getB([pm1,pm1],sens1)
    result = np.array([fb,fb])
    assert B.shape == result.shape, "FAILOR3 shape"
    assert np.allclose(B, result), 'FAILOR3 values'

    fb = pm2.getB([[(x,0,0),(x,0,0)] for x in np.linspace(0,-1,11)])
    B=mag3.getB([pm1,pm1],[sens1,sens1])
    result = np.array([fb,fb])
    assert B.shape == result.shape, "FAILOR3 shape"
    assert np.allclose(B, result), 'FAILOR3 values'

    fb = pm2.getB([[[(x,0,0),(x,0,1),(x,0,2)]]*2 for x in np.linspace(0,-1,11)])
    B=mag3.getB([pm1,pm1],[sens2,sens2])
    result = np.array([fb,fb])
    assert B.shape == result.shape, "FAILOR3 shape"
    assert np.allclose(B, result), 'FAILOR3 values'

    fb = pm2.getB([[(x,0,0),(x,0,1),(x,0,2),(x,0,0)] for x in np.linspace(0,-1,11)])
    B=mag3.getB([pm1,pm1],[sens2,sens1])
    result = np.array([fb,fb])
    assert B.shape == result.shape, "FAILOR3 shape sens_merge"
    assert np.allclose(B, result), 'FAILOR3 values sens_merge'

