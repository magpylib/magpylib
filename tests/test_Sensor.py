import numpy as np
import magpylib as mag3

def test_sensor():
    """ self-consistent test of the sensor class
    """
    pm = mag3.magnet.Box((11,22,33),(1,2,3))
    possis = [(3*np.cos(t/180*np.pi),3*np.sin(t/180*np.pi),1) for t in np.linspace(0,555,44)]
    sens = mag3.Sensor()
    sens.move_to((3,0,1))
    sens.rotate_from_angax(555,'z',anchor=0,steps=43)
    sens.rotate_from_angax(-555,'z',steps=-43)

    B1 = pm.getB(possis)
    B2 = sens.getB(pm)

    assert B1.shape==B2.shape, 'FAIL sensor shape'
    assert np.allclose(B1,B2), 'FAIL sensor values'

    poss1 = [(t,0,2) for t in np.linspace(0,5,33)]
    poss2 = [(t,0,3) for t in np.linspace(0,5,33)]
    poss3 = [(t,0,4) for t in np.linspace(0,5,33)]
    B1 = np.array([pm.getB(poss) for poss in [poss1,poss2,poss3]])
    B1 = np.swapaxes(B1,0,1)

    sens = mag3.Sensor(pos_pix=[(0,0,2),(0,0,3),(0,0,4)])
    sens.move_by((5,0,0),steps=32)
    B2 = sens.getB(pm)

    assert B1.shape==B2.shape, 'FAIL sensor shape'
    assert np.allclose(B1,B2), 'FAIL sensor values'
