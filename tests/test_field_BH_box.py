import numpy as np
from magpylib3._lib.fields.field_BH_box import field_B_box

def test_field_B_box():
    
    # special case test ---------------------------------------------
    # all solutions at corners and edges should be zero
    # corner positions
    pos = [(1,1,1), (1,1,-1), (1,-1,1), (-1,1,1),
            (1,-1,-1), (-1,1,-1), (-1,-1,1), (-1,-1,-1)]
    # random edge positions
    pos += [(1,1,.2), (1,-1,.3), (-1,1,.4), (-1,-1,-.3)]
    pos += [(1,.5,1), (1,.6,-1), (-1,.7,1), (-1,.8,-1)]
    pos += [(-.2,1,1), (-.3,1,-1), (-.4,-1,1), (-.5,-1,-1)]
    pos = np.array(pos)

    dim = np.array([[2,2,2]]*len(pos))
    mag = np.array([[1,1,1]]*len(pos))

    B = field_B_box(mag,dim,pos)
    
    assert np.all(B==0), 'field_B_Box not 0 at edges/corners'


    # general case test ---------------------------------------------
    # load random positions and solutions and compare
    mag, dim, pos, B = np.load('tests/testdata/testdata_field_B_box.npy')

    Btest = field_B_box(mag, dim, pos)

    assert np.allclose(Btest, B), 'field_B_Box general case fails'

