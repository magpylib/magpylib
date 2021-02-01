import pickle
import numpy as np
from magpylib3.magnet import Box

def test_Box():

    data = pickle.load(open('tests/testdata/testdata_Box.p', 'rb'))
    mags,dims,posos,angs,axs,anchs,movs,B = data

    Btest = []
    for mag,dim,ang,ax,anch,mov,poso in zip(mags,dims,angs,axs,anchs,movs,posos):
        pm = Box(mag,dim)

        # 18 subsequent operations
        for a,aa,aaa,mv in zip(ang,ax,anch,mov):
            pm.move(mv).rotate_angle_axis(a,aa,aaa)
        
        Btest += [pm.getB(poso)]
    Btest = np.array(Btest)

    assert np.allclose(B, Btest), "test_Box failed big time"
    

