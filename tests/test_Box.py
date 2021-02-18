import pickle
import numpy as np
from magpylib3.magnet import Box

def test_Box():
    # data generated below
    data = pickle.load(open('tests/testdata/testdata_Box.p', 'rb'))
    mags,dims,posos,angs,axs,anchs,movs,B = data

    Btest = []
    for mag,dim,ang,ax,anch,mov,poso in zip(mags,dims,angs,axs,anchs,movs,posos):
        pm = Box(mag,dim)

        # 18 subsequent operations
        for a,aa,aaa,mv in zip(ang,ax,anch,mov):
            pm.move_by(mv).rotate_from_angax(a,aa,aaa)
        
        Btest += [pm.getB(poso)]
    Btest = np.array(Btest)

    assert np.allclose(B, Btest), "test_Box failed big time"
    

if __name__ == "__main__":
    if False:
        """data generation for test_Box()"""

        N = 100
        
        mags = (np.random.rand(N,3)-0.5)*1000
        dims = (np.random.rand(N,3)-0.5)*5
        posos = (np.random.rand(N,333,3)-0.5)*10 #readout at 333 positions

        angs =  (np.random.rand(N,18)-0.5)*2*10 # each step rote by max 10 deg
        axs =   (np.random.rand(N,18,3)-0.5)
        anchs = (np.random.rand(N,18,3)-0.5)*5.5
        movs =  (np.random.rand(N,18,3)-0.5)*0.5

        B = []
        for mag,dim,ang,ax,anch,mov,poso in zip(mags,dims,angs,axs,anchs,movs,posos):
            pm = Box(mag,dim)

            # 18 subsequent operations
            for a,aa,aaa,mv in zip(ang,ax,anch,mov):
                pm.move_by(mv).rotate_from_angax(a,aa,aaa)
            
            B += [pm.getB(poso)]
        B = np.array(B)

        inp = [mags,dims,posos,angs,axs,anchs,movs,B]

        pickle.dump(inp,open('testdata_Box.p', 'wb'))