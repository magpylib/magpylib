import pickle
import numpy as np
from magpylib3._lib.fields.field_BH_box import field_BH_box
import magpylib3 as mag3

# # GENERATE TEST DATA
# n = 500
# mag3.Config.EDGESIZE = 1e-14

# # dim general
# dim_gen = np.random.rand(n,3)
# # dim edge
# dim_edge = np.array([(2,2,2)]*n)

# # pure edge positions
# pos_edge = (np.random.rand(n,3)-0.5)*2
# pos_edge[:,0]=1 + (np.random.rand(n)-0.5)*1e-14
# pos_edge[:,1]=1  + (np.random.rand(n)-0.5)*1e-14
# for i in range(n):
#     np.random.shuffle(pos_edge[i])
# # general positions
# pos_gen = (np.random.rand(n,3)-0.5)*.5 # half inner half outer
# # mixed positions
# aa = np.r_[pos_edge, pos_gen]
# np.random.shuffle(aa)
# pos_mix = aa[:n]

# # general mag (no special cases at this point)
# mag = (np.random.rand(n,3)-0.5)*1000

# poss = [pos_edge, pos_gen, pos_mix]
# dims = [dim_edge, dim_gen, dim_edge]

# Bs = []
# for dim,pos in zip(dims,poss):
#     Bs += [field_BH_box(True, mag, dim, pos)]
# Bs = np.array(Bs)

# print(Bs)
# pickle.dump([mag, dims, poss, Bs], open('testdata_field_BH_box.p','wb'))

def test_field_BH_box():
    """ test box field
    """
    mag3.Config.EDGESIZE=1e-14
    mag, dims, poss, B = pickle.load(open('tests/testdata/testdata_field_BH_box.p','rb'))
    Btest = []
    for dim,pos in zip(dims,poss):
        Btest += [field_BH_box(True, mag, dim, pos)]
    Btest = np.array(Btest)
    assert np.allclose(B, Btest), 'Box field computation broken'


def test_field_BH_box_mag0():
    """ test box field mag=0
    """
    n = 10
    mag = np.zeros((n,3))
    dim = np.random.rand(n,3)
    pos = np.random.rand(n,3)
    B = field_BH_box(True, mag, dim, pos)
    assert np.allclose(mag,B), 'Box mag=0 case broken'
