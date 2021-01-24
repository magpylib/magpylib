import numpy as np
from magpylib._lib.fields.PM_Box_vector import Bfield_BoxV
from magpylib._lib.fields.PM_Box import Bfield_Box
from magpylib3._lib.fields.field_BH_box import field_B_box

# random test ----------------------------------------------
if False:
    n = 1000
    dims = np.random.rand(n,3)
    mags = np.random.rand(n,3)-0.5
    poss = (np.random.rand(n,3)-0.5)*2 

    import time

    t0 = time.perf_counter()
    B_oldv = Bfield_BoxV(mags,poss,dims)
    t1 = time.perf_counter()
    print('time old: ', t1-t0)

    t0 = time.perf_counter()
    B = field_B_box(mags,dims,poss)
    t1 = time.perf_counter()
    print('time new: ', t1-t0)

    print('Berr = ', np.amax(np.linalg.norm(B-B_oldv,axis=1)))
    print('NOTE: larger errors of old formulas happen near numerical instabilities fixed in B_new')


# represent deviation between new and old formula
if True:
    n = 500000
    dims = np.array([[2,2,2]]*n)
    mags = np.array([[1000,1000,1000]]*n)
    poss = (np.random.rand(n,3)-0.5)*2*10 
    poss[:,2] = -1

    X = poss[:,0]
    Y = poss[:,1]

    B_oldv = Bfield_BoxV(mags,poss,dims)
    B_new = field_B_box(mags,dims,poss)
    dB = np.linalg.norm((B_oldv-B_new),axis=1)
    dB_rel = dB/np.linalg.norm(B_new,axis=1)
    print(np.amax(dB))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(constrained_layout=True)
    sc = ax.scatter(X,Y,marker='.',c=np.log10(dB),s=3,cmap='hot')
    fig.colorbar(sc)
    plt.show()

# # special cases testing ---------------------
if False:
    print('---')
    pp = np.array([[1,1,.5],[1,1,1],[-1,-1,1],[-1,.5,-1],[1,1,1.000001]])
    dd = np.array([[2,2,2]]*len(pp))
    mm = np.array([[1,1,1]]*len(pp))

    print('new getB:')
    print(field_B_box(mm,dd,pp))

    Bold = []
    for p,d,m in zip(pp,dd,mm):
        Bold += [Bfield_Box(m,p,d)]
    Bold = np.array(Bold)
    print('old getB:')
    print(Bold)