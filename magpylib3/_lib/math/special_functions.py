'''
special functions computations beyond scipy.special
'''

import numpy as np
def celv(kc,p,c,s):
    '''
    vectorized version of the cel integral
    original implementation from [Kirby]
    '''

    #if kc == 0:
    #    return NaN
    errtol = .000001
    N = len(kc)
    
    k = np.abs(kc)
    em = np.ones(N,dtype=float)

    cc = c.copy()
    pp = p.copy()
    ss = s.copy()
    
    # apply a mask for evaluation of respective cases
    mask = p>0
    maskInv = np.invert(mask)

    #if p>0:
    pp[mask] = np.sqrt(p[mask])
    ss[mask] = s[mask]/pp[mask]

    #else:
    f = kc[maskInv]*kc[maskInv]
    q = 1.-f
    g = 1. - pp[maskInv]
    f = f - pp[maskInv]
    q = q*(ss[maskInv] - c[maskInv]*pp[maskInv])
    pp[maskInv] = np.sqrt(f/g)
    cc[maskInv] = (c[maskInv]-ss[maskInv])/g
    ss[maskInv] = -q/(g*g*pp[maskInv]) + cc[maskInv]*pp[maskInv]

    f = cc.copy()
    cc = cc + ss/pp
    g = k/pp
    ss = 2*(ss + f*g)
    pp = g + pp
    g = em.copy()
    em = k + em
    kk = k.copy()

    #define a mask that adjusts with every evauation
    #   step so that only non-converged entries are
    #   further iterated.   
    mask = np.ones(N,dtype=bool)
    while np.any(mask):
        k[mask] = 2*np.sqrt(kk[mask])
        kk[mask] = np.copy(k[mask]*em[mask])
        f[mask] = cc[mask]
        cc[mask] = cc[mask] + ss[mask]/pp[mask]
        g[mask] = kk[mask]/pp[mask]
        ss[mask] = 2*(ss[mask] + f[mask]*g[mask])
        pp[mask] = g[mask] + pp[mask]
        g[mask] = em[mask]
        em[mask] = k[mask]+em[mask]

        #redefine mask so only non-convergent 
        #   entries are reiterated
        mask = (np.abs(g-k) > g*errtol)

    return(np.pi/2)*(ss+cc*em)/(em*(em+pp))


def ellipticKV(x):
    '''
    special case complete cel integral of first kind ellipticK
    0 <= x <1
    '''
    N = len(x)
    onez = np.ones([N])
    return celv((1-x)**(1/2.), onez, onez, onez)



def ellipticEV(x):
    '''
    special case complete elliptic integral of second kind ellipticE
    E(x) = int_0^pi/2 (1-x sin(phi)^2)^(1/2) dphi
    requires x < 1 ! 
    '''
    N = len(x)
    onez = np.ones([N])
    return celv((1-x)**(1/2.), onez, onez, 1-x)



def ellipticPiV(x, y):
    '''
    special case complete elliptic integral of third kind ellipticPi
    E(x) = int_0^pi/2 (1-x sin(phi)^2)^(1/2) dphi
    requires x < 1 ! 
    '''
    N = len(x)
    onez = np.ones([N])
    return celv((1-y)**(1/2.), 1-x, onez, onez)





if __name__ == '__main__':
    import time
    import numpy as np

    N = 1
    kp = np.random.rand(N)
    p = np.random.rand(N)
    c = np.random.rand(N)
    s = np.random.rand(N)
    t0 = time.perf_counter()
    ell = celv(kp,p,c,s)
    t1 = time.perf_counter()
    print(t1-t0)


    x = np.random.rand(N)
    t0 = time.perf_counter()
    ellipticEV(x)
    t1 = time.perf_counter()
    print(t1-t0)