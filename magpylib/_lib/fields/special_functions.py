'''
special functions computations beyond scipy.special
'''

import numpy as np

def celv(kc, p, c, s):
    """
    vectorized version of the cel integral
    original implementation from [Kirby]
    """

    #if kc == 0:
    #    return NaN
    errtol = .000001
    n = len(kc)

    k = np.abs(kc)
    em = np.ones(n, dtype=float)

    cc = c.copy()
    pp = p.copy()
    ss = s.copy()

    # apply a mask for evaluation of respective cases
    mask = p<=0

    #if p>0:
    pp[~mask] = np.sqrt(p[~mask])
    ss[~mask] = s[~mask]/pp[~mask]

    #else:
    f = kc[mask]*kc[mask]
    q = 1.-f
    g = 1. - pp[mask]
    f = f - pp[mask]
    q = q*(ss[mask] - c[mask]*pp[mask])
    pp[mask] = np.sqrt(f/g)
    cc[mask] = (c[mask]-ss[mask])/g
    ss[mask] = -q/(g*g*pp[mask]) + cc[mask]*pp[mask]

    f = cc.copy()
    cc = cc + ss/pp
    g = k/pp
    ss = 2*(ss + f*g)
    pp = g + pp
    g = em.copy()
    em = k + em
    kk = k.copy()

    # define a mask that adjusts with every evaluation step so that only
    # non-converged entries are further iterated.
    mask = np.ones(n, dtype=bool)
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

        # redefine mask
        mask = (np.abs(g-k) > g*errtol)

    return(np.pi/2)*(ss+cc*em)/(em*(em+pp))


# unused quantities -----------------------------------------

# def ellipticKV(x):
#     '''
#     special case complete cel integral of first kind ellipticK
#     0 <= x <1
#     '''
#     N = len(x)
#     onez = np.ones([N])
#     return celv((1-x)**(1/2.), onez, onez, onez)


# def ellipticEV(x):
#     '''
#     special case complete elliptic integral of second kind ellipticE
#     E(x) = int_0^pi/2 (1-x sin(phi)^2)^(1/2) dphi
#     requires x < 1 !
#     '''
#     N = len(x)
#     onez = np.ones([N])
#     return celv((1-x)**(1/2.), onez, onez, 1-x)


# def ellipticPiV(x, y):
#     '''
#     special case complete elliptic integral of third kind ellipticPi
#     E(x) = int_0^pi/2 (1-x sin(phi)^2)^(1/2) dphi
#     requires x < 1 !
#     '''
#     N = len(x)
#     onez = np.ones([N])
#     return celv((1-y)**(1/2.), 1-x, onez, onez)
