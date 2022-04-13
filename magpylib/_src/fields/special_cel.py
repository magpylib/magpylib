import numpy as np


def cel0(kc, p, c, s):
    """
    complete elliptic integral algorithm vom Kirby2009
    """
    if kc == 0:
        raise RuntimeError("FAIL")
    errtol = 0.000001
    k = abs(kc)
    pp = p
    cc = c
    ss = s
    em = 1.0
    if p > 0:
        pp = np.sqrt(p)
        ss = s / pp
    else:
        f = kc * kc
        q = 1.0 - f
        g = 1.0 - pp
        f = f - pp
        q = q * (ss - c * pp)
        pp = np.sqrt(f / g)
        cc = (c - ss) / g
        ss = -q / (g * g * pp) + cc * pp
    f = cc
    cc = cc + ss / pp
    g = k / pp
    ss = 2 * (ss + f * g)
    pp = g + pp
    g = em
    em = k + em
    kk = k
    while abs(g - k) > g * errtol:
        k = 2 * np.sqrt(kk)
        kk = k * em
        f = cc
        cc = cc + ss / pp
        g = kk / pp
        ss = 2 * (ss + f * g)
        pp = g + pp
        g = em
        em = k + em
    return (np.pi / 2) * (ss + cc * em) / (em * (em + pp))


def celv(kc, p, c, s):
    """
    vectorized version of the cel integral above
    """

    # if kc == 0:
    #    return NaN
    errtol = 0.000001
    n = len(kc)

    k = np.abs(kc)
    em = np.ones(n, dtype=float)

    cc = c.copy()
    pp = p.copy()
    ss = s.copy()

    # apply a mask for evaluation of respective cases
    mask = p <= 0

    # if p>0:
    pp[~mask] = np.sqrt(p[~mask])
    ss[~mask] = s[~mask] / pp[~mask]

    # else:
    f = kc[mask] * kc[mask]
    q = 1.0 - f
    g = 1.0 - pp[mask]
    f = f - pp[mask]
    q = q * (ss[mask] - c[mask] * pp[mask])
    pp[mask] = np.sqrt(f / g)
    cc[mask] = (c[mask] - ss[mask]) / g
    ss[mask] = -q / (g * g * pp[mask]) + cc[mask] * pp[mask]

    f = cc.copy()
    cc = cc + ss / pp
    g = k / pp
    ss = 2 * (ss + f * g)
    pp = g + pp
    g = em.copy()
    em = k + em
    kk = k.copy()

    # define a mask that adjusts with every evaluation step so that only
    # non-converged entries are further iterated.
    mask = np.ones(n, dtype=bool)
    while np.any(mask):
        k[mask] = 2 * np.sqrt(kk[mask])
        kk[mask] = np.copy(k[mask] * em[mask])
        f[mask] = cc[mask]
        cc[mask] = cc[mask] + ss[mask] / pp[mask]
        g[mask] = kk[mask] / pp[mask]
        ss[mask] = 2 * (ss[mask] + f[mask] * g[mask])
        pp[mask] = g[mask] + pp[mask]
        g[mask] = em[mask]
        em[mask] = k[mask] + em[mask]

        # redefine mask
        mask = np.abs(g - k) > g * errtol

    return (np.pi / 2) * (ss + cc * em) / (em * (em + pp))


def cel(kcv: np.ndarray, pv: np.ndarray, cv: np.ndarray, sv: np.ndarray) -> np.ndarray:
    """
    combine vectorized and non-vectorized implementations for improved performance

    def ellipticK(x):
        return elliptic((1-x)**(1/2.), 1, 1, 1)

    def ellipticE(x):
        return elliptic((1-x)**(1/2.), 1, 1, 1-x)

    def ellipticPi(x, y):
        return elliptic((1-y)**(1/2.), 1-x, 1, 1)
    """
    n_input = len(kcv)

    if n_input < 10:
        return np.array([cel0(kc, p, c, s) for kc, p, c, s in zip(kcv, pv, cv, sv)])

    return celv(kcv, pv, cv, sv)


def cel_loop_stable(k2):
    """
    numerically stabilized version of the function
    xi_loop = (2-k**2)E(k**2)-2(1-k**2)K(k**2) = cel(np.sqrt(1-k2), 1, k2, k2*(k2-1))
    needed for the circular current loop field. This modicfication of the cel()
    algorithm is numerically stable when k2 becomes small.
    """
    # if k2 == 0: return 0    # on axis

    pp = 1 - k2  # allocate pp and use temporarily for 1-k2
    k = np.sqrt(pp)
    kk = 1 + k  # allocate pp and use temporarily for 1+k

    g = 1 if isinstance(k2, (float, int)) else np.ones(len(k2))
    cc = k2**2
    ss = 2 * cc * pp / (kk - k2)
    pp = kk
    em = kk
    kk = k

    errtol = 0.000001
    while np.any(abs(g - k) > g * errtol):
        k = 2 * np.sqrt(kk)
        kk = k * em
        f = cc
        cc = cc + ss / pp
        g = kk / pp
        ss = 2 * (ss + f * g)
        pp = g + pp
        g = em
        em = k + em
    return (np.pi / 2) * (ss + cc * em) / (em * (em + pp))


# def cel_loop_stable_old(k2):
#     """
#     numerically stabilized version of the function
#         xi_loop = (2-k**2)E(k**2)-2(1-k**2)K(k**2)
#         needed for the current.loop fields. See paper
#         Leitner2021
#     """
#     n = len(k2)
#     result = np.empty(n)

#     mask1 = (k2 > 0.04)

#     if np.any(mask1):
#         k2m = k2[mask1]
#         result[mask1] = cel(np.sqrt(1-k2m), np.ones(np.sum(mask1)), k2m, k2m*(k2m-1))
#     if np.any(~mask1):
#         result[~mask1] = cel_loop_taylor0(k2[~mask1])
#     return result


# def cel_loop_taylor0(k2):
#     """
#     taylor expansion of the function xi_loop about k2=0
#         See paper Leitner2021
#     """
#     C2 = 0.5890486225480862
#     C3 = 0.1472621556370216
#     C4 = 0.06902913545485386
#     C5 = 0.04026699568199808
#     C6 = 0.02642521591631124
#     C7 = 0.01868640268367724
#     C8 = 0.01391747699878044
#     C9 = 0.01076947624905629
#     C10 = 0.008581926385966734
#     return (C2*k2**2 + C3*k2**3 + C4*k2**4 + C5*k2**5
#         + C6*k2**6 + C7*k2**7 + C8*k2**8 + C9*k2**9 + C10*k2**10)
