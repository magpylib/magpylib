"""Special functions cel."""

# pylint: disable=too-many-positional-arguments

import math as m

import numpy as np


def _cel0(kc, p, c, s):
    """
    Complete elliptic integral algorithm after
    R. Bulirsch, Numerical Calculation of Elliptic Integrals and Elliptic Functions. III
    Numerische Mathematik 13, 305-315 (1969).
    """
    if kc == 0:
        msg = "FAIL cel: kc==0 not allowed."
        raise RuntimeError(msg)
    errtol = 0.000001
    if p > 0:
        p = np.sqrt(p)
        s = s / p
    else:
        f = kc * kc
        q = 1 - f
        g = 1 - p
        f -= p
        q *= s - c * p
        p = np.sqrt(f / g)
        c = (c - s) / g
        s = -q / (g * g * p) + c * p
    mu = 1
    nu = abs(kc)
    munu = nu
    while True:
        g = munu / p
        c, s = c + s / p, 2 * (s + c * g)
        p += g
        g = mu
        mu += nu
        if abs(g - nu) <= g * errtol:
            break
        nu = 2 * np.sqrt(munu)
        munu = mu * nu
    return (np.pi / 2) * (s + c * mu) / (mu * (mu + p))


def _celv(kc, p, c, s):
    """
    vectorized version of the cel integral above
    """

    # if kc == 0:
    #    return NaN
    errtol = 0.000001
    n = len(kc)

    pp = p.copy()
    cc = c.copy()
    ss = s.copy()

    # apply a mask for evaluation of respective cases
    mask = p <= 0

    # if p>0:
    pp[~mask] = np.sqrt(pp[~mask])
    ss[~mask] = ss[~mask] / pp[~mask]
    # else:
    f = kc[mask] * kc[mask]
    q = 1 - f
    g = 1 - pp[mask]
    f = f - pp[mask]
    q *= ss[mask] - cc[mask] * pp[mask]
    pp[mask] = np.sqrt(f / g)
    cc[mask] = (cc[mask] - ss[mask]) / g
    ss[mask] = -q / (g * g * pp[mask]) + cc[mask] * pp[mask]

    mu = np.ones(n)
    nu = np.abs(kc)
    munu = nu.copy()

    # define a mask that adjusts with every evaluation step so that only
    # non-converged entries are further iterated.
    mask = np.ones(n, dtype=bool)
    g = np.empty(n)
    while True:
        g[mask] = munu[mask] / pp[mask]
        cc[mask], ss[mask] = (
            cc[mask] + ss[mask] / pp[mask],
            2 * (ss[mask] + cc[mask] * g[mask]),
        )
        pp[mask] += g[mask]
        g[mask] = mu[mask]
        mu[mask] += nu[mask]
        mask[mask] = np.abs(g[mask] - nu[mask]) > g[mask] * errtol
        if not np.any(mask[mask]):
            break
        nu[mask] = 2 * np.sqrt(munu[mask])
        munu[mask] = mu[mask] * nu[mask]

    return (np.pi / 2) * (ss + cc * mu) / (mu * (mu + pp))


def _cel(kcv: np.ndarray, pv: np.ndarray, cv: np.ndarray, sv: np.ndarray) -> np.ndarray:
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
        return np.array(
            [_cel0(kc, p, c, s) for kc, p, c, s in zip(kcv, pv, cv, sv, strict=False)]
        )

    return _celv(kcv, pv, cv, sv)


def _cel_iter(qc, p, g, cc, ss, em, kk):
    """
    Iterative part of Bulirsch cel algorithm
    """
    # case1: scalar input
    #   This cannot happen in core functions

    # case2: small input vector - loop is faster than vectorized computation
    n_input = len(qc)
    if n_input < 15:
        result = np.zeros(n_input)
        for i in range(n_input):
            result[i] = _cel_iter0(qc[i], p[i], g[i], cc[i], ss[i], em[i], kk[i])

    # case3: vectorized evaluation
    else:
        result = _cel_iterv(qc, p, g, cc, ss, em, kk)

    return result


def _cel_iter0(qc, p, g, cc, ss, em, kk):
    """
    Iterative part of Bulirsch cel algorithm
    """
    while m.fabs(g - qc) >= qc * 1e-8:
        qc = 2 * m.sqrt(kk)
        kk = qc * em
        f = cc
        cc = cc + ss / p
        g = kk / p
        ss = 2 * (ss + f * g)
        p = p + g
        g = em
        em = em + qc
    return 1.5707963267948966 * (ss + cc * em) / (em * (em + p))


def _cel_iterv(qc, p, g, cc, ss, em, kk):
    """
    Iterative part of Bulirsch cel algorithm
    """
    while np.any(np.fabs(g - qc) >= qc * 1e-8):
        qc = 2 * np.sqrt(kk)
        kk = qc * em
        f = cc
        cc = cc + ss / p
        g = kk / p
        ss = 2 * (ss + f * g)
        p = p + g
        g = em
        em = em + qc
    return 1.5707963267948966 * (ss + cc * em) / (em * (em + p))
