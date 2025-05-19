# pylint: disable=too-many-positional-arguments
from __future__ import annotations

import math as m

import array_api_extra as xpx
import numpy as np
from array_api_compat import array_namespace


def cel0(kc, p, c, s):
    """
    complete elliptic integral algorithm vom Kirby2009
    """
    if kc == 0:
        msg = "FAIL"
        raise RuntimeError(msg)
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
    xp = array_namespace(kc, p, c, s)

    # if kc == 0:
    #    return NaN
    # errtol = 0.000001
    errtol = 0.000001
    n = kc.shape[0]

    k = xp.abs(kc)
    em = xp.ones(n, dtype=xp.float64)

    # cc = xp.asarray(c, copy=True)
    # pp = xp.asarray(p, copy=True)
    # ss = xp.asarray(s, copy=True)

    # apply a mask for evaluation of respective cases
    mask = p <= 0

    f = kc * kc
    q = 1.0 - f
    g = 1.0 - p
    f = f - p
    q = q * (s - c * p)
    pp = xpx.apply_where(
        mask,
        (p, f, g),
        lambda p, f, g: xp.sqrt(f / g),
        lambda p, f, g: xp.sqrt(p),
    )
    cc = xpx.apply_where(
        mask, (c, s, g), lambda c, s, g: (c - s) / g, lambda c, ss, g: c
    )
    ss = xpx.apply_where(
        mask,
        (s, pp, q, g, cc),
        lambda s, pp, q, g, cc: -q / (g * g * pp) + cc * pp,
        lambda s, pp, q, g, cc: s / pp,
    )

    f = xp.asarray(cc, copy=True)
    cc = cc + ss / pp
    g = k / pp
    ss = 2 * (ss + f * g)
    pp = g + pp
    g = xp.asarray(em, copy=True)
    em = k + em
    kk = xp.asarray(k, copy=True)

    # define a mask that adjusts with every evaluation step so that only
    # non-converged entries are further iterated.
    mask = xp.ones(n, dtype=xp.bool)
    while xp.any(mask):
        k = 2 * xp.sqrt(kk)
        kk = k * em
        f = cc
        cc = cc + ss / pp
        g = kk / pp
        ss = 2 * (ss + f * g)
        pp = g + pp
        g = em
        em = k + em

        # redefine mask
        err = g - k
        tol = g * errtol
        mask = xp.abs(err) > tol

    return (xp.pi / 2) * (ss + cc * em) / (em * (em + pp))


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
    # n_input = len(kcv)

    # if n_input < 10:
    # return np.array(
    # [cel0(kc, p, c, s) for kc, p, c, s in zip(kcv, pv, cv, sv, strict=False)]
    # )

    return celv(kcv, pv, cv, sv)


def cel_iter(qc, p, g, cc, ss, em, kk):
    """
    Iterative part of Bulirsch cel algorithm
    """
    # case1: scalar input
    #   This cannot happen in core functions

    # case2: small input vector - loop is faster than vectorized computation
    # n_input = len(qc)
    # if n_input < 15:
    #    result = np.zeros(n_input)
    #    for i in range(n_input):
    #        result[i] = cel_iter0(qc[i], p[i], g[i], cc[i], ss[i], em[i], kk[i])

    # case3: vectorized evaluation
    return cel_iterv(qc, p, g, cc, ss, em, kk)


def cel_iter0(qc, p, g, cc, ss, em, kk):
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


def cel_iterv(qc, p, g, cc, ss, em, kk):
    """
    Iterative part of Bulirsch cel algorithm
    """
    xp = array_namespace(qc, p, g, cc, ss, em, kk)
    while xp.any(xp.abs(g - qc) >= qc * 1e-8):
        qc = 2 * xp.sqrt(kk)
        kk = qc * em
        f = cc
        cc = cc + ss / p
        g = kk / p
        ss = 2 * (ss + f * g)
        p = p + g
        g = em
        em = em + qc
    return 1.5707963267948966 * (ss + cc * em) / (em * (em + p))
