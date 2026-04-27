"""Special functions cel."""

import numpy as np

_errtol = 1e-8  # ~ np.sqrt(np.finfo(np.float64).eps)


def _cel0(kc, p, c, s):
    """
    Complete elliptic integral algorithm after
    R. Bulirsch, Numerical Calculation of Elliptic Integrals and Elliptic Functions. III
    Numerische Mathematik 13, 305-315 (1969).

    See also https://dlmf.nist.gov/19.2#E11, and other pages in https://dlmf.nist.gov/19
    """
    if kc == 0:
        msg = "FAIL cel: kc==0 not allowed."
        raise RuntimeError(msg)
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
        if abs(g - nu) <= g * _errtol:
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
    mask = kc != 0  # if kc == 0: skip iteration
    g = np.empty(n)
    while True:
        g[mask] = munu[mask] / pp[mask]
        cc[mask], ss[mask] = (
            cc[mask] + ss[mask] / pp[mask],
            2 * (ss[mask] + cc[mask] * g[mask]),
        )
        pp[mask] = pp[mask] + g[mask]
        g[mask] = mu[mask]
        mu[mask] += nu[mask]
        mask[mask] = np.abs(g[mask] - nu[mask]) > g[mask] * _errtol
        if not np.any(mask[mask]):
            break
        nu[mask] = 2 * np.sqrt(munu[mask])
        munu[mask] = mu[mask] * nu[mask]

    result = (np.pi / 2) * (ss + cc * mu) / (mu * (mu + pp))
    # deal with the kc == 0 special case
    result[kc == 0] = np.nan
    return result


def _cel(kcv: np.ndarray, pv: np.ndarray, cv: np.ndarray, sv: np.ndarray) -> np.ndarray:
    """
    Complete elliptic integral
                     π/2
                      ⌠  a cos²𝜗 + b sin²𝜗         d𝜗
    cel(q, p, c, s) = |  ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯  ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
                      ⌡   cos²𝜗 + p sin²𝜗   √(cos²𝜗 + q sin²𝜗)
                      0
    Combines vectorized and non-vectorized implementations for improved performance.

    See also:
    R. Bulirsch, Numerical Calculation of Elliptic Integrals and Elliptic Functions. III
    Numerische Mathematik 13, 305-315 (1969).
    https://dlmf.nist.gov/19.2#E11, and other pages in https://dlmf.nist.gov/19

    def ellipticK(k2):
        return _cel(np.sqrt(1-k2), 1, 1, 1)

    def ellipticE(k2):
        return _cel(np.sqrt(1-k2), 1, 1, 1-k2)

    def ellipticPi(alpha2, k2):
        return _cel(np.sqrt(1-k2), 1-alpha2, 1, 1)
    """
    n_input = len(kcv)

    if n_input < 10:
        return np.array(
            [_cel0(kc, p, c, s) for kc, p, c, s in zip(kcv, pv, cv, sv, strict=False)]
        )

    return _celv(kcv, pv, cv, sv)
