"""Special functions cel."""

import math as m

import numpy as np

_CEL_SCALAR_THRESHOLD = 10
_CEL_ERRORTOL = 1e-8


def _cel_scalar(kc, p, c, s):
    """
    scalar version of bulirsch cel algorithm
    caller ensures that kc != 0
    """
    if p > 0:
        p = m.sqrt(p)
        s /= p
    else:
        f = kc * kc
        q = 1 - f
        g = 1 - p
        f -= p
        q *= s - c * p

        p = m.sqrt(f / g)
        c = (c - s) / g
        s = -q / (g * g * p) + c * p

    em = 1.0
    qc = abs(kc)
    kk = qc

    while True:
        g = kk / p
        c, s = c + s / p, 2 * (s + c * g)
        p += g

        g = em
        em += qc

        if abs(g - qc) <= g * _CEL_ERRORTOL:
            break

        qc = 2 * m.sqrt(kk)
        kk = em * qc

    return (m.pi / 2) * (s + c * em) / (em * (em + p))


def _cel_vector(kc, p, c, s):
    """
    vectorized version of Bulirsch cel algorithm.
    """

    n = len(kc)

    pp = p.copy()
    cc = c.copy()
    ss = s.copy()

    mask = pp <= 0
    pos = ~mask

    pp[pos] = np.sqrt(pp[pos])
    ss[pos] /= pp[pos]

    if np.any(mask):
        f = kc[mask] * kc[mask]
        q = 1 - f
        g = 1 - pp[mask]
        f -= pp[mask]
        q *= ss[mask] - cc[mask] * pp[mask]

        pp[mask] = np.sqrt(f / g)
        cc[mask] = (cc[mask] - ss[mask]) / g
        ss[mask] = -q / (g * g * pp[mask]) + cc[mask] * pp[mask]

    em = np.ones(n)

    kc_zero = kc == 0

    qc = np.abs(kc)
    qc[kc_zero] = 1.0

    kk = qc.copy()

    while True:
        g = kk / pp

        cc, ss = cc + ss / pp, 2 * (ss + cc * g)
        pp += g

        g = em.copy()
        em += qc

        if np.all(np.abs(g - qc) <= g * _CEL_ERRORTOL):
            break

        qc = 2 * np.sqrt(kk)
        kk = em * qc

    result = (np.pi / 2) * (ss + cc * em) / (em * (em + pp))
    result[kc_zero] = np.nan

    return result


def _cel(kc: np.ndarray, p: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Complete elliptic integral
                     π/2
                      ⌠  c cos²𝜗 + s sin²𝜗         d𝜗
    cel(kc, p, c, s) = |  ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯  ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
                      ⌡   cos²𝜗 + p sin²𝜗   √(cos²𝜗 + kc² sin²𝜗)
                      0
    Combines vectorized and non-vectorized implementations for improved performance.

    See also:
    R. Bulirsch, Numerical Calculation of Elliptic Integrals and Elliptic Functions. III
    Numerische Mathematik 13, 305-315 (1969).
    https://dlmf.nist.gov/19.2#E11, and other pages in https://dlmf.nist.gov/19

    The following identities hold:
    K(k²)  = cel(sqrt(1 - k²), 1, 1, 1)
    E(k²)  = cel(sqrt(1 - k²), 1, 1, 1 - k²)
    Π(a²,k²) = cel(sqrt(1 - k²), 1 - a², 1, 1)
    """

    if kc.size < _CEL_SCALAR_THRESHOLD:
        out = np.empty(kc.size, dtype=float)
        for i in range(kc.size):
            out[i] = np.nan if kc[i] == 0 else _cel_scalar(kc[i], p[i], c[i], s[i])
        return out

    return _cel_vector(kc, p, c, s)


# pylint: disable=too-many-positional-arguments


def _cel_iter_scalarvector(qc, p, g, cc, ss, em, kk, xp=m, any_=lambda b: b):
    """
    Iterative part of the Bulirsch cel algorithm.
    This routine continues the iteration from an already prepared state.

    Handles scalar (xp=m) or vectorized (xp=np, any=np.any) inputs.
    Does not modify input arrays in-place.
    """

    while any_(abs(g - qc) > g * _CEL_ERRORTOL):
        qc = 2 * xp.sqrt(kk)
        kk = em * qc

        g = kk / p
        cc, ss = cc + ss / p, 2 * (ss + cc * g)
        p = p + g

        g = em
        em = em + qc

    return (xp.pi / 2) * (ss + cc * em) / (em * (em + p))


def _cel_iter(
    qc: np.ndarray,
    p: np.ndarray,
    g: np.ndarray,
    cc: np.ndarray,
    ss: np.ndarray,
    em: np.ndarray,
    kk: np.ndarray,
) -> np.ndarray:
    """
    Iterative part of Bulirsch cel algorithm where the first iteration was already performed.
    This function improves computation of current loop only and is separate from the _cel
    implementation above.
    """

    if qc.size < _CEL_SCALAR_THRESHOLD:
        return np.array([_cel_iter_scalarvector(*args)
                         for args in zip(qc, p, g, cc, ss, em, kk, strict=False)])  # fmt: skip
    return _cel_iter_scalarvector(qc, p, g, cc, ss, em, kk, xp=np, any_=np.any)


# import math as m

# import numpy as np

# _errtol = 1e-8
# # def _errtol(x) = np.sqrt(np.finfo(x.dtype)))


# def _cel0(kc, p, c, s):
#     """
#     Complete elliptic integral algorithm after
#     R. Bulirsch, Numerical Calculation of Elliptic Integrals and Elliptic Functions. III
#     Numerische Mathematik 13, 305-315 (1969).
#     """
#     if kc == 0:
#         msg = "FAIL cel: kc==0 not allowed."
#         raise RuntimeError(msg)
#     if p > 0:
#         p = np.sqrt(p)
#         s = s / p
#     else:
#         f = kc * kc
#         q = 1 - f
#         g = 1 - p
#         f -= p
#         q *= s - c * p
#         p = np.sqrt(f / g)
#         c = (c - s) / g
#         s = -q / (g * g * p) + c * p
#     mu = 1
#     nu = abs(kc)
#     munu = nu
#     while True:
#         g = munu / p
#         c, s = c + s / p, 2 * (s + c * g)
#         p += g
#         g = mu
#         mu += nu
#         if abs(g - nu) <= g * _errtol:
#             break
#         nu = 2 * np.sqrt(munu)
#         munu = mu * nu
#     return (np.pi / 2) * (s + c * mu) / (mu * (mu + p))


# def _celv(kc, p, c, s):
#     """
#     vectorized version of the cel integral above
#     """

#     # if kc == 0:
#     #    return NaN
#     n = len(kc)

#     pp = p.copy()
#     cc = c.copy()
#     ss = s.copy()

#     # apply a mask for evaluation of respective cases
#     mask = p <= 0

#     # if p>0:
#     pp[~mask] = np.sqrt(pp[~mask])
#     ss[~mask] = ss[~mask] / pp[~mask]
#     # else:
#     f = kc[mask] * kc[mask]
#     q = 1 - f
#     g = 1 - pp[mask]
#     f = f - pp[mask]
#     q *= ss[mask] - cc[mask] * pp[mask]
#     pp[mask] = np.sqrt(f / g)
#     cc[mask] = (cc[mask] - ss[mask]) / g
#     ss[mask] = -q / (g * g * pp[mask]) + cc[mask] * pp[mask]

#     mu = np.ones(n)
#     nu = np.abs(kc)
#     munu = nu.copy()

#     # define a mask that adjusts with every evaluation step so that only
#     # non-converged entries are further iterated.
#     mask = np.ones(n, dtype=bool)
#     g = np.empty(n)
#     while True:
#         g[mask] = munu[mask] / pp[mask]
#         cc[mask], ss[mask] = (
#             cc[mask] + ss[mask] / pp[mask],
#             2 * (ss[mask] + cc[mask] * g[mask]),
#         )
#         pp[mask] += g[mask]
#         g[mask] = mu[mask]
#         mu[mask] += nu[mask]
#         mask[mask] = np.abs(g[mask] - nu[mask]) > g[mask] * _errtol
#         if not np.any(mask[mask]):
#             break
#         nu[mask] = 2 * np.sqrt(munu[mask])
#         munu[mask] = mu[mask] * nu[mask]

#     return (np.pi / 2) * (ss + cc * mu) / (mu * (mu + pp))


# def _cel(kcv: np.ndarray, pv: np.ndarray, cv: np.ndarray, sv: np.ndarray) -> np.ndarray:
#     """
#     combine vectorized and non-vectorized implementations for improved performance

#     def ellipticK(x):
#         return elliptic((1-x)**(1/2.), 1, 1, 1)

#     def ellipticE(x):
#         return elliptic((1-x)**(1/2.), 1, 1, 1-x)

#     def ellipticPi(x, y):
#         return elliptic((1-y)**(1/2.), 1-x, 1, 1)
#     """
#     n_input = len(kcv)

#     if n_input < 10:
#         return np.array(
#             [_cel0(kc, p, c, s) for kc, p, c, s in zip(kcv, pv, cv, sv, strict=False)]
#         )

#     return _celv(kcv, pv, cv, sv)


# def _cel_iter(qc, p, g, cc, ss, em, kk):
#     """
#     Iterative part of Bulirsch cel algorithm
#     """
#     # case1: scalar input
#     #   This cannot happen in core functions

#     # case2: small input vector - loop is faster than vectorized computation
#     n_input = len(qc)
#     if n_input < 15:
#         result = np.zeros(n_input)
#         for i in range(n_input):
#             result[i] = _cel_iter0(qc[i], p[i], g[i], cc[i], ss[i], em[i], kk[i])

#     # case3: vectorized evaluation
#     else:
#         result = _cel_iterv(qc, p, g, cc, ss, em, kk)

#     return result


# def _cel_iter0(qc, p, g, cc, ss, em, kk):
#     """
#     Iterative part of Bulirsch cel algorithm
#     """
#     while m.fabs(g - qc) >= g * _errtol:
#         qc = 2 * m.sqrt(kk)
#         kk = qc * em
#         f = cc
#         cc = cc + ss / p
#         g = kk / p
#         ss = 2 * (ss + f * g)
#         p = p + g
#         g = em
#         em = em + qc
#     return 1.5707963267948966 * (ss + cc * em) / (em * (em + p))


# def _cel_iterv(qc, p, g, cc, ss, em, kk):
#     """
#     Iterative part of Bulirsch cel algorithm
#     """
#     while np.any(np.fabs(g - qc) >= g * _errtol):
#         qc = 2 * np.sqrt(kk)
#         kk = qc * em
#         f = cc
#         cc = cc + ss / p
#         g = kk / p
#         ss = 2 * (ss + f * g)
#         p = p + g
#         g = em
#         em = em + qc
#     return 1.5707963267948966 * (ss + cc * em) / (em * (em + p))
