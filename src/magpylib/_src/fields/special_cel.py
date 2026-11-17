"""Special functions cel."""

import math as m

import numpy as np
from array_api_compat import array_namespace, is_numpy_namespace

from magpylib._src.array_api_utils import xp_promote

_CEL_SCALAR_THRESHOLD = 10
_CEL_ERRORTOL = 1e-8


def _cel_scalar(kc, p, c, s):
    """
    Scalar version of Bulirsch cel algorithm
    """
    if kc == 0:
        return np.nan

    if p > 0:
        p = m.sqrt(p)
        s = s / p
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
    Vectorized version of Bulirsch cel algorithm.
    """
    xp = array_namespace(kc, p, c, s)
    kc, p, c, s = xp_promote(kc, p, c, s, xp=xp)

    mask = p <= 0
    kc_zero = kc == 0
    kc = xp.where(kc_zero, 1.0, kc)  # NaN is returned for these in the end

    # case p > 0, computed with p=1 where it does not apply
    p_pos = xp.sqrt(xp.where(mask, 1.0, p))
    s_pos = s / p_pos

    # case p <= 0, computed with p=-1 where it does not apply
    p_neg = xp.where(mask, p, -1.0)
    f = kc * kc
    q = 1 - f
    g = 1 - p_neg
    f = f - p_neg
    q = q * (s - c * p_neg)

    p_neg = xp.sqrt(f / g)
    c_neg = (c - s) / g
    s_neg = -q / (g * g * p_neg) + c_neg * p_neg

    p = xp.where(mask, p_neg, p_pos)
    c = xp.where(mask, c_neg, c)
    s = xp.where(mask, s_neg, s_pos)

    em = xp.ones_like(kc)
    qc = xp.abs(kc)
    kk = qc

    while True:
        g = kk / p
        c, s = c + s / p, 2 * (s + c * g)
        p = p + g

        g = em
        em = em + qc

        if bool(xp.all(xp.abs(g - qc) <= g * _CEL_ERRORTOL)):
            break

        qc = 2 * xp.sqrt(kk)
        kk = em * qc

    result = (xp.pi / 2) * (s + c * em) / (em * (em + p))
    return xp.where(kc_zero, xp.nan, result)


def _cel(kc: np.ndarray, p: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Complete elliptic integral
                      π/2
                       ⌠  c cos²𝜗 + s sin²𝜗          d𝜗
    cel(kc, p, c, s) = |  ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯  ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
                       ⌡   cos²𝜗 + p sin²𝜗   √(cos²𝜗 + kc² sin²𝜗)
                       0
    Combines vectorized and non-vectorized implementations for improved performance:
    small NumPy arrays are computed element by element, all other input vectorized.

    See also:
    R. Bulirsch, Numerical Calculation of Elliptic Integrals and Elliptic Functions. III
    Numerische Mathematik 13, 305-315 (1969).
    https://dlmf.nist.gov/19.2#E11, and other pages in https://dlmf.nist.gov/19

    The following identities hold:
    K(k²)  = cel(sqrt(1 - k²), 1, 1, 1)
    E(k²)  = cel(sqrt(1 - k²), 1, 1, 1 - k²)
    Π(a²,k²) = cel(sqrt(1 - k²), 1 - a², 1, 1)
    """

    xp = array_namespace(kc, p, c, s)
    if is_numpy_namespace(xp) and kc.size < _CEL_SCALAR_THRESHOLD:
        return np.array([_cel_scalar(*args)
                         for args in zip(kc, p, c, s, strict=False)])  # fmt: skip
    return _cel_vector(kc, p, c, s)


# pylint: disable=too-many-positional-arguments


def _cel_iter_scalarvector(qc, p, g, cc, ss, em, kk, xp=m, any_=lambda b: b):
    """
    Iterative part of the Bulirsch cel algorithm.
    This routine continues the iteration from an already prepared state.

    Handles scalar (xp=m) or vectorized (xp an array namespace, any_=xp.any) inputs.
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

    xp = array_namespace(qc, p, g, cc, ss, em, kk)
    if is_numpy_namespace(xp) and qc.size < _CEL_SCALAR_THRESHOLD:
        return np.array([_cel_iter_scalarvector(*args)
                         for args in zip(qc, p, g, cc, ss, em, kk, strict=False)])  # fmt: skip
    return _cel_iter_scalarvector(qc, p, g, cc, ss, em, kk, xp=xp, any_=xp.any)


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
