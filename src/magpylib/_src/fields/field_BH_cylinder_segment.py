# pylint: disable=too-many-lines
# pylint: disable=line-too-long
# pylint: disable=missing-function-docstring
# pylint: disable=no-name-in-module
# pylint: disable=too-many-statements
# pylint: disable=too-many-positional-arguments
from __future__ import annotations

import array_api_extra as xpx
import numpy as np
from array_api_compat import array_namespace
from scipy.constants import mu_0 as MU0

from magpylib._src.array_api_utils import xp_promote
from magpylib._src.fields.field_BH_cylinder import BHJM_magnet_cylinder
from magpylib._src.fields.special_el3 import el3_angle
from magpylib._src.fields.special_elliptic import ellipeinc, ellipkinc
from magpylib._src.input_checks import check_field_input


def arctan_k_tan_2(k, phi):
    """
    help function for periodic continuation

    what is this function doing exactly ? what are the argument types, ranges, ...

    can be replaced by non-masked version ?
    """
    xp = array_namespace(k, phi)

    full_periods = xp.round(phi / (2.0 * xp.pi))
    phi_red = phi - full_periods * 2.0 * xp.pi

    result = full_periods * xp.pi

    return xpx.apply_where(
        xp.abs(phi_red) < xp.pi,
        (result, k, phi_red),
        lambda result, k, phi_red: result + xp.atan(k * xp.tan(phi_red / 2.0)),
        lambda result, k, phi_red: result + phi_red / 2.0,
    )


def close(arg1: np.ndarray, arg2: np.ndarray) -> np.ndarray:
    """
    determine if arg1 and arg2 lie close to each other
    input: ndarray, shape (n,) or numpy-interpretable scalar
    output: ndarray, dtype=bool
    """
    return xpx.isclose(arg1, arg2, rtol=1e-12, atol=1e-12)


def determine_cases(r, phi, z, r1, phi1, z1):
    """
    Determine case of input parameter set.
        r, phi, z: observer positions
        r1, phi1, z1: boundary values

    All inputs must be ndarrays, shape (n,)

    Returns: case numbers, ndarray, shape (n,), dtype=int

    The case number is a three digits integer, where the digits can be the following values
      1st digit: 1:z=z1,  2:general
      2nd digit: 1:phi-phi1= 2n*pi,  2:phi-phi1=(2n+1)*pi,  3:general
      3rd digit: 1:r=r1=0,  2:r=0,  3:r1=0,  4:r=r1>0,  5:general
    """
    xp = array_namespace(r, phi, z, r1, phi1, z1)
    n = r.shape[0]  # input length

    # allocate result
    result = xp.ones((3, n))

    mask_ind0 = xp.full_like(result, False, dtype=xp.bool)
    mask_ind0 = xpx.at(mask_ind0)[0, ...].set(True)

    # identify z-case
    mask_z = close(z, z1)
    result = xpx.at(result)[0, ...].set(200)
    result = xpx.at(result)[mask_ind0 & mask_z[xp.newaxis, ...]].set(100)

    # identify phi-case
    mod_2pi = xp.abs(phi - phi1) % (2 * xp.pi)
    mask_phi1 = xp.logical_or(close(mod_2pi, 0), close(mod_2pi, 2 * xp.pi))
    mod_pi = xp.abs(phi - phi1) % xp.pi
    mask_phi2 = xp.logical_or(close(mod_pi, 0), close(mod_pi, xp.pi))

    mask_ind1 = xp.full_like(result, False, dtype=xp.bool)
    mask_ind1 = xpx.at(mask_ind1)[1, ...].set(True)

    result = xpx.at(result)[1, ...].set(30)
    result = xpx.at(result)[mask_ind1 & mask_phi2[xp.newaxis, ...]].set(20)
    result = xpx.at(result)[mask_ind1 & mask_phi1[xp.newaxis, ...]].set(10)

    mask_ind2 = xp.full_like(result, False, dtype=xp.bool)
    mask_ind2 = xpx.at(mask_ind2)[2, ...].set(True)

    # identify r-case
    mask_r2 = close(r, 0)
    mask_r3 = close(r1, 0)
    mask_r4 = close(r, r1)
    mask_r1 = mask_r2 & mask_r3
    result = xpx.at(result)[2, ...].set(5)
    result = xpx.at(result)[mask_ind2 & mask_r4].set(4)
    result = xpx.at(result)[mask_ind2 & mask_r3].set(3)
    result = xpx.at(result)[mask_ind2 & mask_r2].set(2)
    result = xpx.at(result)[mask_ind2 & mask_r1].set(1)

    return xp.asarray(xp.sum(result, axis=0), dtype=xp.int32)


# Implementation of all non-zero field components in every special case
# e.g. Hphi_zk stands for field component in phi-direction originating
# from the cylinder tile face at zk

# 112 ##############


def Hphi_zk_case112(xp, r_i, theta_M):
    return xp.cos(theta_M) * xp.log(r_i)


def Hz_ri_case112(xp, phi_bar_M, theta_M):
    return -xp.sin(theta_M) * xp.sin(phi_bar_M)


def Hz_phij_case112(xp, r_i, phi_bar_M, theta_M):
    return xp.sin(theta_M) * xp.sin(phi_bar_M) * xp.log(r_i)


# 113 ##############


def Hphi_zk_case113(xp, r, theta_M):
    return -xp.cos(theta_M) * xp.log(r)


def Hz_phij_case113(xp, r, phi_bar_M, theta_M):
    return -xp.sin(theta_M) * xp.sin(phi_bar_M) * xp.log(r)


# 115 ##############


def Hr_zk_case115(xp, r, r_i, r_bar_i, phi_bar_j, theta_M):
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r * r_i / r_bar_i**2)
    E_coef = xp.cos(theta_M) * xp.abs(r_bar_i) / r
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r * r_i / r_bar_i**2)
    F_coef = -xp.cos(theta_M) * (r**2 + r_i**2) / (r * xp.abs(r_bar_i))
    return E_coef * E + F_coef * F


def Hphi_zk_case115(xp, r, r_i, r_bar_i, theta_M):
    t1 = r_i / r
    t1_coef = -xp.cos(theta_M) * xp.sign(r_bar_i)
    t2 = xp.log(xp.abs(r_bar_i)) * xp.sign(r_bar_i)
    t2_coef = -xp.cos(theta_M)
    return t1_coef * t1 + t2_coef * t2


def Hz_ri_case115(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M):
    t1 = xp.abs(r_bar_i) / r
    t1_coef = xp.sin(theta_M) * xp.sin(phi_bar_M)
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r * r_i / r_bar_i**2)
    E_coef = xp.sin(theta_M) * xp.cos(phi_bar_M) * xp.abs(r_bar_i) / r
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r * r_i / r_bar_i**2)
    F_coef = (
        -xp.sin(theta_M) * xp.cos(phi_bar_M) * (r**2 + r_i**2) / (r * xp.abs(r_bar_i))
    )
    return t1_coef * t1 + E_coef * E + F_coef * F


def Hz_phij_case115(xp, r_bar_i, phi_bar_M, theta_M):
    t1 = xp.log(xp.abs(r_bar_i)) * xp.sign(r_bar_i)
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_M)
    return t1_coef * t1


# 122 ##############


def Hphi_zk_case122(xp, r_i, theta_M):
    return -xp.cos(theta_M) * xp.log(r_i)


def Hz_ri_case122(xp, phi_bar_M, theta_M):
    return xp.sin(theta_M) * xp.sin(phi_bar_M)


def Hz_phij_case122(xp, r_i, phi_bar_M, theta_M):
    return -xp.sin(theta_M) * xp.sin(phi_bar_M) * xp.log(r_i)


# 123 ##############


def Hphi_zk_case123(xp, r, theta_M):
    return -xp.cos(theta_M) * xp.log(r)


def Hz_phij_case123(xp, r, phi_bar_M, theta_M):
    return -xp.sin(theta_M) * xp.sin(phi_bar_M) * xp.log(r)


# 124 ##############


def Hphi_zk_case124(xp, r, theta_M):
    return xp.cos(theta_M) * (1.0 - xp.log(2.0 * r))


def Hz_ri_case124(xp, phi_bar_M, theta_M):
    return 2.0 * xp.sin(theta_M) * xp.sin(phi_bar_M)


def Hz_phij_case124(xp, r, phi_bar_M, theta_M):
    return -xp.sin(theta_M) * xp.sin(phi_bar_M) * xp.log(2.0 * r)


# 125 ##############


def Hr_zk_case125(xp, r, r_i, r_bar_i, phi_bar_j, theta_M):
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r * r_i / r_bar_i**2)
    E_coef = xp.cos(theta_M) * xp.abs(r_bar_i) / r
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r * r_i / r_bar_i**2)
    F_coef = -xp.cos(theta_M) * (r**2 + r_i**2) / (r * xp.abs(r_bar_i))
    return E_coef * E + F_coef * F


def Hphi_zk_case125(xp, r, r_i, theta_M):
    return xp.cos(theta_M) / r * (r_i - r * xp.log(r + r_i))


def Hz_ri_case125(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M):
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r * r_i / r_bar_i**2)
    E_coef = xp.sin(theta_M) * xp.cos(phi_bar_M) * xp.abs(r_bar_i) / r
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r * r_i / r_bar_i**2)
    F_coef = (
        -xp.sin(theta_M) * xp.cos(phi_bar_M) * (r**2 + r_i**2) / (r * xp.abs(r_bar_i))
    )
    return xp.sin(theta_M) * xp.sin(phi_bar_M) * (r + r_i) / r + E_coef * E + F_coef * F


def Hz_phij_case125(xp, r, r_i, phi_bar_M, theta_M):
    return -xp.sin(theta_M) * xp.sin(phi_bar_M) * xp.log(r + r_i)


# 132 ##############


def Hr_zk_case132(xp, r_i, phi_bar_j, theta_M):
    return xp.cos(theta_M) * xp.sin(phi_bar_j) * xp.log(r_i)


def Hphi_zk_case132(xp, r_i, phi_bar_j, theta_M):
    return xp.cos(theta_M) * xp.cos(phi_bar_j) * xp.log(r_i)


def Hz_ri_case132(xp, phi_bar_Mj, theta_M):
    return -xp.sin(theta_M) * xp.sin(phi_bar_Mj)


def Hz_phij_case132(xp, r_i, phi_bar_Mj, theta_M):
    return xp.sin(theta_M) * xp.sin(phi_bar_Mj) * xp.log(r_i)


# 133 ##############


def Hr_zk_case133(xp, r, phi_bar_j, theta_M):
    return -xp.cos(theta_M) * xp.sin(phi_bar_j) + xp.cos(theta_M) * xp.sin(
        phi_bar_j
    ) * xp.log(r * (1.0 - xp.cos(phi_bar_j)))


def Hphi_zk_case133(xp, phi_bar_j, theta_M):
    return xp.cos(theta_M) - xp.cos(theta_M) * xp.cos(phi_bar_j) * xp.atanh(
        xp.cos(phi_bar_j)
    )


def Hz_phij_case133(xp, phi_bar_j, phi_bar_Mj, theta_M):
    return -xp.sin(theta_M) * xp.sin(phi_bar_Mj) * xp.atanh(xp.cos(phi_bar_j))


# 134 ##############


def Hr_zk_case134(xp, r, phi_bar_j, theta_M):
    t1 = xp.sin(phi_bar_j)
    t1_coef = -xp.cos(theta_M)
    t2 = xp.sin(phi_bar_j) / xp.sqrt(1.0 - xp.cos(phi_bar_j))
    t2_coef = -xp.sqrt(2.0) * xp.cos(theta_M)
    t3 = xp.log(
        r * (1.0 - xp.cos(phi_bar_j) + xp.sqrt(2.0) * xp.sqrt(1.0 - xp.cos(phi_bar_j)))
    )
    t3_coef = xp.cos(theta_M) * xp.sin(phi_bar_j)
    t4 = xp.atanh(xp.sin(phi_bar_j) / (xp.sqrt(2.0) * xp.sqrt(1.0 - xp.cos(phi_bar_j))))
    t4_coef = xp.cos(theta_M)
    return t1_coef * t1 + t2_coef * t2 + t3_coef * t3 + t4_coef * t4


def Hphi_zk_case134(xp, phi_bar_j, theta_M):
    return xp.sqrt(2) * xp.cos(theta_M) * xp.sqrt(1 - xp.cos(phi_bar_j)) + xp.cos(
        theta_M
    ) * xp.cos(phi_bar_j) * xp.atanh(xp.sqrt((1 - xp.cos(phi_bar_j)) / 2))


def Hz_ri_case134(xp, phi_bar_j, phi_bar_M, theta_M):
    t1 = xp.sqrt(1.0 - xp.cos(phi_bar_j))
    t1_coef = xp.sqrt(2.0) * xp.sin(theta_M) * xp.sin(phi_bar_M)
    t2 = xp.sin(phi_bar_j) / t1
    t2_coef = -xp.sqrt(2.0) * xp.sin(theta_M) * xp.cos(phi_bar_M)
    t3 = xp.atanh(t2 / xp.sqrt(2.0))
    t3_coef = xp.sin(theta_M) * xp.cos(phi_bar_M)
    return t1_coef * t1 + t2_coef * t2 + t3_coef * t3


def Hz_phij_case134(xp, phi_bar_j, phi_bar_Mj, theta_M):
    return (
        xp.sin(theta_M)
        * xp.sin(phi_bar_Mj)
        * xp.atanh(xp.sqrt((1.0 - xp.cos(phi_bar_j)) / 2.0))
    )


# 135 ##############


def Hr_zk_case135(xp, r, r_i, r_bar_i, phi_bar_j, theta_M):
    t1 = xp.sin(phi_bar_j)
    t1_coef = -xp.cos(theta_M)
    t2 = xp.log(
        r_i
        - r * xp.cos(phi_bar_j)
        + xp.sqrt(r_i**2 + r**2 - 2 * r_i * r * xp.cos(phi_bar_j))
    )
    t2_coef = xp.cos(theta_M) * xp.sin(phi_bar_j)
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r * r_i / r_bar_i**2)
    E_coef = xp.cos(theta_M) * xp.abs(r_bar_i) / r
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r * r_i / r_bar_i**2)
    F_coef = -xp.cos(theta_M) * (r**2 + r_i**2) / (r * xp.abs(r_bar_i))
    return t1_coef * t1 + t2_coef * t2 + E_coef * E + F_coef * F


def Hphi_zk_case135(xp, r, r_i, phi_bar_j, theta_M):
    t1 = xp.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * xp.cos(phi_bar_j))
    t1_coef = xp.cos(theta_M) / r
    t2 = xp.atanh((r * xp.cos(phi_bar_j) - r_i) / t1)
    t2_coef = -xp.cos(theta_M) * xp.cos(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2


def Hz_ri_case135(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M):
    t = r_bar_i**2
    t1 = xp.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * xp.cos(phi_bar_j)) / r
    t1_coef = xp.sin(theta_M) * xp.sin(phi_bar_M)
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r * r_i / t)
    E_coef = xp.sin(theta_M) * xp.cos(phi_bar_M) * xp.sqrt(t) / r
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r * r_i / t)
    F_coef = -xp.sin(theta_M) * xp.cos(phi_bar_M) * (r**2 + r_i**2) / (r * xp.sqrt(t))
    return t1_coef * t1 + E_coef * E + F_coef * F


def Hz_phij_case135(xp, r, r_i, phi_bar_j, phi_bar_Mj, theta_M):
    t1 = xp.atanh(
        (r * xp.cos(phi_bar_j) - r_i)
        / xp.sqrt(r**2 + r_i**2 - 2 * r * r_i * xp.cos(phi_bar_j))
    )
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_Mj)
    return t1_coef * t1


# 211 ##############


def Hr_phij_case211(xp, phi_bar_M, theta_M, z_bar_k):
    return (
        -xp.sin(theta_M)
        * xp.sin(phi_bar_M)
        * xp.sign(z_bar_k)
        * xp.log(xp.abs(z_bar_k))
    )


def Hz_zk_case211(xp, phi_j, theta_M, z_bar_k):
    return -xp.cos(theta_M) * xp.sign(z_bar_k) * phi_j


# 212 ##############


def Hr_ri_case212(xp, r_i, phi_j, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.sin(theta_M) * z_bar_k / xp.sqrt(r_i**2 + z_bar_k**2)
    t2 = 1.0 / 2.0 * phi_j * xp.cos(phi_bar_M)
    t3 = 1.0 / 4.0 * xp.sin(phi_bar_M)
    return t1 * (t2 - t3)


def Hr_phij_case212(xp, r_i, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.atanh(z_bar_k / xp.sqrt(r_i**2 + z_bar_k**2))
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_M)
    return t1_coef * t1


def Hphi_ri_case212(xp, r_i, phi_j, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.sin(theta_M) * z_bar_k / xp.sqrt(r_i**2 + z_bar_k**2)
    t2 = 1.0 / 4.0 * xp.cos(phi_bar_M)
    t3 = 1.0 / 2.0 * phi_j * xp.sin(phi_bar_M)
    return t1 * (-t2 + t3)


def Hphi_zk_case212(xp, r_i, theta_M, z_bar_k):
    t1 = r_i / xp.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = -xp.cos(theta_M)
    t2 = xp.atanh(t1)
    t2_coef = xp.cos(theta_M)
    return t1_coef * t1 + t2_coef * t2


def Hz_ri_case212(xp, r_i, phi_bar_M, theta_M, z_bar_k):
    t1 = r_i / xp.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_M)
    return t1_coef * t1


def Hz_phij_case212(xp, r_i, phi_bar_M, theta_M, z_bar_k):
    return (
        xp.sin(theta_M)
        * xp.sin(phi_bar_M)
        * xp.atanh(r_i / xp.sqrt(r_i**2 + z_bar_k**2))
    )


def Hz_zk_case212(xp, r_i, phi_j, theta_M, z_bar_k):
    t1 = phi_j / xp.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = -xp.cos(theta_M) * z_bar_k
    return t1_coef * t1


# 213 ##############


def Hr_phij_case213(xp, r, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.atanh(z_bar_k / xp.sqrt(r**2 + z_bar_k**2))
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_M)
    return t1_coef * t1


def Hphi_zk_case213(xp, r, theta_M, z_bar_k):
    t1 = xp.sqrt(r**2 + z_bar_k**2)
    t1_coef = xp.cos(theta_M) / r
    t2 = xp.atanh(r / t1)
    t2_coef = -xp.cos(theta_M)
    return t1_coef * t1 + t2_coef * t2


def Hz_phij_case213(xp, r, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.atanh(r / xp.sqrt(r**2 + z_bar_k**2))
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_M)
    return t1_coef * t1


def Hz_zk_case213(xp, phi_bar_j, theta_M, z_bar_k):
    t1 = xp.sign(z_bar_k)
    t1_coef = xp.cos(theta_M) * phi_bar_j
    return t1_coef * t1


# 214 ##############


def Hr_ri_case214(xp, r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = (
        -xp.sin(theta_M)
        * xp.cos(phi_bar_M)
        * z_bar_k**2
        * xp.sign(z_bar_k)
        / (2.0 * r**2)
    )
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = (
        xp.sin(theta_M)
        * xp.cos(phi_bar_M)
        * xp.sign(z_bar_k)
        * (2.0 * r**2 + z_bar_k**2)
        / (2.0 * r**2)
    )
    return (
        -xp.sin(theta_M)
        * xp.sin(phi_bar_M)
        * xp.sign(z_bar_k)
        * z_bar_k**2
        / (2.0 * r**2)
        + E_coef * E
        + F_coef * F
    )


def Hr_phij_case214(xp, phi_bar_M, theta_M, z_bar_k):
    return (
        -xp.sin(theta_M)
        * xp.sin(phi_bar_M)
        * xp.sign(z_bar_k)
        * xp.log(xp.abs(z_bar_k))
    )


def Hr_zk_case214(xp, r, phi_bar_j, theta_M, z_bar_k):
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = xp.cos(theta_M) * xp.abs(z_bar_k) / r
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = -xp.cos(theta_M) * (2.0 * r**2 + z_bar_k**2) / (r * xp.abs(z_bar_k))
    t = xp.sqrt(r**2 + z_bar_k**2)

    def Pi1(sign):
        return el3_angle(phi_bar_j / 2, 2 * r / (r + sign * t), -4 * r**2 / z_bar_k**2)

    def Pi1_coef(sign):
        return (
            -xp.cos(theta_M)
            / (r * xp.sqrt((r**2 + z_bar_k**2) * z_bar_k**2))
            * (t - sign * r)
            * (r + sign * t) ** 2
        )

    def Pi2(sign):
        return el3_angle(
            phi_bar_j / 2.0,
            1.0 - z_bar_k**4 / ((4.0 * r**2 + z_bar_k**2) * (r + sign * t) ** 2),
            4.0 * r**2 / (4.0 * r**2 + z_bar_k**2),
        )

    def Pi2_coef(sign):
        return (
            sign
            * xp.cos(theta_M)
            * z_bar_k**4
            / (
                r
                * xp.sqrt((r**2 + z_bar_k**2) * (4.0 * r**2 + z_bar_k**2))
                * (r + sign * t)
            )
        )

    return (
        E_coef * E
        + F_coef * F
        + Pi1_coef(1) * Pi1(1)
        + Pi1_coef(-1) * Pi1(-1)
        + Pi2_coef(1) * Pi2(1)
        + Pi2_coef(-1) * Pi2(-1)
    )


def Hphi_ri_case214(xp, r, phi_j, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = -xp.sin(theta_M) * xp.cos(phi_bar_M) * xp.sign(z_bar_k) / 2.0
    t2 = phi_j
    t2_coef = xp.sin(theta_M) * xp.sin(phi_bar_M) * xp.sign(z_bar_k) / 2.0
    t3 = xp.sign(z_bar_k) * z_bar_k**2 / (2.0 * r**2)
    t3_coef = -xp.sin(theta_M) * xp.cos(phi_bar_M)
    t4 = xp.log(xp.abs(z_bar_k) / (xp.sqrt(2.0) * r))
    t4_coef = -xp.sin(theta_M) * xp.cos(phi_bar_M) * xp.sign(z_bar_k)
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = (
        xp.sin(theta_M)
        * xp.sin(phi_bar_M)
        * z_bar_k**2
        * xp.sign(z_bar_k)
        / (2.0 * r**2)
    )
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = (
        -xp.sin(theta_M)
        * xp.sin(phi_bar_M)
        * xp.sign(z_bar_k)
        * (4.0 * r**2 + z_bar_k**2)
        / (2.0 * r**2)
    )
    return t1 + t2_coef * t2 + t3_coef * t3 + t4_coef * t4 + E_coef * E + F_coef * F


def Hphi_zk_case214(xp, r, theta_M, z_bar_k):
    t1 = xp.abs(z_bar_k)
    t1_coef = xp.cos(theta_M) / r
    return t1_coef * t1


def Hz_ri_case214(xp, r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = xp.sin(theta_M) * xp.cos(phi_bar_M) * xp.abs(z_bar_k) / r
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = (
        -xp.sin(theta_M)
        * xp.cos(phi_bar_M)
        * (2.0 * r**2 + z_bar_k**2)
        / (r * xp.abs(z_bar_k))
    )
    return (
        xp.sin(theta_M) * xp.sin(phi_bar_M) * xp.abs(z_bar_k) / r
        + E_coef * E
        + F_coef * F
    )


def Hz_zk_case214(xp, r, phi_bar_j, theta_M, z_bar_k):
    t = xp.sqrt(r**2 + z_bar_k**2)

    def Pi(sign):
        return el3_angle(phi_bar_j / 2, 2 * r / (r + sign * t), -4 * r**2 / z_bar_k**2)

    Pi_coef = xp.cos(theta_M) * xp.sign(z_bar_k)
    return Pi_coef * Pi(1) + Pi_coef * Pi(-1)


# 215 ##############


def Hr_ri_case215(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t2 = xp.atanh(z_bar_k / xp.sqrt(r_bar_i**2 + z_bar_k**2))
    t2_coef = xp.sin(theta_M) * xp.sin(phi_bar_M) / 2.0 * (1.0 - r_i**2 / r**2)
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = (
        -xp.sin(theta_M)
        * xp.cos(phi_bar_M)
        * z_bar_k
        * xp.sqrt(r_bar_i**2 + z_bar_k**2)
        / (2 * r**2)
    )
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = (
        xp.sin(theta_M)
        * xp.cos(phi_bar_M)
        * z_bar_k
        * (2.0 * r_i**2 + z_bar_k**2)
        / (2 * r**2 * xp.sqrt(r_bar_i**2 + z_bar_k**2))
    )
    Pi = el3_angle(
        phi_bar_j / 2.0,
        -4.0 * r * r_i / r_bar_i**2,
        -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2),
    )
    Pi_coef = (
        xp.sin(theta_M)
        * xp.cos(phi_bar_M)
        * z_bar_k
        * (r**2 + r_i**2)
        * (r + r_i)
        / (2.0 * r**2 * r_bar_i * xp.sqrt(r_bar_i**2 + z_bar_k**2))
    )
    return (
        -xp.sin(theta_M)
        * xp.sin(phi_bar_M)
        * z_bar_k
        * xp.sqrt(r_bar_i**2 + z_bar_k**2)
        / (2.0 * r**2)
        + t2_coef * t2
        + E_coef * E
        + F_coef * F
        + Pi_coef * Pi
    )


def Hr_phij_case215(xp, r_bar_i, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.atanh(z_bar_k / xp.sqrt(r_bar_i**2 + z_bar_k**2))
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_M)
    return t1_coef * t1


def Hr_zk_case215(xp, r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k):
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = xp.cos(theta_M) * xp.sqrt(r_bar_i**2 + z_bar_k**2) / r
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = (
        -xp.cos(theta_M)
        * (r**2 + r_i**2 + z_bar_k**2)
        / (r * xp.sqrt(r_bar_i**2 + z_bar_k**2))
    )
    t = xp.sqrt(r**2 + z_bar_k**2)

    def Pi1(sign):
        return el3_angle(
            phi_bar_j / 2.0,
            2.0 * r / (r + sign * t),
            -4 * r * r_i / (r_bar_i**2 + z_bar_k**2),
        )

    def Pi1_coef(sign):
        return (
            -xp.cos(theta_M)
            / (r * xp.sqrt((r**2 + z_bar_k**2) * (r_bar_i**2 + z_bar_k**2)))
            * (t - sign * r)
            * (r_i + sign * t) ** 2
        )

    def Pi2(sign):
        return el3_angle(
            phi_bar_j / 2.0,
            1.0
            - z_bar_k**2
            * (r_bar_i**2 + z_bar_k**2)
            / (((r + r_i) ** 2 + z_bar_k**2) * (r + sign * t) ** 2),
            4 * r * r_i / ((r + r_i) ** 2 + z_bar_k**2),
        )

    def Pi2_coef(sign):
        return (
            sign
            * xp.cos(theta_M)
            * z_bar_k**2
            * (r_bar_i**2 + z_bar_k**2)
            / (
                r
                * xp.sqrt((r**2 + z_bar_k**2) * ((r + r_i) ** 2 + z_bar_k**2))
                * (r + sign * t)
            )
        )

    return (
        E_coef * E
        + F_coef * F
        + Pi1_coef(1) * Pi1(1)
        + Pi1_coef(-1) * Pi1(-1)
        + Pi2_coef(1) * Pi2(1)
        + Pi2_coef(-1) * Pi2(-1)
    )


def Hphi_ri_case215(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.sqrt(r_bar_i**2 + z_bar_k**2) * z_bar_k / (2.0 * r**2)
    t1_coef = -xp.sin(theta_M) * xp.cos(phi_bar_M)
    t2 = xp.atanh(z_bar_k / xp.sqrt(r_bar_i**2 + z_bar_k**2))
    t2_coef = -xp.sin(theta_M) * xp.cos(phi_bar_M) * (r**2 + r_i**2) / (2.0 * r**2)
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = (
        xp.sin(theta_M)
        * xp.sin(phi_bar_M)
        * z_bar_k
        * xp.sqrt(r_bar_i**2 + z_bar_k**2)
        / (2 * r**2)
    )
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = (
        -xp.sin(theta_M)
        * xp.sin(phi_bar_M)
        * z_bar_k
        * (2.0 * r**2 + 2.0 * r_i**2 + z_bar_k**2)
        / (2.0 * r**2 * xp.sqrt(r_bar_i**2 + z_bar_k**2))
    )
    Pi = el3_angle(
        phi_bar_j / 2.0,
        -4.0 * r * r_i / r_bar_i**2,
        -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2),
    )
    Pi_coef = (
        xp.sin(theta_M)
        * xp.sin(phi_bar_M)
        * z_bar_k
        * (r + r_i) ** 2
        / (2.0 * r**2 * xp.sqrt(r_bar_i**2 + z_bar_k**2))
    )
    return t1_coef * t1 + t2_coef * t2 + E_coef * E + F_coef * F + Pi_coef * Pi


def Hphi_zk_case215(xp, r, r_bar_i, theta_M, z_bar_k):
    t1 = xp.sqrt(r_bar_i**2 + z_bar_k**2)
    t1_coef = xp.cos(theta_M) / r
    t2 = xp.atanh(r_bar_i / t1)
    t2_coef = -xp.cos(theta_M)
    return t1_coef * t1 + t2_coef * t2


def Hz_ri_case215(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t = r_bar_i**2 + z_bar_k**2
    t1 = xp.sqrt(r_bar_i**2 + z_bar_k**2) / r
    t1_coef = xp.sin(theta_M) * xp.sin(phi_bar_M)
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r * r_i / t)
    E_coef = xp.sin(theta_M) * xp.cos(phi_bar_M) * xp.sqrt(t) / r
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r * r_i / t)
    F_coef = (
        -xp.sin(theta_M)
        * xp.cos(phi_bar_M)
        * (r**2 + r_i**2 + z_bar_k**2)
        / (r * xp.sqrt(t))
    )
    return t1_coef * t1 + E_coef * E + F_coef * F


def Hz_phij_case215(xp, r_bar_i, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.atanh(r_bar_i / xp.sqrt(r_bar_i**2 + z_bar_k**2))
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_M)
    return t1_coef * t1


def Hz_zk_case215(xp, r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k):
    t = xp.sqrt(r**2 + z_bar_k**2)

    def Pi(sign):
        return el3_angle(
            phi_bar_j / 2.0,
            2.0 * r / (r + sign * t),
            -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2),
        )

    def Pi_coef(sign):
        return (
            xp.cos(theta_M)
            * z_bar_k
            * (r_i + sign * t)
            / (xp.sqrt(r_bar_i**2 + z_bar_k**2) * (r + sign * t))
        )

    return Pi_coef(1) * Pi(1) + Pi_coef(-1) * Pi(-1)


# 221 ##############


def Hr_phij_case221(xp, phi_bar_M, theta_M, z_bar_k):
    return (
        -xp.sin(theta_M)
        * xp.sin(phi_bar_M)
        * xp.sign(z_bar_k)
        * xp.log(xp.abs(z_bar_k))
    )


def Hz_zk_case221(xp, phi_j, theta_M, z_bar_k):
    return -xp.cos(theta_M) * xp.sign(z_bar_k) * phi_j


# 222 ##############


def Hr_ri_case222(xp, r_i, phi_j, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.sin(theta_M) * z_bar_k / xp.sqrt(r_i**2 + z_bar_k**2)
    t2 = 1.0 / 2.0 * phi_j * xp.cos(phi_bar_M)
    t3 = 1.0 / 4.0 * xp.sin(phi_bar_M)
    return t1 * (t2 - t3)


def Hr_phij_case222(xp, r_i, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.atanh(z_bar_k / xp.sqrt(r_i**2 + z_bar_k**2))
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_M)
    return t1_coef * t1


def Hphi_ri_case222(xp, r_i, phi_j, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.sin(theta_M) * z_bar_k / xp.sqrt(r_i**2 + z_bar_k**2)
    t2 = 1.0 / 4.0 * xp.cos(phi_bar_M)
    t3 = 1.0 / 2.0 * phi_j * xp.sin(phi_bar_M)
    return t1 * (-t2 + t3)


def Hphi_zk_case222(xp, r_i, theta_M, z_bar_k):
    t1 = r_i / xp.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = xp.cos(theta_M)
    t2 = xp.atanh(t1)
    t2_coef = -xp.cos(theta_M)
    return t1_coef * t1 + t2_coef * t2


def Hz_ri_case222(xp, r_i, phi_bar_M, theta_M, z_bar_k):
    t1 = r_i / xp.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = xp.sin(theta_M) * xp.sin(phi_bar_M)
    return t1_coef * t1


def Hz_phij_case222(xp, r_i, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.atanh(r_i / xp.sqrt(r_i**2 + z_bar_k**2))
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_M)
    return t1_coef * t1


def Hz_zk_case222(xp, r_i, phi_j, theta_M, z_bar_k):
    t1 = z_bar_k / xp.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = -xp.cos(theta_M) * phi_j
    return t1_coef * t1


# 223 ##############


def Hr_phij_case223(xp, r, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.atanh(z_bar_k / xp.sqrt(r**2 + z_bar_k**2))
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_M)
    return t1_coef * t1


def Hphi_zk_case223(xp, r, theta_M, z_bar_k):
    t1 = xp.sqrt(r**2 + z_bar_k**2)
    t1_coef = xp.cos(theta_M) / r
    t2 = xp.atanh(r / t1)
    t2_coef = -xp.cos(theta_M)
    return t1_coef * t1 + t2_coef * t2


def Hz_phij_case223(xp, r, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.atanh(r / xp.sqrt(r**2 + z_bar_k**2))
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_M)
    return t1_coef * t1


def Hz_zk_case223(xp, r, phi_bar_j, theta_M, z_bar_k):
    t1 = arctan_k_tan_2(xp.sqrt(r**2 + z_bar_k**2) / xp.abs(z_bar_k), 2.0 * phi_bar_j)
    t1_coef = xp.cos(theta_M) * xp.sign(z_bar_k)
    return t1_coef * t1


# 224 ##############


def Hr_ri_case224(xp, r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.sqrt(4.0 * r**2 + z_bar_k**2) * z_bar_k / (2.0 * r**2)
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_M)
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = (
        -xp.sin(theta_M)
        * xp.cos(phi_bar_M)
        * z_bar_k**2
        * xp.sign(z_bar_k)
        / (2.0 * r**2)
    )
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = (
        xp.sin(theta_M)
        * xp.cos(phi_bar_M)
        * xp.sign(z_bar_k)
        * (2.0 * r**2 + z_bar_k**2)
        / (2.0 * r**2)
    )
    return t1_coef * t1 + E_coef * E + F_coef * F


def Hr_phij_case224(xp, r, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.atanh(z_bar_k / xp.sqrt(4.0 * r**2 + z_bar_k**2))
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_M)
    return t1_coef * t1


def Hr_zk_case224(xp, r, phi_bar_j, theta_M, z_bar_k):
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = xp.cos(theta_M) * xp.abs(z_bar_k) / r
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = -xp.cos(theta_M) * (2.0 * r**2 + z_bar_k**2) / (r * xp.abs(z_bar_k))
    t = xp.sqrt(r**2 + z_bar_k**2)

    def Pi1(sign):
        return el3_angle(
            phi_bar_j / 2.0, 2.0 * r / (r + sign * t), -4.0 * r**2 / z_bar_k**2
        )

    def Pi1_coef(sign):
        return (
            -xp.cos(theta_M)
            / (r * xp.sqrt((r**2 + z_bar_k**2) * z_bar_k**2))
            * (t - sign * r)
            * (r + sign * t) ** 2
        )

    def Pi2(sign):
        return el3_angle(
            phi_bar_j / 2.0,
            1.0 - z_bar_k**4 / ((4.0 * r**2 + z_bar_k**2) * (r + sign * t) ** 2),
            4.0 * r**2 / (4.0 * r**2 + z_bar_k**2),
        )

    def Pi2_coef(sign):
        return (
            sign
            * xp.cos(theta_M)
            * z_bar_k**4
            / (
                r
                * xp.sqrt((r**2 + z_bar_k**2) * (4.0 * r**2 + z_bar_k**2))
                * (r + sign * t)
            )
        )

    return (
        E_coef * E
        + F_coef * F
        + Pi1_coef(1) * Pi1(1)
        + Pi1_coef(-1) * Pi1(-1)
        + Pi2_coef(1) * Pi2(1)
        + Pi2_coef(-1) * Pi2(-1)
    )


def Hphi_ri_case224(xp, r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.sqrt(4.0 * r**2 + z_bar_k**2) * z_bar_k / (2.0 * r**2)
    t1_coef = -xp.sin(theta_M) * xp.cos(phi_bar_M)
    t2 = xp.atanh(z_bar_k / xp.sqrt(4.0 * r**2 + z_bar_k**2))
    t2_coef = -xp.sin(theta_M) * xp.cos(phi_bar_M)
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = (
        xp.sin(theta_M)
        * xp.sin(phi_bar_M)
        * z_bar_k**2
        * xp.sign(z_bar_k)
        / (2.0 * r**2)
    )
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = (
        -xp.sin(theta_M)
        * xp.sin(phi_bar_M)
        * xp.sign(z_bar_k)
        * (4.0 * r**2 + z_bar_k**2)
        / (2.0 * r**2)
    )
    return t1_coef * t1 + t2_coef * t2 + E_coef * E + F_coef * F


def Hphi_zk_case224(xp, r, theta_M, z_bar_k):
    t1 = xp.sqrt(4.0 * r**2 + z_bar_k**2)
    t1_coef = xp.cos(theta_M) / r
    t2 = xp.atanh(2.0 * r / t1)
    t2_coef = -xp.cos(theta_M)
    return t1_coef * t1 + t2_coef * t2


def Hz_ri_case224(xp, r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.sqrt(4.0 * r**2 + z_bar_k**2) / r
    t1_coef = xp.sin(theta_M) * xp.sin(phi_bar_M)
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = xp.sin(theta_M) * xp.cos(phi_bar_M) * xp.abs(z_bar_k) / r
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = (
        -xp.sin(theta_M)
        * xp.cos(phi_bar_M)
        * (2.0 * r**2 + z_bar_k**2)
        / (r * xp.abs(z_bar_k))
    )
    return t1_coef * t1 + E_coef * E + F_coef * F


def Hz_phij_case224(xp, r, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.atanh(2.0 * r / xp.sqrt(4.0 * r**2 + z_bar_k**2))
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_M)
    return t1_coef * t1


def Hz_zk_case224(xp, r, phi_bar_j, theta_M, z_bar_k):
    t = xp.sqrt(r**2 + z_bar_k**2)

    def Pi(sign):
        return el3_angle(
            phi_bar_j / 2.0, 2.0 * r / (r + sign * t), -4.0 * r**2 / z_bar_k**2
        )

    Pi_coef = xp.cos(theta_M) * xp.sign(z_bar_k)
    return Pi_coef * Pi(1) + Pi_coef * Pi(-1)


# 225 ##############


def Hr_ri_case225(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.sqrt((r + r_i) ** 2 + z_bar_k**2) * z_bar_k / (2.0 * r**2)
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_M)
    t2 = xp.atanh(z_bar_k / xp.sqrt((r + r_i) ** 2 + z_bar_k**2))
    t2_coef = xp.sin(theta_M) * xp.sin(phi_bar_M) / 2.0 * (1.0 - r_i**2 / r**2)
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = (
        -xp.sin(theta_M)
        * xp.cos(phi_bar_M)
        * z_bar_k
        * xp.sqrt(r_bar_i**2 + z_bar_k**2)
        / (2.0 * r**2)
    )
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = (
        xp.sin(theta_M)
        * xp.cos(phi_bar_M)
        * z_bar_k
        * (2.0 * r_i**2 + z_bar_k**2)
        / (2.0 * r**2 * xp.sqrt(r_bar_i**2 + z_bar_k**2))
    )
    Pi = el3_angle(
        phi_bar_j / 2.0,
        -4.0 * r * r_i / r_bar_i**2,
        -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2),
    )
    Pi_coef = (
        xp.sin(theta_M)
        * xp.cos(phi_bar_M)
        * z_bar_k
        * (r**2 + r_i**2)
        * (r + r_i)
        / (2.0 * r**2 * r_bar_i * xp.sqrt(r_bar_i**2 + z_bar_k**2))
    )
    return t1_coef * t1 + t2_coef * t2 + E_coef * E + F_coef * F + Pi_coef * Pi


def Hr_phij_case225(xp, r, r_i, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.atanh(z_bar_k / xp.sqrt((r + r_i) ** 2 + z_bar_k**2))
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_M)
    return t1_coef * t1


def Hr_zk_case225(xp, r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k):
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = xp.cos(theta_M) * xp.sqrt(r_bar_i**2 + z_bar_k**2) / r
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = (
        -xp.cos(theta_M)
        * (r**2 + r_i**2 + z_bar_k**2)
        / (r * xp.sqrt(r_bar_i**2 + z_bar_k**2))
    )
    t = xp.sqrt(r**2 + z_bar_k**2)

    def Pi1(sign):
        return el3_angle(
            phi_bar_j / 2.0,
            2.0 * r / (r + sign * t),
            -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2),
        )

    def Pi1_coef(sign):
        return (
            -xp.cos(theta_M)
            / (r * xp.sqrt((r**2 + z_bar_k**2) * (r_bar_i**2 + z_bar_k**2)))
            * (t - sign * r)
            * (r_i + sign * t) ** 2
        )

    def Pi2(sign):
        return el3_angle(
            phi_bar_j / 2.0,
            1.0
            - z_bar_k**2
            * (r_bar_i**2 + z_bar_k**2)
            / (((r + r_i) ** 2 + z_bar_k**2) * (r + sign * t) ** 2),
            4.0 * r * r_i / ((r + r_i) ** 2 + z_bar_k**2),
        )

    def Pi2_coef(sign):
        return (
            sign
            * xp.cos(theta_M)
            * z_bar_k**2
            * (r_bar_i**2 + z_bar_k**2)
            / (
                r
                * xp.sqrt((r**2 + z_bar_k**2) * ((r + r_i) ** 2 + z_bar_k**2))
                * (r + sign * t)
            )
        )

    return (
        E_coef * E
        + F_coef * F
        + Pi1_coef(1) * Pi1(1)
        + Pi1_coef(-1) * Pi1(-1)
        + Pi2_coef(1) * Pi2(1)
        + Pi2_coef(-1) * Pi2(-1)
    )


def Hphi_ri_case225(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.sqrt((r + r_i) ** 2 + z_bar_k**2) * z_bar_k / (2.0 * r**2)
    t1_coef = -xp.sin(theta_M) * xp.cos(phi_bar_M)
    t2 = xp.atanh(z_bar_k / xp.sqrt((r + r_i) ** 2 + z_bar_k**2))
    t2_coef = -xp.sin(theta_M) * xp.cos(phi_bar_M) * (r**2 + r_i**2) / (2.0 * r**2)
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = (
        xp.sin(theta_M)
        * xp.sin(phi_bar_M)
        * z_bar_k
        * xp.sqrt(r_bar_i**2 + z_bar_k**2)
        / (2.0 * r**2)
    )
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = (
        -xp.sin(theta_M)
        * xp.sin(phi_bar_M)
        * z_bar_k
        * (2.0 * r**2 + 2.0 * r_i**2 + z_bar_k**2)
        / (2.0 * r**2 * xp.sqrt(r_bar_i**2 + z_bar_k**2))
    )
    Pi = el3_angle(
        phi_bar_j / 2.0,
        -4.0 * r * r_i / r_bar_i**2,
        -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2),
    )
    Pi_coef = (
        xp.sin(theta_M)
        * xp.sin(phi_bar_M)
        * z_bar_k
        * (r + r_i) ** 2
        / (2.0 * r**2 * xp.sqrt(r_bar_i**2 + z_bar_k**2))
    )
    return t1_coef * t1 + t2_coef * t2 + E_coef * E + F_coef * F + Pi_coef * Pi


def Hphi_zk_case225(xp, r, r_i, theta_M, z_bar_k):
    t1 = xp.sqrt((r + r_i) ** 2 + z_bar_k**2)
    t1_coef = xp.cos(theta_M) / r
    t2 = xp.atanh((r + r_i) / t1)
    t2_coef = -xp.cos(theta_M)
    return t1_coef * t1 + t2_coef * t2


def Hz_ri_case225(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t = r_bar_i**2 + z_bar_k**2
    t1 = xp.sqrt((r + r_i) ** 2 + z_bar_k**2) / r
    t1_coef = xp.sin(theta_M) * xp.sin(phi_bar_M)
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r * r_i / t)
    E_coef = xp.sin(theta_M) * xp.cos(phi_bar_M) * xp.sqrt(t) / r
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r * r_i / t)
    F_coef = (
        -xp.sin(theta_M)
        * xp.cos(phi_bar_M)
        * (r**2 + r_i**2 + z_bar_k**2)
        / (r * xp.sqrt(t))
    )
    return t1_coef * t1 + E_coef * E + F_coef * F


def Hz_phij_case225(xp, r, r_i, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.atanh((r + r_i) / xp.sqrt((r + r_i) ** 2 + z_bar_k**2))
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_M)
    return t1_coef * t1


def Hz_zk_case225(xp, r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k):
    t = xp.sqrt(r**2 + z_bar_k**2)

    def Pi(sign):
        return el3_angle(
            phi_bar_j / 2.0,
            2.0 * r / (r + sign * t),
            -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2),
        )

    def Pi_coef(sign):
        return (
            xp.cos(theta_M)
            * z_bar_k
            * (r_i + sign * t)
            / (xp.sqrt(r_bar_i**2 + z_bar_k**2) * (r + sign * t))
        )

    return Pi_coef(1) * Pi(1) + Pi_coef(-1) * Pi(-1)


# 231 ##############


def Hr_phij_case231(xp, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    return (
        -xp.sin(theta_M)
        * xp.sin(phi_bar_Mj)
        * xp.cos(phi_bar_j)
        * xp.sign(z_bar_k)
        * xp.log(xp.abs(z_bar_k))
    )


def Hphi_phij_case231(xp, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t1 = xp.log(xp.abs(z_bar_k))
    t1_coef = (
        xp.sin(theta_M) * xp.sin(phi_bar_Mj) * xp.sin(phi_bar_j) * xp.sign(z_bar_k)
    )
    return t1_coef * t1


def Hz_zk_case231(xp, phi_j, theta_M, z_bar_k):
    t1 = phi_j * xp.sign(z_bar_k)
    t1_coef = -xp.cos(theta_M)
    return t1_coef * t1


# 232 ##############


def Hr_ri_case232(xp, r_i, phi_j, phi_bar_j, phi_bar_M, phi_bar_Mj, theta_M, z_bar_k):
    t1 = xp.sin(theta_M) * z_bar_k / xp.sqrt(r_i**2 + z_bar_k**2)
    t2 = 1.0 / 2.0 * phi_j * xp.cos(phi_bar_M)
    t3 = 1.0 / 4.0 * xp.sin(phi_bar_Mj + phi_bar_j)
    return t1 * (t2 - t3)


def Hr_phij_case232(xp, r_i, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t1 = xp.atanh(z_bar_k / xp.sqrt(r_i**2 + z_bar_k**2))
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_Mj) * xp.cos(phi_bar_j)
    return t1_coef * t1


def Hr_zk_case232(xp, r_i, phi_bar_j, theta_M, z_bar_k):
    t1 = r_i / xp.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = -xp.cos(theta_M) * xp.sin(phi_bar_j)
    t2 = xp.atanh(t1)
    t2_coef = xp.cos(theta_M) * xp.sin(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2


def Hphi_ri_case232(xp, r_i, phi_j, phi_bar_j, phi_bar_M, phi_bar_Mj, theta_M, z_bar_k):
    t1 = xp.sin(theta_M) * z_bar_k / xp.sqrt(r_i**2 + z_bar_k**2)
    t2 = 1.0 / 4.0 * xp.cos(phi_bar_Mj + phi_bar_j)
    t3 = 1.0 / 2.0 * phi_j * xp.sin(phi_bar_M)
    return t1 * (-t2 + t3)


def Hphi_phij_case232(xp, r_i, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t1 = xp.atanh(z_bar_k / xp.sqrt(r_i**2 + z_bar_k**2))
    t1_coef = xp.sin(theta_M) * xp.sin(phi_bar_Mj) * xp.sin(phi_bar_j)
    return t1_coef * t1


def Hphi_zk_case232(xp, r_i, phi_bar_j, theta_M, z_bar_k):
    t1 = r_i / xp.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = -xp.cos(theta_M) * xp.cos(phi_bar_j)
    t2 = xp.atanh(t1)
    t2_coef = xp.cos(theta_M) * xp.cos(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2


def Hz_ri_case232(xp, r_i, phi_bar_Mj, theta_M, z_bar_k):
    t1 = r_i / xp.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_Mj)
    return t1_coef * t1


def Hz_phij_case232(xp, r_i, phi_bar_Mj, theta_M, z_bar_k):
    t1 = xp.atanh(r_i / xp.sqrt(r_i**2 + z_bar_k**2))
    t1_coef = xp.sin(theta_M) * xp.sin(phi_bar_Mj)
    return t1_coef * t1


def Hz_zk_case232(xp, r_i, phi_j, theta_M, z_bar_k):
    t1 = z_bar_k / xp.sqrt(r_i**2 + z_bar_k**2)
    t1_coef = -xp.cos(theta_M) * phi_j
    return t1_coef * t1


# 233 ##############


def Hr_phij_case233(xp, r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t1 = xp.atanh(z_bar_k / xp.sqrt(r**2 + z_bar_k**2))
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_Mj) * xp.cos(phi_bar_j)
    t2 = xp.atan(
        z_bar_k * xp.cos(phi_bar_j) / xp.sin(phi_bar_j) / xp.sqrt(r**2 + z_bar_k**2)
    )
    t2_coef = xp.sin(theta_M) * xp.sin(phi_bar_Mj) * xp.sin(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2


def Hr_zk_case233(xp, r, phi_bar_j, theta_M, z_bar_k):
    t = xp.sqrt(r**2 + z_bar_k**2)
    t1 = xp.sin(phi_bar_j)
    t1_coef = -xp.cos(theta_M)
    t2 = xp.log(-r * xp.cos(phi_bar_j) + t)
    t2_coef = xp.cos(theta_M) * xp.sin(phi_bar_j)
    t3 = xp.atan(r * xp.sin(phi_bar_j) / z_bar_k)
    t3_coef = xp.cos(theta_M) * z_bar_k / r
    t4 = arctan_k_tan_2(t / xp.abs(z_bar_k), 2.0 * phi_bar_j)
    t4_coef = -t3_coef

    def t5(sign):
        return arctan_k_tan_2(xp.abs(z_bar_k) / xp.abs(r + sign * t), phi_bar_j)

    t5_coef = t3_coef
    return (
        t1_coef * t1
        + t2_coef * t2
        + t3_coef * t3
        + t4_coef * t4
        + t5_coef * t5(1)
        + t5_coef * t5(-1)
    )


def Hphi_phij_case233(xp, r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t = xp.sqrt(r**2 + z_bar_k**2)
    t1 = xp.atan(z_bar_k * xp.cos(phi_bar_j) / (xp.sin(phi_bar_j) * t))
    t1_coef = xp.sin(theta_M) * xp.sin(phi_bar_Mj) * xp.cos(phi_bar_j)
    t2 = xp.atanh(z_bar_k / t)
    t2_coef = xp.sin(theta_M) * xp.sin(phi_bar_Mj) * xp.sin(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2


def Hphi_zk_case233(xp, r, phi_bar_j, theta_M, z_bar_k):
    t1 = xp.sqrt(r**2 + z_bar_k**2)
    t1_coef = xp.cos(theta_M) / r
    t2 = xp.atanh(r * xp.cos(phi_bar_j) / t1)
    t2_coef = -xp.cos(theta_M) * xp.cos(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2


def Hz_phij_case233(xp, r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t1 = xp.atanh(r * xp.cos(phi_bar_j) / xp.sqrt(r**2 + z_bar_k**2))
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_Mj)
    return t1_coef * t1


def Hz_zk_case233(xp, r, phi_bar_j, theta_M, z_bar_k):
    t1 = arctan_k_tan_2(xp.sqrt(r**2 + z_bar_k**2) / xp.abs(z_bar_k), 2.0 * phi_bar_j)
    t1_coef = xp.cos(theta_M) * xp.sign(z_bar_k)
    return t1_coef * t1


# 234 ##############


def Hr_ri_case234(xp, r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.sqrt(2.0 * r**2 * (1.0 - xp.cos(phi_bar_j)) + z_bar_k**2)
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_M) * z_bar_k / (2.0 * r**2)
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = (
        -xp.sin(theta_M)
        * xp.cos(phi_bar_M)
        * z_bar_k**2
        * xp.sign(z_bar_k)
        / (2.0 * r**2)
    )
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = (
        xp.sin(theta_M)
        * xp.cos(phi_bar_M)
        * xp.sign(z_bar_k)
        * (2.0 * r**2 + z_bar_k**2)
        / (2.0 * r**2)
    )
    return t1_coef * t1 + E_coef * E + F_coef * F


def Hr_phij_case234(xp, r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t1 = xp.atanh(
        z_bar_k / xp.sqrt(2.0 * r**2 * (1.0 - xp.cos(phi_bar_j)) + z_bar_k**2)
    )
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_Mj) * xp.cos(phi_bar_j)
    t2 = xp.atan(
        z_bar_k
        * (1.0 - xp.cos(phi_bar_j))
        / (
            xp.sin(phi_bar_j)
            * xp.sqrt(2.0 * r**2 * (1.0 - xp.cos(phi_bar_j)) + z_bar_k**2)
        )
    )
    t2_coef = -xp.sin(theta_M) * xp.sin(phi_bar_Mj) * xp.sin(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2


def Hr_zk_case234(xp, r, phi_bar_j, theta_M, z_bar_k):
    t1 = xp.sin(phi_bar_j)
    t1_coef = -xp.cos(theta_M)
    t2 = xp.log(
        r * (1.0 - xp.cos(phi_bar_j))
        + xp.sqrt(2.0 * r**2 * (1.0 - xp.cos(phi_bar_j)) + z_bar_k**2)
    )
    t2_coef = xp.cos(theta_M) * xp.sin(phi_bar_j)
    t3 = xp.atan(r * xp.sin(phi_bar_j) / z_bar_k)
    t3_coef = xp.cos(theta_M) * z_bar_k / r
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = xp.cos(theta_M) * xp.abs(z_bar_k) / r
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = -xp.cos(theta_M) * (2.0 * r**2 + z_bar_k**2) / (r * xp.abs(z_bar_k))
    t = xp.sqrt(r**2 + z_bar_k**2)

    def Pi1(sign):
        return el3_angle(
            phi_bar_j / 2.0, 2.0 * r / (r + sign * t), -4.0 * r**2 / z_bar_k**2
        )

    def Pi1_coef(sign):
        return (
            -xp.cos(theta_M)
            / (r * xp.sqrt((r**2 + z_bar_k**2) * z_bar_k**2))
            * (t - sign * r)
            * (r + sign * t) ** 2
        )

    def Pi2(sign):
        return el3_angle(
            arctan_k_tan_2(xp.sqrt((4.0 * r**2 + z_bar_k**2) / z_bar_k**2), phi_bar_j),
            1.0 - z_bar_k**4 / ((4.0 * r**2 + z_bar_k**2) * (r + sign * t) ** 2),
            4.0 * r**2 / (4.0 * r**2 + z_bar_k**2),
        )

    def Pi2_coef(sign):
        return (
            sign
            * xp.cos(theta_M)
            * z_bar_k**4
            / (
                r
                * xp.sqrt((r**2 + z_bar_k**2) * (4.0 * r**2 + z_bar_k**2))
                * (r + sign * t)
            )
        )

    return (
        t1_coef * t1
        + t2_coef * t2
        + t3_coef * t3
        + E_coef * E
        + F_coef * F
        + Pi1_coef(1) * Pi1(1)
        + Pi1_coef(-1) * Pi1(-1)
        + Pi2_coef(1) * Pi2(1)
        + Pi2_coef(-1) * Pi2(-1)
    )


def Hphi_ri_case234(xp, r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.sqrt(2.0 * r**2 * (1.0 - xp.cos(phi_bar_j)) + z_bar_k**2)
    t1_coef = -xp.sin(theta_M) * xp.cos(phi_bar_M) * z_bar_k / (2.0 * r**2)
    t2 = xp.atanh(
        z_bar_k / xp.sqrt(2.0 * r**2 * (1.0 - xp.cos(phi_bar_j)) + z_bar_k**2)
    )
    t2_coef = -xp.sin(theta_M) * xp.cos(phi_bar_M)
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = (
        xp.sin(theta_M)
        * xp.sin(phi_bar_M)
        * z_bar_k**2
        * xp.sign(z_bar_k)
        / (2.0 * r**2)
    )
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = (
        -xp.sin(theta_M)
        * xp.sin(phi_bar_M)
        * xp.sign(z_bar_k)
        * (4.0 * r**2 + z_bar_k**2)
        / (2.0 * r**2)
    )
    return t1_coef * t1 + t2_coef * t2 + E_coef * E + F_coef * F


def Hphi_phij_case234(xp, r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t = xp.sqrt(2.0 * r**2 * (1.0 - xp.cos(phi_bar_j)) + z_bar_k**2)
    t1 = xp.atan(z_bar_k * (1.0 - xp.cos(phi_bar_j)) / (xp.sin(phi_bar_j) * t))
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_Mj) * xp.cos(phi_bar_j)
    t2 = xp.atanh(z_bar_k / t)
    t2_coef = xp.sin(theta_M) * xp.sin(phi_bar_Mj) * xp.sin(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2


def Hphi_zk_case234(xp, r, phi_bar_j, theta_M, z_bar_k):
    t1 = xp.sqrt(2.0 * r**2 * (1.0 - xp.cos(phi_bar_j)) + z_bar_k**2)
    t1_coef = xp.cos(theta_M) / r
    t2 = xp.atanh(r * (1.0 - xp.cos(phi_bar_j)) / t1)
    t2_coef = xp.cos(theta_M) * xp.cos(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2


def Hz_ri_case234(xp, r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.sqrt(2.0 * r**2 * (1.0 - xp.cos(phi_bar_j)) + z_bar_k**2)
    t1_coef = xp.sin(theta_M) * xp.sin(phi_bar_M) / r
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r**2 / z_bar_k**2)
    E_coef = xp.sin(theta_M) * xp.cos(phi_bar_M) * xp.abs(z_bar_k) / r
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r**2 / z_bar_k**2)
    F_coef = (
        -xp.sin(theta_M)
        * xp.cos(phi_bar_M)
        * (2.0 * r**2 + z_bar_k**2)
        / (r * xp.abs(z_bar_k))
    )
    return t1_coef * t1 + E_coef * E + F_coef * F


def Hz_phij_case234(xp, r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t1 = xp.atanh(
        r
        * (1.0 - xp.cos(phi_bar_j))
        / xp.sqrt(2.0 * r**2 * (1.0 - xp.cos(phi_bar_j)) + z_bar_k**2)
    )
    t1_coef = xp.sin(theta_M) * xp.sin(phi_bar_Mj)
    return t1_coef * t1


def Hz_zk_case234(xp, r, phi_bar_j, theta_M, z_bar_k):
    t = xp.sqrt(r**2 + z_bar_k**2)

    def Pi(sign):
        return el3_angle(
            phi_bar_j / 2.0, 2.0 * r / (r + sign * t), -4.0 * r**2 / z_bar_k**2
        )

    Pi_coef = xp.cos(theta_M) * xp.sign(z_bar_k)
    return Pi_coef * Pi(1) + Pi_coef * Pi(-1)


# 235 ##############


def Hr_ri_case235(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * xp.cos(phi_bar_j) + z_bar_k**2)
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_M) * z_bar_k / (2.0 * r**2)
    t2 = xp.atanh(
        z_bar_k
        / xp.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * xp.cos(phi_bar_j) + z_bar_k**2)
    )
    t2_coef = xp.sin(theta_M) * xp.sin(phi_bar_M) / 2.0 * (1.0 - r_i**2 / r**2)
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = (
        -xp.sin(theta_M)
        * xp.cos(phi_bar_M)
        * z_bar_k
        * xp.sqrt(r_bar_i**2 + z_bar_k**2)
        / (2.0 * r**2)
    )
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = (
        xp.sin(theta_M)
        * xp.cos(phi_bar_M)
        * z_bar_k
        * (2.0 * r_i**2 + z_bar_k**2)
        / (2.0 * r**2 * xp.sqrt(r_bar_i**2 + z_bar_k**2))
    )
    Pi = el3_angle(
        phi_bar_j / 2.0,
        -4.0 * r * r_i / r_bar_i**2,
        -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2),
    )
    Pi_coef = (
        xp.sin(theta_M)
        * xp.cos(phi_bar_M)
        * z_bar_k
        * (r**2 + r_i**2)
        * (r + r_i)
        / (2.0 * r**2 * r_bar_i * xp.sqrt(r_bar_i**2 + z_bar_k**2))
    )
    return t1_coef * t1 + t2_coef * t2 + E_coef * E + F_coef * F + Pi_coef * Pi


def Hr_phij_case235(xp, r, r_i, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t1 = xp.atanh(
        z_bar_k
        / xp.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * xp.cos(phi_bar_j) + z_bar_k**2)
    )
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_Mj) * xp.cos(phi_bar_j)
    t2 = xp.atan(
        z_bar_k
        * (r * xp.cos(phi_bar_j) - r_i)
        / (
            r
            * xp.sin(phi_bar_j)
            * xp.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * xp.cos(phi_bar_j) + z_bar_k**2)
        )
    )
    t2_coef = xp.sin(theta_M) * xp.sin(phi_bar_Mj) * xp.sin(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2


def Hr_zk_case235(xp, r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k):
    t1 = xp.sin(phi_bar_j)
    t1_coef = -xp.cos(theta_M)
    t2 = xp.log(
        r_i
        - r * xp.cos(phi_bar_j)
        + xp.sqrt(r_i**2 + r**2 - 2.0 * r_i * r * xp.cos(phi_bar_j) + z_bar_k**2)
    )
    t2_coef = xp.cos(theta_M) * xp.sin(phi_bar_j)
    t3 = xp.atan(r * xp.sin(phi_bar_j) / z_bar_k)
    t3_coef = xp.cos(theta_M) * z_bar_k / r
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = xp.cos(theta_M) * xp.sqrt(r_bar_i**2 + z_bar_k**2) / r
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = (
        -xp.cos(theta_M)
        * (r**2 + r_i**2 + z_bar_k**2)
        / (r * xp.sqrt(r_bar_i**2 + z_bar_k**2))
    )
    t = xp.sqrt(r**2 + z_bar_k**2)

    def Pi1(sign):
        return el3_angle(
            phi_bar_j / 2.0,
            2.0 * r / (r + sign * t),
            -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2),
        )

    def Pi1_coef(sign):
        return (
            -xp.cos(theta_M)
            / (r * xp.sqrt((r**2 + z_bar_k**2) * (r_bar_i**2 + z_bar_k**2)))
            * (t - sign * r)
            * (r_i + sign * t) ** 2
        )

    def Pi2(sign):
        return el3_angle(
            arctan_k_tan_2(
                xp.sqrt(((r_i + r) ** 2 + z_bar_k**2) / (r_bar_i**2 + z_bar_k**2)),
                phi_bar_j,
            ),
            1.0
            - z_bar_k**2
            * (r_bar_i**2 + z_bar_k**2)
            / (((r + r_i) ** 2 + z_bar_k**2) * (r + sign * t) ** 2),
            4.0 * r * r_i / ((r + r_i) ** 2 + z_bar_k**2),
        )

    def Pi2_coef(sign):
        return (
            sign
            * xp.cos(theta_M)
            * z_bar_k**2
            * (r_bar_i**2 + z_bar_k**2)
            / (
                r
                * xp.sqrt((r**2 + z_bar_k**2) * ((r + r_i) ** 2 + z_bar_k**2))
                * (r + sign * t)
            )
        )

    return (
        t1_coef * t1
        + t2_coef * t2
        + t3_coef * t3
        + E_coef * E
        + F_coef * F
        + Pi1_coef(1) * Pi1(1)
        + Pi1_coef(-1) * Pi1(-1)
        + Pi2_coef(1) * Pi2(1)
        + Pi2_coef(-1) * Pi2(-1)
    )


def Hphi_ri_case235(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t1 = xp.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * xp.cos(phi_bar_j) + z_bar_k**2)
    t1_coef = -xp.sin(theta_M) * xp.cos(phi_bar_M) * z_bar_k / (2.0 * r**2)
    t2 = xp.atanh(
        z_bar_k
        / xp.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * xp.cos(phi_bar_j) + z_bar_k**2)
    )
    t2_coef = -xp.sin(theta_M) * xp.cos(phi_bar_M) * (r**2 + r_i**2) / (2.0 * r**2)
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    E_coef = (
        xp.sin(theta_M)
        * xp.sin(phi_bar_M)
        * z_bar_k
        * xp.sqrt(r_bar_i**2 + z_bar_k**2)
        / (2.0 * r**2)
    )
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2))
    F_coef = (
        -xp.sin(theta_M)
        * xp.sin(phi_bar_M)
        * z_bar_k
        * (2.0 * r**2 + 2.0 * r_i**2 + z_bar_k**2)
        / (2.0 * r**2 * xp.sqrt(r_bar_i**2 + z_bar_k**2))
    )
    Pi = el3_angle(
        phi_bar_j / 2.0,
        -4.0 * r * r_i / r_bar_i**2,
        -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2),
    )
    Pi_coef = (
        xp.sin(theta_M)
        * xp.sin(phi_bar_M)
        * z_bar_k
        * (r + r_i) ** 2
        / (2.0 * r**2 * xp.sqrt(r_bar_i**2 + z_bar_k**2))
    )
    return t1_coef * t1 + t2_coef * t2 + E_coef * E + F_coef * F + Pi_coef * Pi


def Hphi_phij_case235(xp, r, r_i, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t = xp.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * xp.cos(phi_bar_j) + z_bar_k**2)
    t1 = xp.atan(z_bar_k * (r * xp.cos(phi_bar_j) - r_i) / (r * xp.sin(phi_bar_j) * t))
    t1_coef = xp.sin(theta_M) * xp.sin(phi_bar_Mj) * xp.cos(phi_bar_j)
    t2 = xp.atanh(z_bar_k / t)
    t2_coef = xp.sin(theta_M) * xp.sin(phi_bar_Mj) * xp.sin(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2


def Hphi_zk_case235(xp, r, r_i, phi_bar_j, theta_M, z_bar_k):
    t1 = xp.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * xp.cos(phi_bar_j) + z_bar_k**2)
    t1_coef = xp.cos(theta_M) / r
    t2 = xp.atanh((r * xp.cos(phi_bar_j) - r_i) / t1)
    t2_coef = -xp.cos(theta_M) * xp.cos(phi_bar_j)
    return t1_coef * t1 + t2_coef * t2


def Hz_ri_case235(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    t = r_bar_i**2 + z_bar_k**2
    t1 = xp.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * xp.cos(phi_bar_j) + z_bar_k**2)
    t1_coef = xp.sin(theta_M) * xp.sin(phi_bar_M) / r
    E = ellipeinc(phi_bar_j / 2.0, -4.0 * r * r_i / t)
    E_coef = xp.sin(theta_M) * xp.cos(phi_bar_M) * xp.sqrt(t) / r
    F = ellipkinc(phi_bar_j / 2.0, -4.0 * r * r_i / t)
    F_coef = (
        -xp.sin(theta_M)
        * xp.cos(phi_bar_M)
        * (r**2 + r_i**2 + z_bar_k**2)
        / (r * xp.sqrt(t))
    )
    return t1_coef * t1 + E_coef * E + F_coef * F


def Hz_phij_case235(xp, r, r_i, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    t1 = xp.atanh(
        (r * xp.cos(phi_bar_j) - r_i)
        / xp.sqrt(r**2 + r_i**2 - 2.0 * r * r_i * xp.cos(phi_bar_j) + z_bar_k**2)
    )
    t1_coef = -xp.sin(theta_M) * xp.sin(phi_bar_Mj)
    return t1_coef * t1


def Hz_zk_case235(xp, r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k):
    t = xp.sqrt(r**2 + z_bar_k**2)

    def Pi(sign):
        return el3_angle(
            phi_bar_j / 2.0,
            2.0 * r / (r + sign * t),
            -4.0 * r * r_i / (r_bar_i**2 + z_bar_k**2),
        )

    def Pi_coef(sign):
        return (
            xp.cos(theta_M)
            * z_bar_k
            * (r_i + sign * t)
            / (xp.sqrt(r_bar_i**2 + z_bar_k**2) * (r + sign * t))
        )

    return Pi_coef(1) * Pi(1) + Pi_coef(-1) * Pi(-1)


####################
####################
####################
# calculation of all field components for each case
# especially these function show, which inputs are needed for the calculation
# full vectorization for all cases could be implemented here
# input: ndarray, shape (n,)
# out: ndarray, shape (n,3,3) # (n)vector, (3)r_phi_z, (3)face


def case112(xp, r_i, phi_bar_M, theta_M):
    results = xp.zeros(((r_i.shape[0]), 3, 3))
    results = xpx.at(results)[:, 1, 2].set(Hphi_zk_case112(xp, r_i, theta_M))
    results = xpx.at(results)[:, 2, 0].set(Hz_ri_case112(xp, phi_bar_M, theta_M))
    results = xpx.at(results)[:, 2, 1].set(Hz_phij_case112(xp, r_i, phi_bar_M, theta_M))
    return results


def case113(xp, r, phi_bar_M, theta_M):
    results = xp.zeros(((r.shape[0]), 3, 3))
    results = xpx.at(results)[:, 1, 2].set(Hphi_zk_case113(xp, r, theta_M))
    results = xpx.at(results)[:, 2, 1].set(Hz_phij_case113(xp, r, phi_bar_M, theta_M))
    return results


def case115(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M):
    results = xp.zeros(((r.shape[0]), 3, 3))
    results = xpx.at(results)[:, 0, 2].set(
        Hr_zk_case115(xp, r, r_i, r_bar_i, phi_bar_j, theta_M)
    )
    results = xpx.at(results)[:, 1, 2].set(
        Hphi_zk_case115(xp, r, r_i, r_bar_i, theta_M)
    )
    results = xpx.at(results)[:, 2, 0].set(
        Hz_ri_case115(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M)
    )
    results = xpx.at(results)[:, 2, 1].set(
        Hz_phij_case115(xp, r_bar_i, phi_bar_M, theta_M)
    )
    return results


def case122(xp, r_i, phi_bar_M, theta_M):
    results = xp.zeros(((r_i.shape[0]), 3, 3))
    results = xpx.at(results)[:, 1, 2].set(Hphi_zk_case122(xp, r_i, theta_M))
    results = xpx.at(results)[:, 2, 0].set(Hz_ri_case122(xp, phi_bar_M, theta_M))
    results = xpx.at(results)[:, 2, 1].set(Hz_phij_case122(xp, r_i, phi_bar_M, theta_M))
    return results


def case123(xp, r, phi_bar_M, theta_M):
    results = xp.zeros(((r.shape[0]), 3, 3))
    results = xpx.at(results)[:, 1, 2].set(Hphi_zk_case123(xp, r, theta_M))
    results = xpx.at(results)[:, 2, 1].set(Hz_phij_case123(xp, r, phi_bar_M, theta_M))
    return results


def case124(xp, r, phi_bar_M, theta_M):
    results = xp.zeros(((r.shape[0]), 3, 3))
    results = xpx.at(results)[:, 1, 2].set(Hphi_zk_case124(xp, r, theta_M))
    results = xpx.at(results)[:, 2, 0].set(Hz_ri_case124(xp, phi_bar_M, theta_M))
    results = xpx.at(results)[:, 2, 1].set(Hz_phij_case124(xp, r, phi_bar_M, theta_M))
    return results


def case125(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M):
    results = xp.zeros(((r.shape[0]), 3, 3))
    results = xpx.at(results)[:, 0, 2].set(
        Hr_zk_case125(xp, r, r_i, r_bar_i, phi_bar_j, theta_M)
    )
    results = xpx.at(results)[:, 1, 2].set(Hphi_zk_case125(xp, r, r_i, theta_M))
    results = xpx.at(results)[:, 2, 0].set(
        Hz_ri_case125(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M)
    )
    results = xpx.at(results)[:, 2, 1].set(
        Hz_phij_case125(xp, r, r_i, phi_bar_M, theta_M)
    )
    return results


def case132(xp, r, r_i, phi_bar_j, phi_bar_Mj, theta_M):
    results = xp.zeros(((r.shape[0]), 3, 3))
    results = xpx.at(results)[:, 0, 2].set(Hr_zk_case132(xp, r_i, phi_bar_j, theta_M))
    results = xpx.at(results)[:, 1, 2].set(Hphi_zk_case132(xp, r_i, phi_bar_j, theta_M))
    results = xpx.at(results)[:, 2, 0].set(Hz_ri_case132(xp, phi_bar_Mj, theta_M))
    results = xpx.at(results)[:, 2, 1].set(
        Hz_phij_case132(xp, r_i, phi_bar_Mj, theta_M)
    )
    return results


def case133(xp, r, phi_bar_j, phi_bar_Mj, theta_M):
    results = xp.zeros(((r.shape[0]), 3, 3))
    results = xpx.at(results)[:, 0, 2].set(Hr_zk_case133(xp, r, phi_bar_j, theta_M))
    results = xpx.at(results)[:, 1, 2].set(Hphi_zk_case133(xp, phi_bar_j, theta_M))
    results = xpx.at(results)[:, 2, 1].set(
        Hz_phij_case133(xp, phi_bar_j, phi_bar_Mj, theta_M)
    )
    return results


def case134(xp, r, phi_bar_j, phi_bar_M, phi_bar_Mj, theta_M):
    results = xp.zeros(((r.shape[0]), 3, 3))
    results = xpx.at(results)[:, 0, 2].set(Hr_zk_case134(xp, r, phi_bar_j, theta_M))
    results = xpx.at(results)[:, 1, 2].set(Hphi_zk_case134(xp, phi_bar_j, theta_M))
    results = xpx.at(results)[:, 2, 0].set(
        Hz_ri_case134(xp, phi_bar_j, phi_bar_M, theta_M)
    )
    results = xpx.at(results)[:, 2, 1].set(
        Hz_phij_case134(xp, phi_bar_j, phi_bar_Mj, theta_M)
    )
    return results


def case135(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, phi_bar_Mj, theta_M):
    results = xp.zeros(((r.shape[0]), 3, 3))
    results = xpx.at(results)[:, 0, 2].set(
        Hr_zk_case135(xp, r, r_i, r_bar_i, phi_bar_j, theta_M)
    )
    results = xpx.at(results)[:, 1, 2].set(
        Hphi_zk_case135(xp, r, r_i, phi_bar_j, theta_M)
    )
    results = xpx.at(results)[:, 2, 0].set(
        Hz_ri_case135(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M)
    )
    results = xpx.at(results)[:, 2, 1].set(
        Hz_phij_case135(xp, r, r_i, phi_bar_j, phi_bar_Mj, theta_M)
    )
    return results


def case211(xp, phi_j, phi_bar_M, theta_M, z_bar_k):
    results = xp.zeros(((phi_j.shape[0]), 3, 3))
    results = xpx.at(results)[:, 0, 1].set(
        Hr_phij_case211(xp, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 2].set(Hz_zk_case211(xp, phi_j, theta_M, z_bar_k))
    return results


def case212(xp, r_i, phi_j, phi_bar_M, theta_M, z_bar_k):
    results = xp.zeros(((r_i.shape[0]), 3, 3))
    results = xpx.at(results)[:, 0, 0].set(
        Hr_ri_case212(xp, r_i, phi_j, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 0, 1].set(
        Hr_phij_case212(xp, r_i, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 0].set(
        Hphi_ri_case212(xp, r_i, phi_j, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 2].set(Hphi_zk_case212(xp, r_i, theta_M, z_bar_k))
    results = xpx.at(results)[:, 2, 0].set(
        Hz_ri_case212(xp, r_i, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 1].set(
        Hz_phij_case212(xp, r_i, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 2].set(
        Hz_zk_case212(xp, r_i, phi_j, theta_M, z_bar_k)
    )
    return results


def case213(xp, r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    results = xp.zeros(((r.shape[0]), 3, 3))
    results = xpx.at(results)[:, 0, 1].set(
        Hr_phij_case213(xp, r, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 2].set(Hphi_zk_case213(xp, r, theta_M, z_bar_k))
    results = xpx.at(results)[:, 2, 1].set(
        Hz_phij_case213(xp, r, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 2].set(
        Hz_zk_case213(xp, phi_bar_j, theta_M, z_bar_k)
    )
    return results


def case214(xp, r, phi_j, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    results = xp.zeros(((r.shape[0]), 3, 3))
    results = xpx.at(results)[:, 0, 0].set(
        Hr_ri_case214(xp, r, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 0, 1].set(
        Hr_phij_case214(xp, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 0, 2].set(
        Hr_zk_case214(xp, r, phi_bar_j, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 0].set(
        Hphi_ri_case214(xp, r, phi_j, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 2].set(Hphi_zk_case214(xp, r, theta_M, z_bar_k))
    results = xpx.at(results)[:, 2, 0].set(
        Hz_ri_case214(xp, r, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 2].set(
        Hz_zk_case214(xp, r, phi_bar_j, theta_M, z_bar_k)
    )
    return results


def case215(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    results = xp.zeros(((r.shape[0]), 3, 3))
    results = xpx.at(results)[:, 0, 0].set(
        Hr_ri_case215(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 0, 1].set(
        Hr_phij_case215(xp, r_bar_i, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 0, 2].set(
        Hr_zk_case215(xp, r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 0].set(
        Hphi_ri_case215(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 2].set(
        Hphi_zk_case215(xp, r, r_bar_i, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 0].set(
        Hz_ri_case215(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 1].set(
        Hz_phij_case215(xp, r_bar_i, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 2].set(
        Hz_zk_case215(xp, r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k)
    )
    return results


def case221(xp, phi_j, phi_bar_M, theta_M, z_bar_k):
    results = xp.zeros(((phi_j.shape[0]), 3, 3))
    results = xpx.at(results)[:, 0, 1].set(
        Hr_phij_case221(xp, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 2].set(Hz_zk_case221(xp, phi_j, theta_M, z_bar_k))
    return results


def case222(xp, r_i, phi_j, phi_bar_M, theta_M, z_bar_k):
    results = xp.zeros(((r_i.shape[0]), 3, 3))
    results = xpx.at(results)[:, 0, 0].set(
        Hr_ri_case222(xp, r_i, phi_j, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 0, 1].set(
        Hr_phij_case222(xp, r_i, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 0].set(
        Hphi_ri_case222(xp, r_i, phi_j, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 2].set(Hphi_zk_case222(xp, r_i, theta_M, z_bar_k))
    results = xpx.at(results)[:, 2, 0].set(
        Hz_ri_case222(xp, r_i, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 1].set(
        Hz_phij_case222(xp, r_i, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 2].set(
        Hz_zk_case222(xp, r_i, phi_j, theta_M, z_bar_k)
    )
    return results


def case223(xp, r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    results = xp.zeros(((r.shape[0]), 3, 3))
    results = xpx.at(results)[:, 0, 1].set(
        Hr_phij_case223(xp, r, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 2].set(Hphi_zk_case223(xp, r, theta_M, z_bar_k))
    results = xpx.at(results)[:, 2, 1].set(
        Hz_phij_case223(xp, r, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 2].set(
        Hz_zk_case223(xp, r, phi_bar_j, theta_M, z_bar_k)
    )
    return results


def case224(xp, r, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    results = xp.zeros(((r.shape[0]), 3, 3))
    results = xpx.at(results)[:, 0, 0].set(
        Hr_ri_case224(xp, r, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 0, 1].set(
        Hr_phij_case224(xp, r, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 0, 2].set(
        Hr_zk_case224(xp, r, phi_bar_j, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 0].set(
        Hphi_ri_case224(xp, r, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 2].set(Hphi_zk_case224(xp, r, theta_M, z_bar_k))
    results = xpx.at(results)[:, 2, 0].set(
        Hz_ri_case224(xp, r, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 1].set(
        Hz_phij_case224(xp, r, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 2].set(
        Hz_zk_case224(xp, r, phi_bar_j, theta_M, z_bar_k)
    )
    return results


def case225(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k):
    results = xp.zeros(((r.shape[0]), 3, 3))
    results = xpx.at(results)[:, 0, 0].set(
        Hr_ri_case225(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 0, 1].set(
        Hr_phij_case225(xp, r, r_i, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 0, 2].set(
        Hr_zk_case225(xp, r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 0].set(
        Hphi_ri_case225(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 2].set(
        Hphi_zk_case225(xp, r, r_i, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 0].set(
        Hz_ri_case225(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 1].set(
        Hz_phij_case225(xp, r, r_i, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 2].set(
        Hz_zk_case225(xp, r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k)
    )
    return results


def case231(xp, phi_j, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    results = xp.zeros(((phi_j.shape[0]), 3, 3))
    results = xpx.at(results)[:, 0, 1].set(
        Hr_phij_case231(xp, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 1].set(
        Hphi_phij_case231(xp, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 2].set(Hz_zk_case231(xp, phi_j, theta_M, z_bar_k))
    return results


def case232(xp, r_i, phi_j, phi_bar_j, phi_bar_M, phi_bar_Mj, theta_M, z_bar_k):
    results = xp.zeros(((r_i.shape[0]), 3, 3))
    results = xpx.at(results)[:, 0, 0].set(
        Hr_ri_case232(
            xp, r_i, phi_j, phi_bar_j, phi_bar_M, phi_bar_Mj, theta_M, z_bar_k
        )
    )
    results = xpx.at(results)[:, 0, 1].set(
        Hr_phij_case232(xp, r_i, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 0, 2].set(
        Hr_zk_case232(xp, r_i, phi_bar_j, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 0].set(
        Hphi_ri_case232(
            xp, r_i, phi_j, phi_bar_j, phi_bar_M, phi_bar_Mj, theta_M, z_bar_k
        )
    )
    results = xpx.at(results)[:, 1, 1].set(
        Hphi_phij_case232(xp, r_i, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 2].set(
        Hphi_zk_case232(xp, r_i, phi_bar_j, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 0].set(
        Hz_ri_case232(xp, r_i, phi_bar_Mj, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 1].set(
        Hz_phij_case232(xp, r_i, phi_bar_Mj, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 2].set(
        Hz_zk_case232(xp, r_i, phi_j, theta_M, z_bar_k)
    )
    return results


def case233(xp, r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k):
    results = xp.zeros(((r.shape[0]), 3, 3))
    results = xpx.at(results)[:, 0, 1].set(
        Hr_phij_case233(xp, r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 0, 2].set(
        Hr_zk_case233(xp, r, phi_bar_j, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 1].set(
        Hphi_phij_case233(xp, r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 2].set(
        Hphi_zk_case233(xp, r, phi_bar_j, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 1].set(
        Hz_phij_case233(xp, r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 2].set(
        Hz_zk_case233(xp, r, phi_bar_j, theta_M, z_bar_k)
    )
    return results


def case234(xp, r, phi_bar_j, phi_bar_M, phi_bar_Mj, theta_M, z_bar_k):
    results = xp.zeros(((r.shape[0]), 3, 3))
    results = xpx.at(results)[:, 0, 0].set(
        Hr_ri_case234(xp, r, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 0, 1].set(
        Hr_phij_case234(xp, r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 0, 2].set(
        Hr_zk_case234(xp, r, phi_bar_j, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 0].set(
        Hphi_ri_case234(xp, r, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 1].set(
        Hphi_phij_case234(xp, r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 2].set(
        Hphi_zk_case234(xp, r, phi_bar_j, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 0].set(
        Hz_ri_case234(xp, r, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 1].set(
        Hz_phij_case234(xp, r, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 2].set(
        Hz_zk_case234(xp, r, phi_bar_j, theta_M, z_bar_k)
    )
    return results


def case235(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, phi_bar_Mj, theta_M, z_bar_k):
    results = xp.zeros(((r.shape[0]), 3, 3))
    results = xpx.at(results)[:, 0, 0].set(
        Hr_ri_case235(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 0, 1].set(
        Hr_phij_case235(xp, r, r_i, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 0, 2].set(
        Hr_zk_case235(xp, r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 0].set(
        Hphi_ri_case235(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 1].set(
        Hphi_phij_case235(xp, r, r_i, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 1, 2].set(
        Hphi_zk_case235(xp, r, r_i, phi_bar_j, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 0].set(
        Hz_ri_case235(xp, r, r_i, r_bar_i, phi_bar_j, phi_bar_M, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 1].set(
        Hz_phij_case235(xp, r, r_i, phi_bar_j, phi_bar_Mj, theta_M, z_bar_k)
    )
    results = xpx.at(results)[:, 2, 2].set(
        Hz_zk_case235(xp, r, r_i, r_bar_i, phi_bar_j, theta_M, z_bar_k)
    )
    return results


# CORE
def magnet_cylinder_segment_Hfield(
    observers: np.ndarray,
    dimensions: np.ndarray,
    magnetizations: np.ndarray,
) -> np.ndarray:
    """Magnetic field of homogeneously magnetized cylinder ring segments
    in Cartesian Coordinates.

    The cylinder axes coincide with the z-axis of the Cylindrical CS and the
    geometric center of the cylinder lies in the origin. The result is
    proportional to the magnetization magnitude, and independent of the
    length units used for dimensions and observers.

    Parameters
    ----------
    observers : ndarray, shape (n,3)
        Observer positions (r,phi,z) in Cylinder coordinates, where phi is
        given in rad.

    dimensions: ndarray, shape (n,6)
        Segment dimension [(r1,r2,phi1,phi2,z1,z2), ...]. r1 and r2 are
        inner and outer radii. phi1 and phi2 are azimuth section angles
        in rad. z1 and z2 are z-values of bottom and top.

    magnetizations: ndarray, shape (n,3)
        Magnetization vectors [(M, phi, th), ...] in spherical CS. M is the
        magnitude of magnetization, phi and th are azimuth and polar angles
        in rad.

    Returns
    -------
    H-field: ndarray, shape (n,3)
        H-field generated by Cylinder Segments at observer positions.

    Examples
    --------
    >>> import numpy as np
    >>> import magpylib as magpy
    >>> B = magpy.core.magnet_cylinder_segment_Hfield(
    ...     observers=xp.array([(1,1,2), (0,0,0)]),
    ...     dimensions=xp.array([(1,2,.1,.2,-1,1), (1,2,.3,.9,0,1)]),
    ...     magnetizations=xp.array([(1e7,.1,.2), (1e6,1.1,2.2)]),
    ... )
    >>> with xp.printoptions(precision=3):
    ...     print(B)
    [[-1948.144 32319.944 17616.886]
     [14167.65   1419.941 17921.646]]

    Notes
    -----
    Implementation based on F.Slanovc, Journal of Magnetism and Magnetic
    Materials, Volume 559, 1 October 2022, 169482
    """

    xp = array_namespace(observers, dimensions, magnetizations)
    observers, dimensions, magnetizations = xp_promote(
        observers, dimensions, magnetizations, force_floating=True, xp=xp
    )
    # tile inputs into 8-stacks (boundary cases)
    rphiz = xp.repeat(observers, 8, axis=0).T
    r, phi, z = (rphiz[i, ...] for i in range(3))
    r_i = xp.repeat(dimensions[:, :2], 4)
    phi_j = xp.repeat(xp.tile(dimensions[:, 2:4], (2,)), 2)
    z_k = xp.reshape(xp.tile(dimensions[:, 4:6], (4,)), (-1,))
    rphiz_M = xp.repeat(magnetizations, 8, axis=0).T
    _, phi_M, theta_M = (rphiz_M[i, ...] for i in range(3))

    # initialize results array with nan
    result = xp.empty(((r.shape[0]), 3, 3))
    result = xpx.at(result)[:, ...].set(xp.nan)

    # cases to evaluate
    cases = determine_cases(r, phi, z, r_i, phi_j, z_k)

    # list of all possible cases - excluding the nan-cases 111, 114, 121, 131
    case_id = xp.asarray(
        [
            112,
            113,
            115,
            122,
            123,
            124,
            125,
            132,
            133,
            134,
            135,
            211,
            212,
            213,
            214,
            215,
            221,
            222,
            223,
            224,
            225,
            231,
            232,
            233,
            234,
            235,
        ]
    )

    # corresponding case evaluation functions
    case_fkt = [
        case112,
        case113,
        case115,
        case122,
        case123,
        case124,
        case125,
        case132,
        case133,
        case134,
        case135,
        case211,
        case212,
        case213,
        case214,
        case215,
        case221,
        case222,
        case223,
        case224,
        case225,
        case231,
        case232,
        case233,
        case234,
        case235,
    ]

    # required case function arguments
    r_bar_i = r - r_i
    phi_bar_j = phi - phi_j
    phi_bar_M = phi_M - phi
    phi_bar_Mj = phi_M - phi_j
    z_bar_k = z - z_k
    #          0   1      2         3          4          5          6        7       8
    allargs = [
        r,
        r_i,
        r_bar_i,
        phi_bar_j,
        phi_bar_M,
        phi_bar_Mj,
        theta_M,
        z_bar_k,
        phi_j,
    ]
    case_args = [
        (1, 4, 6),
        (0, 4, 6),
        (0, 1, 2, 3, 4, 6),
        (1, 4, 6),
        (0, 4, 6),
        (0, 4, 6),
        (0, 1, 2, 3, 4, 6),
        (0, 1, 3, 5, 6),
        (0, 3, 5, 6),
        (0, 3, 4, 5, 6),
        (0, 1, 2, 3, 4, 5, 6),
        (8, 4, 6, 7),
        (1, 8, 4, 6, 7),
        (0, 3, 4, 6, 7),
        (0, 8, 3, 4, 6, 7),
        (0, 1, 2, 3, 4, 6, 7),
        (8, 4, 6, 7),
        (1, 8, 4, 6, 7),
        (0, 3, 4, 6, 7),
        (0, 3, 4, 6, 7),
        (0, 1, 2, 3, 4, 6, 7),
        (8, 3, 5, 6, 7),
        (1, 8, 3, 4, 5, 6, 7),
        (0, 3, 5, 6, 7),
        (0, 3, 4, 5, 6, 7),
        (0, 1, 2, 3, 4, 5, 6, 7),
    ]

    # calling case functions with respective masked arguments
    for cid, cfkt, cargs in zip(case_id, case_fkt, case_args, strict=False):
        mask = cases == cid
        if any(mask):
            result = xpx.at(result)[mask].set(
                cfkt(xp, *[allargs[aid][mask] for aid in cargs])
            )

    # sum up contributions from different boundary cases (ax1) and different face types (ax3)
    result = xp.reshape(result, (-1, 8, 3, 3))
    result0 = result[:, 1, ...] - result[:, 0, ...]
    result1 = result[:, 2, ...] - result[:, 3, ...]
    result2 = result[:, 4, ...] - result[:, 5, ...]
    result3 = result[:, 7, ...] - result[:, 6, ...]
    result = xp.sum(result0 + result1 + result2 + result3, axis=2)

    # multiply with magnetization amplitude
    result = result.T * magnetizations[:, 0] * 1e-7 / MU0

    return result.T


def BHJM_cylinder_segment_internal(
    field: str,
    observers: np.ndarray,
    polarization: np.ndarray,
    dimension: np.ndarray,
) -> np.ndarray:
    """
    internal version of BHJM_cylinder_segment used for object oriented interface.

    Falls back to magnet_cylinder_field whenever the section angles describe the full
    360° cylinder.
    """
    xp = array_namespace(observers, polarization, dimension)

    BHfinal = xp.zeros_like(observers, dtype=float)

    r1, r2, h, phi1, phi2 = dimension.T

    # case1: segment
    mask1 = (phi2 - phi1) < 360

    BHfinal[mask1] = BHJM_cylinder_segment(
        field=field,
        observers=observers[mask1],
        polarization=polarization[mask1],
        dimension=dimension[mask1],
    )

    # case2: full cylinder
    mask1x = ~mask1
    BHfinal[mask1x] = BHJM_magnet_cylinder(
        field=field,
        observers=observers[mask1x],
        polarization=polarization[mask1x],
        dimension=xp.c_[2 * r2[mask1x], h[mask1x]],
    )

    # case2a: hollow cylinder <- should be vectorized together with above
    mask2 = (r1 != 0) & mask1x
    BHfinal[mask2] -= BHJM_magnet_cylinder(
        field=field,
        observers=observers[mask2],
        polarization=polarization[mask2],
        dimension=xp.c_[2 * r1[mask2], h[mask2]],
    )

    return BHfinal


def BHJM_cylinder_segment(
    field: str,
    observers: np.ndarray,
    dimension: np.ndarray,
    polarization: np.ndarray,
) -> np.ndarray:
    """
    - translate cylinder segment field to BHJM
    - special cases catching
    """
    check_field_input(field)

    BHJM = polarization.astype(float)

    r1, r2, h, phi1, phi2 = dimension.T
    r1 = abs(r1)
    r2 = abs(r2)
    h = abs(h)
    z1, z2 = -h / 2, h / 2

    # transform dim deg->rad
    phi1 = phi1 / 180 * np.pi
    phi2 = phi2 / 180 * np.pi
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T

    # transform obs_pos to Cy CS --------------------------------------------
    x, y, z = observers.T
    r, phi = np.sqrt(x**2 + y**2), np.arctan2(y, x)
    pos_obs_cy = np.concatenate(((r,), (phi,), (z,)), axis=0).T

    # determine when points lie inside and on surface of magnet -------------

    # mask_inside = None
    # if in_out == "auto":
    # phip1 in [-2pi,0], phio2 in [0,2pi]
    phio1 = phi
    phio2 = phi - np.sign(phi) * 2 * np.pi

    # phi=phi1, phi=phi2
    mask_phi1 = close(phio1, phi1) | close(phio2, phi1)
    mask_phi2 = close(phio1, phi2) | close(phio2, phi2)

    # r, phi ,z lies in-between, avoid numerical fluctuations (e.g. due to rotations) by including 1e-14
    mask_r_in = (r1 - 1e-14 < r) & (r < r2 + 1e-14)
    mask_phi_in = (np.sign(phio1 - phi1) != np.sign(phio1 - phi2)) | (
        np.sign(phio2 - phi1) != np.sign(phio2 - phi2)
    )
    mask_z_in = (z1 - 1e-14 < z) & (z < z2 + 1e-14)

    # on surface
    mask_surf_z = (
        (close(z, z1) | close(z, z2)) & mask_phi_in & mask_r_in
    )  # top / bottom
    mask_surf_r = (close(r, r1) | close(r, r2)) & mask_phi_in & mask_z_in  # in / out
    mask_surf_phi = (mask_phi1 | mask_phi2) & mask_r_in & mask_z_in  # in / out
    mask_not_on_surf = ~(mask_surf_z | mask_surf_r | mask_surf_phi)

    # inside
    mask_inside = mask_r_in & mask_phi_in & mask_z_in
    # else:
    #     mask_inside = np.full(len(observers), in_out == "inside")
    #     mask_not_on_surf = np.full(len(observers), True)
    # WARNING @alex
    #   1. inside and not_on_surface are not the same! Can't just put to true.

    # return 0 when all points are on surface
    if not np.any(mask_not_on_surf):
        return BHJM * 0

    if field == "J":
        BHJM[~mask_inside] = 0
        return BHJM

    if field == "M":
        BHJM[~mask_inside] = 0
        return BHJM / MU0

    BHJM *= 0

    # redefine input if there are some surface-points -------------------------
    pol = polarization[mask_not_on_surf]
    dim = dim[mask_not_on_surf]
    pos_obs_cy = pos_obs_cy[mask_not_on_surf]
    phi = phi[mask_not_on_surf]

    # transform mag to spherical CS -----------------------------------------
    m = np.sqrt(pol[:, 0] ** 2 + pol[:, 1] ** 2 + pol[:, 2] ** 2) / MU0  # J -> M
    phi_m = np.arctan2(pol[:, 1], pol[:, 0])
    th_m = np.arctan2(np.sqrt(pol[:, 0] ** 2 + pol[:, 1] ** 2), pol[:, 2])
    mag_sph = np.concatenate(((m,), (phi_m,), (th_m,)), axis=0).T

    # compute H and transform to cart CS -------------------------------------
    H_cy = magnet_cylinder_segment_Hfield(
        magnetizations=mag_sph, dimensions=dim, observers=pos_obs_cy
    )
    Hr, Hphi, Hz = H_cy.T
    Hx = Hr * np.cos(phi) - Hphi * np.sin(phi)
    Hy = Hr * np.sin(phi) + Hphi * np.cos(phi)
    BHJM[mask_not_on_surf] = np.concatenate(((Hx,), (Hy,), (Hz,)), axis=0).T

    if field == "H":
        return BHJM

    if field == "B":
        BHJM *= MU0
        BHJM[mask_inside] += polarization[mask_inside]
        BHJM[~mask_not_on_surf] *= 0
        return BHJM

    msg = f"`output_field_type` must be one of ('B', 'H', 'M', 'J'), got {field!r}"
    raise ValueError(msg)  # pragma: no cover

    # return convert_HBMJ(
    #     output_field_type=field,
    #     polarization=polarization,
    #     input_field_type="H",
    #     field_values=H_all,
    #     mask_inside=mask_inside & mask_not_on_surf,
    # )


def BHJM_cylinder_segment_internal(
    field: str,
    observers: np.ndarray,
    polarization: np.ndarray,
    dimension: np.ndarray,
) -> np.ndarray:
    """
    internal version of BHJM_cylinder_segment used for object oriented interface.

    Falls back to magnet_cylinder_field whenever the section angles describe the full
    360° cylinder.
    """
    xp = array_namespace(observers, polarization, dimension)

    BHfinal = xp.zeros_like(observers, dtype=float)

    r1, r2, h, phi1, phi2 = dimension.T

    # case1: segment
    mask1 = (phi2 - phi1) < 360

    BHfinal[mask1] = BHJM_cylinder_segment(
        field=field,
        observers=observers[mask1],
        polarization=polarization[mask1],
        dimension=dimension[mask1],
    )

    # case2: full cylinder
    mask1x = ~mask1
    BHfinal[mask1x] = BHJM_magnet_cylinder(
        field=field,
        observers=observers[mask1x],
        polarization=polarization[mask1x],
        dimension=xp.c_[2 * r2[mask1x], h[mask1x]],
    )

    # case2a: hollow cylinder <- should be vectorized together with above
    mask2 = (r1 != 0) & mask1x
    BHfinal[mask2] -= BHJM_magnet_cylinder(
        field=field,
        observers=observers[mask2],
        polarization=polarization[mask2],
        dimension=xp.c_[2 * r1[mask2], h[mask2]],
    )

    return BHfinal


def BHJM_cylinder_segment(
    field: str,
    observers: np.ndarray,
    dimension: np.ndarray,
    polarization: np.ndarray,
) -> np.ndarray:
    """
    - translate cylinder segment field to BHJM
    - special cases catching
    """
    check_field_input(field)

    BHJM = polarization.astype(float)

    r1, r2, h, phi1, phi2 = dimension.T
    r1 = abs(r1)
    r2 = abs(r2)
    h = abs(h)
    z1, z2 = -h / 2, h / 2

    # transform dim deg->rad
    phi1 = phi1 / 180 * np.pi
    phi2 = phi2 / 180 * np.pi
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T

    # transform obs_pos to Cy CS --------------------------------------------
    x, y, z = observers.T
    r, phi = np.sqrt(x**2 + y**2), np.arctan2(y, x)
    pos_obs_cy = np.concatenate(((r,), (phi,), (z,)), axis=0).T

    # determine when points lie inside and on surface of magnet -------------

    # mask_inside = None
    # if in_out == "auto":
    # phip1 in [-2pi,0], phio2 in [0,2pi]
    phio1 = phi
    phio2 = phi - np.sign(phi) * 2 * np.pi

    # phi=phi1, phi=phi2
    mask_phi1 = close(phio1, phi1) | close(phio2, phi1)
    mask_phi2 = close(phio1, phi2) | close(phio2, phi2)

    # r, phi ,z lies in-between, avoid numerical fluctuations (e.g. due to rotations) by including 1e-14
    mask_r_in = (r1 - 1e-14 < r) & (r < r2 + 1e-14)
    mask_phi_in = (np.sign(phio1 - phi1) != np.sign(phio1 - phi2)) | (
        np.sign(phio2 - phi1) != np.sign(phio2 - phi2)
    )
    mask_z_in = (z1 - 1e-14 < z) & (z < z2 + 1e-14)

    # on surface
    mask_surf_z = (
        (close(z, z1) | close(z, z2)) & mask_phi_in & mask_r_in
    )  # top / bottom
    mask_surf_r = (close(r, r1) | close(r, r2)) & mask_phi_in & mask_z_in  # in / out
    mask_surf_phi = (mask_phi1 | mask_phi2) & mask_r_in & mask_z_in  # in / out
    mask_not_on_surf = ~(mask_surf_z | mask_surf_r | mask_surf_phi)

    # inside
    mask_inside = mask_r_in & mask_phi_in & mask_z_in
    # else:
    #     mask_inside = np.full(len(observers), in_out == "inside")
    #     mask_not_on_surf = np.full(len(observers), True)
    # WARNING @alex
    #   1. inside and not_on_surface are not the same! Can't just put to true.

    # return 0 when all points are on surface
    if not np.any(mask_not_on_surf):
        return BHJM * 0

    if field == "J":
        BHJM[~mask_inside] = 0
        return BHJM

    if field == "M":
        BHJM[~mask_inside] = 0
        return BHJM / MU0

    BHJM *= 0

    # redefine input if there are some surface-points -------------------------
    pol = polarization[mask_not_on_surf]
    dim = dim[mask_not_on_surf]
    pos_obs_cy = pos_obs_cy[mask_not_on_surf]
    phi = phi[mask_not_on_surf]

    # transform mag to spherical CS -----------------------------------------
    m = np.sqrt(pol[:, 0] ** 2 + pol[:, 1] ** 2 + pol[:, 2] ** 2) / MU0  # J -> M
    phi_m = np.arctan2(pol[:, 1], pol[:, 0])
    th_m = np.arctan2(np.sqrt(pol[:, 0] ** 2 + pol[:, 1] ** 2), pol[:, 2])
    mag_sph = np.concatenate(((m,), (phi_m,), (th_m,)), axis=0).T

    # compute H and transform to cart CS -------------------------------------
    H_cy = magnet_cylinder_segment_Hfield(
        magnetizations=mag_sph, dimensions=dim, observers=pos_obs_cy
    )
    Hr, Hphi, Hz = H_cy.T
    Hx = Hr * np.cos(phi) - Hphi * np.sin(phi)
    Hy = Hr * np.sin(phi) + Hphi * np.cos(phi)
    BHJM[mask_not_on_surf] = np.concatenate(((Hx,), (Hy,), (Hz,)), axis=0).T

    if field == "H":
        return BHJM

    if field == "B":
        BHJM *= MU0
        BHJM[mask_inside] += polarization[mask_inside]
        BHJM[~mask_not_on_surf] *= 0
        return BHJM

    msg = f"`output_field_type` must be one of ('B', 'H', 'M', 'J'), got {field!r}"
    raise ValueError(msg)  # pragma: no cover

    # return convert_HBMJ(
    #     output_field_type=field,
    #     polarization=polarization,
    #     input_field_type="H",
    #     field_values=H_all,
    #     mask_inside=mask_inside & mask_not_on_surf,
    # )
