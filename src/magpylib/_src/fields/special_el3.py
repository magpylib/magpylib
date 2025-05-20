from __future__ import annotations

import array_api_extra as xpx
import numpy as np
from array_api_compat import array_namespace

from magpylib._src.fields.special_cel import cel

# ruff: noqa: E741  # Avoid ambiguity with variable names


def el30(x, kc, p):
    """
    incomplete elliptic integral

    el3 from Numerical Calculation of Elliptic Integrals and Elliptic Functions
    ROLAND BULIRSCH Numerische Mathematik 7, 78--90 (t965)
    """
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    # pylint: disable=consider-swap-variables

    if x == 0:
        return 0.0

    ye = 0.0
    k = km2 = l = m = n = 0
    bo = bk = False

    D = 8
    CA = 10.0 ** (-D / 2)
    CB = 10.0 ** (-D - 2)
    ND = D - 2
    ln2 = np.log(2)
    ra, rb, rr = np.zeros((3, ND - 1))
    hh = x * x
    f = p * hh
    s = CA / (1 + np.abs(x)) if (kc == 0.0) else kc
    t = s * s
    pm = 0.5 * t
    e = hh * t
    z = np.abs(f)
    r = np.abs(p)
    h = 1.0 + hh
    if e < 0.1 and z < 0.1 and t < 1 and r < 1:
        for k in range(2, ND + 1):  # k is also a variable !!!
            km2 = int(k - 2)
            rb[km2] = 0.5 / k
            ra[km2] = 1.0 - rb[km2]
        zd = 0.5 / (ND + 1)
        s = p + pm
        for k in range(2, ND + 1):
            km2 = int(k - 2)
            rr[km2] = s
            pm = pm * t * ra[km2]
            s = s * p + pm
        u = s * zd
        s = u
        bo = False
        for k in range(ND, 1, -1):
            km2 = int(k - 2)
            u = u + (rr[km2] - u) * rb[km2]
            bo = not bo
            v = -u if bo else u
            s = s * hh + v
        if bo:
            s = -s
        u = (u + 1) * 0.5
        return (u - s * h) * np.sqrt(h) * x + u * np.asinh(x)

    w = 1 + f
    if w == 0:
        msg = "FAIL"
        raise RuntimeError(msg)
    p1 = CB / hh if p == 0.0 else p
    s = np.abs(s)
    y = np.abs(x)
    g = p1 - 1.0
    if g == 0.0:
        g = CB

    f = p1 - t
    if f == 0.0:
        f = CB * t
    am = 1.0 - t
    ap = 1.0 + e
    r = p1 * h
    fa = g / (f * p1)
    bo = fa > 0.0
    fa = np.abs(fa)
    pz = np.abs(g * f)
    de = np.sqrt(pz)
    q = np.sqrt(np.abs(p1))
    pm = min(0.5, pm)
    pm = p1 - pm

    if pm >= 0.0:
        u = np.sqrt(r * ap)
        v = y * de
        if g < 0.0:
            v = -v
        d = 1.0 / q
        c = 1.0
    else:
        u = np.sqrt(h * ap * pz)
        ye = y * q
        v = am * ye
        q = -de / g
        d = -am / de
        c = 0.0
        pz = ap - r

    if bo:
        r = v / u
        z = 1.0
        k = 1
        if pm < 0.0:
            h = y * np.sqrt(h / (ap * fa))
            h = 1.0 / h - h
            z = h - r - r
            r = 2.0 + r * h
            if r == 0.0:
                r = CB
            if z == 0.0:
                z = h * CB
            z = r = r / z
            w = pz
        u = u / w
        v = v / w
    else:
        t = u + np.abs(v)
        bk = True
        if p1 < 0.0:
            de = v / pz
            ye = u * ye
            ye = ye + ye
            u = t / pz
            v = (-f - g * e) / t
            t = pz * np.abs(w)
            z = (hh * r * f - g * ap + ye) / t
            ye = ye / t
        else:
            de = v / w
            ye = 0.0
            u = (e + p1) / t
            v = t / w
            z = 1.0
        if s > 1.0:
            h = u
            u = v
            v = h
    y = 1.0 / y
    e = s
    n = 1
    t = 1.0
    l = 0
    m = 0
    while True:
        y = y - e / y
        if y == 0.0:
            y = np.sqrt(e) * CB
        f = c
        c = d / q + c
        g = e / q
        d = f * g + d
        d = d + d
        q = g + q
        g = t
        t = s + t
        n = n + n
        m = m + m

        if bo:
            if z < 0:
                m = k + m
            k = np.sign(r)
            h = e / (u * u + v * v)
            u = u * (1.0 + h)
            v = v * (1.0 - h)
        else:
            r = u / v
            h = z * r
            z = h * z
            hh = e / v
            if bk:
                de = de / u
                ye = ye * (h + 1.0 / h) + de * (1.0 + r)
                de = de * (u - hh)
                bk = np.abs(ye) < 1.0
            else:
                b_crack = ln2
                a_crack = np.log(x)
                k = int(a_crack / b_crack) + 1
                a_crack = a_crack - k * b_crack
                m = np.exp(a_crack)
                m = m + k

        if np.abs(g - s) > CA * g:
            if bo:
                g = (1.0 / r - r) * 0.5
                hh = u + v * g
                h = g * u - v
                if hh == 0.0:
                    hh = u * CB
                if h == 0.0:
                    h = v * CB
                z = r * h
                r = hh / h
            else:
                u = u + e / u
                v = v + hh
            s = np.sqrt(e)
            s = s + s
            e = s * t
            l = l + l
            if y < 0.0:
                l += 1
        else:
            break
    if y < 0.0:
        l += 1
    e = np.atan(t / y) + np.pi * l
    e = e * (c * t + d) / (t * (t + q))
    if bo:
        h = v / (t + u)
        z = 1.0 - r * h
        h = r + h
        if z == 0.0:
            z = CB
        if z < 0.0:
            m = m + np.sign(h)
        s = np.atan(h / z) + m * np.pi
    else:
        s = np.asinh(ye) if bk else np.log(z) + m * ln2
        s = s * 0.5
    e = (e + np.sqrt(fa) * s) / n
    return e if (x > 0.0) else -e


def el3v(x, kc, p):
    """
    vectorized form of el3

    el3 from Numerical Calculation of Elliptic Integrals and Elliptic Functions
    ROLAND BULIRSCH Numerische Mathematik 7, 78--90 (t965)

    for large N ~ 20x faster than loop
    for N = 10 same speed
    for N = 1 10x slower
    """
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements

    xp = array_namespace(x, kc, p)
    nnn0 = x.shape[0]
    result0 = xp.zeros(nnn0)

    # return 0 when mask0
    mask0 = x != 0
    x = x[mask0]
    kc = kc[mask0]
    p = p[mask0]

    nnn = x.shape[0]
    result = xp.zeros(nnn)

    D = 8
    CA = 10.0 ** (-D / 2)
    CB = 10.0 ** (-D - 2)
    ND = D - 2
    ln2 = xp.log(xp.asarray(2.0))
    hh = x * x
    f = p * hh

    s = xp.zeros(nnn)
    mask1 = kc == 0
    s = xpx.apply_where(
        mask1,
        (x, kc),
        lambda x, kc: CA / (1 + xp.abs(x)),
        lambda x, kc: kc,
    )
    t = s * s
    pm = 0.5 * t
    e = hh * t
    z = xp.abs(f)
    r = xp.abs(p)
    h = 1.0 + hh
    mask2 = (e < 0.1) & (z < 0.1) & (t < 1) & (r < 1)
    if xp.any(mask2):
        R = xp.zeros((3, ND - 1, xp.count_nonzero(mask2)))
        ra, rb, rr = (R[i] for i in range(3))
        px, pmx, tx, hhx, hx, xx = (
            p[mask2],
            pm[mask2],
            t[mask2],
            hh[mask2],
            h[mask2],
            x[mask2],
        )
        for k in range(2, ND + 1):
            km2 = int(k - 2)
            rb[km2] = 0.5 / k
            ra[km2] = 1.0 - rb[km2]
        zd = 0.5 / (ND + 1)
        sx = px + pmx
        for k in range(2, ND + 1):
            km2 = int(k - 2)
            rr[km2] = sx
            pmx = pmx * tx * ra[km2]
            sx = sx * px + pmx
        ux = sx * zd
        sx = xp.asarray(ux, copy=True)
        bo = False
        for k in range(ND, 1, -1):
            km2 = int(k - 2)
            ux = ux + (rr[km2] - ux) * rb[km2]
            bo = not bo
            vx = -ux if bo else ux
            sx = sx * hhx + vx
        if bo:
            sx = -sx
        ux = (ux + 1) * 0.5
        result = xpx.at(result)[mask2].set(
            (ux - sx * hx) * xp.sqrt(hx) * xx + ux * xp.asinh(xx)
        )

    mask2x = ~mask2
    p, pm, t, hh, h, x = (
        p[mask2x],
        pm[mask2x],
        t[mask2x],
        hh[mask2x],
        h[mask2x],
        x[mask2x],
    )
    f, e, z, s = f[mask2x], e[mask2x], z[mask2x], s[mask2x]
    ye_k = xp.zeros((2, (p.shape[0])))
    ye, k = (ye_k[i, ...] for i in range(2))
    bk = xp.zeros((p.shape[0]), dtype=xp.bool)

    w = 1 + f
    if xp.any(w == 0):
        msg = "FAIL"
        raise RuntimeError(msg)

    p1 = xp.asarray(p, copy=True)
    mask3 = p == 0
    p1 = xpx.apply_where(mask3, (hh, p1), lambda hh, p1: CB / hh, lambda hh, p1: p1)
    s = xp.abs(s)
    y = xp.abs(x)
    g = p1 - 1.0
    g = xpx.at(g)[g == 0].set(CB)
    f = p1 - t
    mask4 = f == 0
    f = xp.where(mask4, CB * t, f)
    am = 1.0 - t
    ap = 1.0 + e
    r = p1 * h
    fa = g / (f * p1)
    bo = fa > 0.0
    fa = xp.abs(fa)
    pz = xp.abs(g * f)
    de = xp.sqrt(pz)
    q = xp.sqrt(xp.abs(p1))
    mask5 = pm > 0.5
    pm = xpx.at(pm)[mask5].set(0.5)
    pm = p1 - pm

    uvdc = xp.zeros((4, (pm.shape[0])))
    u, v, d, c = (uvdc[i, ...] for i in range(4))

    mask6 = pm >= 0.0
    if xp.any(mask6):
        u = xp.where(mask6, xp.sqrt(r * ap), u)
        v = xp.where(mask6, y * de * xp.sign(g), v)
        d = xp.where(mask6, 1 / q, d)
        c = xp.where(mask6, 1.0, c)

    mask6x = ~mask6
    if xp.any(mask6x):
        u = xp.where(mask6x, xp.sqrt(h * ap * pz), u)
        ye = xp.where(mask6x, y * q, ye)
        v = xp.where(mask6x, am * ye, v)
        q = xp.where(mask6x, -de / g, q)
        d = xp.where(mask6x, -am / de, d)
        c = xpx.at(c)[mask6x].set(0)
        pz = xp.where(mask6x, ap - r, pz)

    if xp.any(bo):
        r = xp.where(bo, v / u, r)
        z = xpx.at(z)[bo].set(1)
        k = xpx.at(k)[bo].set(1)

        mask7 = bo & (pm < 0)
        if xp.any(mask7):
            h = xp.where(mask7, y * xp.sqrt(h / (ap * fa)), h)
            h = xp.where(mask7, 1 / h - h, h)
            z = xp.where(mask7, h - 2 * r, z)
            r = xp.where(mask7, 2 + r * h, r)

            mask7a = mask7 & (r == 0)
            r = xpx.at(r)[mask7a].set(CB)

            mask7b = mask7 & (z == 0)
            z = xp.where(mask7b, h * CB, z)

            z = xp.where(mask7, r / z, z)
            r = xp.where(mask7, z, r)
            w = xp.where(mask7, pz, w)

        u = xp.where(bo, u / w, u)
        v = xp.where(bo, v / w, v)

    box = ~bo
    if xp.any(box):
        t = xp.where(box, u + xp.abs(v), t)
        bk = xpx.at(bk)[box].set(True)

        mask8 = box & (p1 < 0)
        if xp.any(mask8):
            de = xp.where(mask8, v / pz, de)
            ye = xp.where(mask8, u * ye, ye)
            ye = xp.where(mask8, 2 * ye, ye)
            u = xp.where(mask8, t / pz, u)
            v = xp.where(mask8, (-f - g * e) / t, v)
            t = xp.where(mask8, pz * xp.abs(w), t)
            z = xp.where(
                mask8,
                (hh * r * f - g * ap + ye) / t,
                z,
            )
            ye = xp.where(mask8, ye / t, ye)

        mask8x = box & (p1 >= 0)
        if xp.any(mask8x):
            de = xp.where(mask8x, v / w, de)
            ye = xpx.at(ye).set(0)
            u = xp.where(mask8x, (e + p1) / t, u)
            v = xp.where(mask8x, t / w, v)
            z = xpx.at(z)[mask8x].set(1.0)

        mask9 = box & (s > 1)
        if xp.any(mask9):
            h = xp.where(mask9, u, h)
            u = xp.where(mask9, v, u)
            v = xp.where(mask9, h, v)

    y = 1 / y
    e = xp.asarray(s, copy=True)
    nt = xp.ones((2, (p.shape[0])))
    n, t = (nt[i, ...] for i in range(2))
    ml = xp.zeros((2, (p.shape[0])))
    m, l = (ml[i, ...] for i in range(2))

    mask10 = xp.ones(
        (p.shape[0]), dtype=xp.bool
    )  # dynamic mask, changed in each loop iteration
    while xp.any(mask10):
        y = xp.where(mask10, y - e / y, y)

        mask11 = mask10 & (y == 0.0)
        y = xp.where(mask11, xp.sqrt(e) * CB, y)

        f = xp.where(mask10, c, f)
        c = xp.where(mask10, d / q + c, c)
        g = xp.where(mask10, e / q, g)
        d = xp.where(mask10, f * g + d, d)
        d = xp.where(mask10, 2 * d, d)
        q = xp.where(mask10, g + q, q)
        g = xp.where(mask10, t, g)
        t = xp.where(mask10, s + t, t)
        n = xp.where(mask10, 2 * n, n)
        m = xp.where(mask10, 2 * m, m)
        bo10 = mask10 & bo
        if xp.any(bo10):
            bo10b = bo10 & (z < 0)
            m = xp.where(bo10b, k + m, m)

            k = xp.where(bo10, xp.sign(r), k)
            h = xp.where(bo10, e / (u * u + v * v), h)
            u = xp.where(bo10, u * (1 + h), u)
            v = xp.where(bo10, v * (1 - h), v)

        bo10x = xp.asarray(mask10 & ~bo10, dtype=xp.bool)
        if xp.any(bo10x):
            r = xp.where(bo10x, u / v, r)
            h = xp.where(bo10x, z * r, h)
            z = xp.where(bo10x, h * z, z)
            hh = xp.where(bo10x, e / v, hh)

            bo10x_bk = xp.asarray(bo10x * bk, dtype=bool)  # if bk
            bo10x_bkx = xp.asarray(bo10x * ~bk, dtype=bool)
            if xp.any(bo10x_bk):
                de = xp.where(bo10x_bk, de / u, de)
                ye = xp.where(bo10x_bk, ye * (h + 1 / h) + de * (1 + r), ye)
                de = xp.where(bo10x_bk, de[bo10x_bk] * (u[bo10x_bk] - hh[bo10x_bk]), de)
                bk = xp.where(bo10x_bk, xp.abs(ye[bo10x_bk]) < 1, bk)
            if xp.any(bo10x_bkx):
                a_crack = xp.log(x)
                k = xp.where(bo10x_bkx, xp.astype(a_crack / ln2, xp.int32) + 1, k)
                a_crack = a_crack - k * ln2
                m = xp.where(bo10x_bkx, xp.exp(a_crack), m)
                m = xp.where(bo10x_bkx, m + k, m)

        mask11 = xp.abs(g - s) > CA * g
        if xp.any(mask11):
            bo11 = mask11 & bo
            if xp.any(bo11):
                g = xp.where(bo11, (1 / r - r) * 0.5, g)
                hh = xp.where(bo11, u + v * g, hh)
                h = xp.where(bo11, g * u - v, h)

                bo11b = bo11 & (hh == 0)
                hh = xp.where(bo11b, u * CB, hh)

                bo11c = bo11 & (h == 0)
                h = xp.where(bo11c, v * CB, h)

                z = xp.where(bo11, r * h, z)
                r = xp.where(bo11, hh / h, r)

            bo11x = mask11 & ~bo
            if xp.any(bo11x):
                u = xp.where(bo11x, u + e / u, u)
                v = xp.where(bo11x, v + hh, v)

            s = xp.where(mask11, xp.sqrt(e), s)
            s = xp.where(mask11, 2 * s, s)
            e = xp.where(mask11, s * t, e)
            l = xp.where(mask11, 2 * l, l)

            mask12 = mask11 & (y < 0)
            l = xp.where(mask12, l + 1, l)

        # break off parts that have completed their iteration
        mask10 = mask11

    mask12 = y < 0
    l = xp.where(mask12, l + 1, l)

    e = xp.atan(t / y) + xp.pi * l
    e = e * (c * t + d) / (t * (t + q))

    if xp.any(bo):
        h = xp.where(bo, v / (t + u), h)
        z = xp.where(bo, 1 - r * h, z)
        h = xp.where(bo, r + h, h)

        bob = bo & (z == 0)
        z = xp.where(bob, CB, z)

        boc = bo & (z < 0)
        m = xp.where(boc, m + xp.sign(h), m)

        s = xp.where(bo, xp.atan(h / z) + m * xp.pi, s)

    box = ~bo
    if xp.any(box):
        box_bk = box & bk
        s = xp.where(box_bk, xp.asinh(ye), s)

        box_bkx = box & ~bk
        s = xp.where(box_bkx, xp.log(z) + m * ln2, s)

        s = xp.where(box, s * 0.5, s)
    e = (e + xp.sqrt(fa) * s) / n
    result = xpx.at(result)[~mask2].set(xp.sign(x) * e)

    # include mask0-case
    result0 = xpx.at(result0)[mask0].set(result)

    return result0


def el3(xv: np.ndarray, kcv: np.ndarray, pv: np.ndarray) -> np.ndarray:
    """
    combine vectorized and non-vectorized implementations for improved performance
    """
    xp = array_namespace(xv, kcv, pv)
    n_input = xv.shape[0]

    # if n_input < 10:
    # return xp.asarray([el30(x, kc, p) for x, kc, p in zip(xv, kcv, pv, strict=False)])

    return el3v(xv, kcv, pv)


def el3_angle(phi: np.ndarray, n: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    vectorized implementation of incomplete elliptic integral for
    arbitrary integration boundaries

    there is still a lot to do here !!!!!

    - cel and el3 are not collected !!!!!
        -> collect all cel and el3 and evaluate in one single go !!!

    - its somehow 2x slower than non-vectorized version when N=1 although
        the underlying functions are used in non-vectorized form.
        -> possibly provide a non-vectorized form of this ? (maybe not worth the effort)
    """
    # pylint: disable=too-many-statements
    xp = array_namespace(phi, n, m)
    n_vec = phi.shape[0]
    results = xp.zeros(n_vec)

    kc = xp.sqrt(1 - m)
    p = 1 - n

    D = 8
    n = xp.round(phi / xp.pi)
    phi_red = phi - n * xp.pi

    mask1 = (n <= 0) & (phi_red < -xp.pi / 2)
    mask2 = (n >= 0) & (phi_red > xp.pi / 2)
    if xp.any(mask1):
        n = xp.where(mask1, n - 1, n)
        phi_red = xp.where(mask1, phi_red + xp.pi, phi_red)

    if xp.any(mask2):
        n = xp.where(mask2, n + 1, n)
        phi_red = xp.where(mask2, phi_red - xp.pi, phi_red)

    mask3 = n != 0
    mask3x = ~mask3
    if xp.any(mask3):
        n3, phi3, p3, kc3 = n[mask3], phi[mask3], p[mask3], kc[mask3]
        phi_red3 = phi_red[mask3]

        results3 = xp.zeros(xp.count_nonzero(mask3))
        onez = xp.ones(xp.count_nonzero(mask3))
        cel3_res = cel(kc3, p3, onez, onez)  # 3rd kind cel

        mask3a = phi_red3 > xp.pi / 2 - 10 ** (-D)
        mask3b = phi_red3 < -xp.pi / 2 + 10 ** (-D)
        mask3c = ~mask3a & ~mask3b

        if xp.any(mask3a):
            results3 = xp.where(mask3a, (2 * n3 + 1) * cel3_res, results3)
        if xp.any(mask3b):
            results3 = xp.where(mask3b, (2 * n3 - 1) * cel3_res, results3)
        if xp.any(mask3c):
            results3 = xpx.apply_where(
                mask3c,
                (
                    phi3,
                    kc3,
                    p3,
                    n3,
                    cel3_res,
                    results3,
                ),
                lambda phi3, kc3, p3, n3, cel3_res, results3: el3(xp.tan(phi3), kc3, p3)
                + 2 * n3 * cel3_res,
                lambda phi3, kc3, p3, n3, cel3_res, results3: results3,
            )

        results = xpx.at(results)[mask3].set(results3)

    if xp.any(mask3x):
        phi_red3x = phi_red[mask3x]
        results3x = xp.zeros(xp.count_nonzero(mask3x))
        phi3x, kc3x, p3x = phi[mask3x], kc[mask3x], p[mask3x]

        mask3xa = phi_red3x > xp.pi / 2 - 10 ** (-D)
        mask3xb = phi_red3x < -xp.pi / 2 + 10 ** (-D)
        mask3xc = ~mask3xa & ~mask3xb

        if xp.any(mask3xa):
            onez = xp.ones(xp.count_nonzero(mask3xa))
            results3x = xpx.apply_where(
                mask3xa,
                (kc3x, p3x, results3x),
                lambda kc3x, p3x, results3x: cel(kc3x, p3x, onez, onez),
                lambda kc3x, p3x, results3x: results3x,
            )  # 3rd kind cel
        if xp.any(mask3xb):
            onez = xp.ones(xp.count_nonzero(mask3xb))
            results3x = xp.where(
                mask3xb,
                (kc3x, p3x, results3x),
                lambda kc3x, p3x, results3x: -cel(kc3x, p3x, onez, onez),
                lambda kc3x, p3x, results3x: results3x,
            )  # 3rd kind cel
        if xp.any(mask3xc):
            results3x = xpx.apply_where(
                mask3xc,
                (phi3x, kc3x, p3x, results3x),
                lambda phi3x, kc3x, p3x, results3x: el3(xp.tan(phi3x), kc3x, p3x),
                lambda phi3x, kc3x, p3x, results3x: results3x,
            )

        results = xpx.at(results)[mask3x].set(results3x)

    return results
