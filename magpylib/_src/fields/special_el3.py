import numpy as np

from magpylib._src.fields.special_cel import cel


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
        for k in range(2, ND + 1):  # outch k ist auch eine variable !!!
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
        result = (u - s * h) * np.sqrt(h) * x + u * np.arcsinh(x)
        return result

    w = 1 + f
    if w == 0:
        raise RuntimeError("FAIL")
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
    e = np.arctan(t / y) + np.pi * l
    e = e * (c * t + d) / (t * (t + q))
    if bo:
        h = v / (t + u)
        z = 1.0 - r * h
        h = r + h
        if z == 0.0:
            z = CB
        if z < 0.0:
            m = m + np.sign(h)
        s = np.arctan(h / z) + m * np.pi
    else:
        s = np.arcsinh(ye) if bk else np.log(z) + m * ln2
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

    nnn0 = len(x)
    result0 = np.zeros(nnn0)

    # return 0 when mask0
    mask0 = x != 0
    x = x[mask0]
    kc = kc[mask0]
    p = p[mask0]

    nnn = len(x)
    result = np.zeros(nnn)

    D = 8
    CA = 10.0 ** (-D / 2)
    CB = 10.0 ** (-D - 2)
    ND = D - 2
    ln2 = np.log(2)
    hh = x * x
    f = p * hh

    s = np.zeros(nnn)
    mask1 = kc == 0
    s[mask1] = CA / (1 + np.abs(x[mask1]))
    s[~mask1] = kc[~mask1]
    t = s * s
    pm = 0.5 * t
    e = hh * t
    z = np.abs(f)
    r = np.abs(p)
    h = 1.0 + hh
    mask2 = (e < 0.1) * (z < 0.1) * (t < 1) * (r < 1)
    if any(mask2):
        ra, rb, rr = np.zeros((3, ND - 1, np.sum(mask2)))
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
        sx = np.copy(ux)
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
        result[mask2] = (ux - sx * hx) * np.sqrt(hx) * xx + ux * np.arcsinh(xx)

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
    ye, k = np.zeros((2, len(p)))
    bk = np.zeros(len(p), dtype=bool)

    w = 1 + f
    if np.any(w == 0):
        raise RuntimeError("FAIL")

    p1 = np.copy(p)
    mask3 = p == 0
    p1[mask3] = CB / hh[mask3]
    s = np.abs(s)
    y = np.abs(x)
    g = p1 - 1.0
    g[g == 0] = CB
    f = p1 - t
    mask4 = f == 0
    f[mask4] = CB * t[mask4]
    am = 1.0 - t
    ap = 1.0 + e
    r = p1 * h
    fa = g / (f * p1)
    bo = fa > 0.0
    fa = np.abs(fa)
    pz = np.abs(g * f)
    de = np.sqrt(pz)
    q = np.sqrt(np.abs(p1))
    mask5 = pm > 0.5
    pm[mask5] = 0.5
    pm = p1 - pm

    u, v, d, c = np.zeros((4, len(pm)))

    mask6 = pm >= 0.0
    if np.any(mask6):
        u[mask6] = np.sqrt(r[mask6] * ap[mask6])
        v[mask6] = y[mask6] * de[mask6] * np.sign(g[mask6])
        d[mask6] = 1 / q[mask6]
        c[mask6] = 1.0

    mask6x = ~mask6
    if np.any(mask6x):
        u[mask6x] = np.sqrt(h[mask6x] * ap[mask6x] * pz[mask6x])
        ye[mask6x] = y[mask6x] * q[mask6x]
        v[mask6x] = am[mask6x] * ye[mask6x]
        q[mask6x] = -de[mask6x] / g[mask6x]
        d[mask6x] = -am[mask6x] / de[mask6x]
        c[mask6x] = 0
        pz[mask6x] = ap[mask6x] - r[mask6x]

    if np.any(bo):
        r[bo] = v[bo] / u[bo]
        z[bo] = 1
        k[bo] = 1

        mask7 = bo * (pm < 0)
        if np.any(mask7):
            h[mask7] = y[mask7] * np.sqrt(h[mask7] / (ap[mask7] * fa[mask7]))
            h[mask7] = 1 / h[mask7] - h[mask7]
            z[mask7] = h[mask7] - 2 * r[mask7]
            r[mask7] = 2 + r[mask7] * h[mask7]

            mask7a = mask7 * (r == 0)
            r[mask7a] = CB

            mask7b = mask7 * (z == 0)
            z[mask7b] = h[mask7b] * CB

            z[mask7] = r[mask7] / z[mask7]
            r[mask7] = np.copy(z[mask7])
            w[mask7] = pz[mask7]

        u[bo] = u[bo] / w[bo]
        v[bo] = v[bo] / w[bo]

    box = ~bo
    if np.any(box):
        t[box] = u[box] + np.abs(v[box])
        bk[box] = True

        mask8 = box * (p1 < 0)
        if np.any(mask8):
            de[mask8] = v[mask8] / pz[mask8]
            ye[mask8] = u[mask8] * ye[mask8]
            ye[mask8] = 2 * ye[mask8]
            u[mask8] = t[mask8] / pz[mask8]
            v[mask8] = (-f[mask8] - g[mask8] * e[mask8]) / t[mask8]
            t[mask8] = pz[mask8] * np.abs(w[mask8])
            z[mask8] = (
                hh[mask8] * r[mask8] * f[mask8] - g[mask8] * ap[mask8] + ye[mask8]
            ) / t[mask8]
            ye[mask8] = ye[mask8] / t[mask8]

        mask8x = box * (p1 >= 0)
        if np.any(mask8x):
            de[mask8x] = v[mask8x] / w[mask8x]
            ye[mask8x] = 0
            u[mask8x] = (e[mask8x] + p1[mask8x]) / t[mask8x]
            v[mask8x] = t[mask8x] / w[mask8x]
            z[mask8x] = 1.0

        mask9 = box * (s > 1)
        if np.any(mask9):
            h[mask9] = u[mask9]
            u[mask9] = v[mask9]
            v[mask9] = h[mask9]

    y = 1 / y
    e = np.copy(s)
    n, t = np.ones((2, len(p)))
    m, l = np.zeros((2, len(p)))

    mask10 = np.ones(len(p), dtype=bool)  # dynamic mask, changed in each loop iteration
    while np.any(mask10):
        y[mask10] = y[mask10] - e[mask10] / y[mask10]

        mask11 = mask10 * (y == 0.0)
        y[mask11] = np.sqrt(e[mask11]) * CB

        f[mask10] = c[mask10]
        c[mask10] = d[mask10] / q[mask10] + c[mask10]
        g[mask10] = e[mask10] / q[mask10]
        d[mask10] = f[mask10] * g[mask10] + d[mask10]
        d[mask10] = 2 * d[mask10]
        q[mask10] = g[mask10] + q[mask10]
        g[mask10] = t[mask10]
        t[mask10] = s[mask10] + t[mask10]
        n[mask10] = 2 * n[mask10]
        m[mask10] = 2 * m[mask10]
        bo10 = mask10 * bo
        if np.any(bo10):
            bo10b = bo10 * (z < 0)
            m[bo10b] = k[bo10b] + m[bo10b]

            k[bo10] = np.sign(r[bo10])
            h[bo10] = e[bo10] / (u[bo10] * u[bo10] + v[bo10] * v[bo10])
            u[bo10] = u[bo10] * (1 + h[bo10])
            v[bo10] = v[bo10] * (1 - h[bo10])

        bo10x = np.array(mask10 * ~bo10, dtype=bool)
        if np.any(bo10x):
            r[bo10x] = u[bo10x] / v[bo10x]
            h[bo10x] = z[bo10x] * r[bo10x]
            z[bo10x] = h[bo10x] * z[bo10x]
            hh[bo10x] = e[bo10x] / v[bo10x]

            bo10x_bk = np.array(bo10x * bk, dtype=bool)  # if bk
            bo10x_bkx = np.array(bo10x * ~bk, dtype=bool)
            if np.any(bo10x_bk):
                de[bo10x_bk] = de[bo10x_bk] / u[bo10x_bk]
                ye[bo10x_bk] = ye[bo10x_bk] * (h[bo10x_bk] + 1 / h[bo10x_bk]) + de[
                    bo10x_bk
                ] * (1 + r[bo10x_bk])
                de[bo10x_bk] = de[bo10x_bk] * (u[bo10x_bk] - hh[bo10x_bk])
                bk[bo10x_bk] = np.abs(ye[bo10x_bk]) < 1
            if np.any(bo10x_bkx):
                a_crack = np.log(x[bo10x_bkx])
                k[bo10x_bkx] = (a_crack / ln2).astype(int) + 1
                a_crack = a_crack - k[bo10x_bkx] * ln2
                m[bo10x_bkx] = np.exp(a_crack)
                m[bo10x_bkx] = m[bo10x_bkx] + k[bo10x_bkx]

        mask11 = np.abs(g - s) > CA * g
        if np.any(mask11):
            bo11 = mask11 * bo
            if np.any(bo11):
                g[bo11] = (1 / r[bo11] - r[bo11]) * 0.5
                hh[bo11] = u[bo11] + v[bo11] * g[bo11]
                h[bo11] = g[bo11] * u[bo11] - v[bo11]

                bo11b = bo11 * (hh == 0)
                hh[bo11b] = u[bo11b] * CB

                bo11c = bo11 * (h == 0)
                h[bo11c] = v[bo11c] * CB

                z[bo11] = r[bo11] * h[bo11]
                r[bo11] = hh[bo11] / h[bo11]

            bo11x = mask11 * ~bo
            if np.any(bo11x):
                u[bo11x] = u[bo11x] + e[bo11x] / u[bo11x]
                v[bo11x] = v[bo11x] + hh[bo11x]

            s[mask11] = np.sqrt(e[mask11])
            s[mask11] = 2 * s[mask11]
            e[mask11] = s[mask11] * t[mask11]
            l[mask11] = 2 * l[mask11]

            mask12 = mask11 * (y < 0)
            l[mask12] = l[mask12] + 1

        # break off parts that have completed their iteration
        mask10 = mask11

    mask12 = y < 0
    l[mask12] = l[mask12] + 1

    e = np.arctan(t / y) + np.pi * l
    e = e * (c * t + d) / (t * (t + q))

    if np.any(bo):
        h[bo] = v[bo] / (t[bo] + u[bo])
        z[bo] = 1 - r[bo] * h[bo]
        h[bo] = r[bo] + h[bo]

        bob = bo * (z == 0)
        z[bob] = CB

        boc = bo * (z < 0)
        m[boc] = m[boc] + np.sign(h[boc])

        s[bo] = np.arctan(h[bo] / z[bo]) + m[bo] * np.pi

    box = ~bo
    if np.any(box):
        box_bk = box * bk
        s[box_bk] = np.arcsinh(ye[box_bk])

        box_bkx = box * ~bk
        s[box_bkx] = np.log(z[box_bkx]) + m[box_bkx] * ln2

        s[box] = s[box] * 0.5
    e = (e + np.sqrt(fa) * s) / n
    result[~mask2] = np.sign(x) * e

    # include mask0-case
    result0[mask0] = result

    return result0


def el3(xv: np.ndarray, kcv: np.ndarray, pv: np.ndarray) -> np.ndarray:
    """
    combine vectorized and non-vectorized implementations for improved performance
    """
    n_input = len(xv)

    if n_input < 10:
        return np.array([el30(x, kc, p) for x, kc, p in zip(xv, kcv, pv)])

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
    n_vec = len(phi)
    results = np.zeros(n_vec)

    kc = np.sqrt(1 - m)
    p = 1 - n

    D = 8
    n = (phi / np.pi).astype(int)
    phi_red = phi - n * np.pi

    mask1 = (n <= 0) * (phi_red < -np.pi / 2)
    mask2 = (n >= 0) * (phi_red > np.pi / 2)
    if np.any(mask1):
        n[mask1] = n[mask1] - 1
        phi_red[mask1] = phi_red[mask1] + np.pi

    if np.any(mask2):
        n[mask2] = n[mask2] + 1
        phi_red[mask2] = phi_red[mask2] - np.pi

    mask3 = n != 0
    mask3x = ~mask3
    if np.any(mask3):
        n3, phi3, p3, kc3 = n[mask3], phi[mask3], p[mask3], kc[mask3]
        phi_red3 = phi_red[mask3]

        results3 = np.zeros(np.sum(mask3))
        onez = np.ones(np.sum(mask3))
        cel3_res = cel(kc3, p3, onez, onez)  # 3rd kind cel

        mask3a = phi_red3 > np.pi / 2 - 10 ** (-D)
        mask3b = phi_red3 < -np.pi / 2 + 10 ** (-D)
        mask3c = ~mask3a * ~mask3b

        if np.any(mask3a):
            results3[mask3a] = (2 * n3[mask3a] + 1) * cel3_res[mask3a]
        if np.any(mask3b):
            results3[mask3b] = (2 * n3[mask3b] - 1) * cel3_res[mask3b]
        if np.any(mask3c):
            el33_res = el3(np.tan(phi3[mask3c]), kc3[mask3c], p3[mask3c])
            results3[mask3c] = 2 * n3[mask3c] * cel3_res[mask3c] + el33_res

        results[mask3] = results3

    if np.any(mask3x):
        phi_red3x = phi_red[mask3x]
        results3x = np.zeros(np.sum(mask3x))
        phi3x, kc3x, p3x = phi[mask3x], kc[mask3x], p[mask3x]

        mask3xa = phi_red3x > np.pi / 2 - 10 ** (-D)
        mask3xb = phi_red3x < -np.pi / 2 + 10 ** (-D)
        mask3xc = ~mask3xa * ~mask3xb

        if np.any(mask3xa):
            onez = np.ones(np.sum(mask3xa))
            results3x[mask3xa] = cel(
                kc3x[mask3xa], p3x[mask3xa], onez, onez
            )  # 3rd kind cel
        if np.any(mask3xb):
            onez = np.ones(np.sum(mask3xb))
            results3x[mask3xb] = -cel(
                kc3x[mask3xb], p3x[mask3xb], onez, onez
            )  # 3rd kind cel
        if np.any(mask3xc):
            results3x[mask3xc] = el3(
                np.tan(phi3x[mask3xc]), kc3x[mask3xc], p3x[mask3xc]
            )

        results[mask3x] = results3x

    return results
