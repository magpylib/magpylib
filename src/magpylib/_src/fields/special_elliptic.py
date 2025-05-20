from __future__ import annotations

import scipy as sp
from typing import Any

import array_api_extra as xpx

type Array = Any


def ellipeinc(phi: Array, m: Array):
    return xpx.lazy_apply(sp.special.ellipeinc, phi, m, as_numpy=True)


def ellipkinc(phi: Array, m: Array):
    return xpx.lazy_apply(sp.special.ellipkinc, phi, m, as_numpy=True)


def ellipe(m: Array):
    return xpx.lazy_apply(sp.special.ellipe, m, as_numpy=True)


def ellipk(m: Array):
    return xpx.lazy_apply(sp.special.ellipk, m, as_numpy=True)
